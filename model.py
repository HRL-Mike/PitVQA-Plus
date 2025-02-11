import math
import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import ViTModel, BlipTextModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vector_MoLoRA(nn.Module):
    def __init__(self, base_layer, lora_rank, alpha, mora_rank, dropout):
        super().__init__()
        self.base_layer = base_layer  # original c_atten layer
        self.lora_r = lora_rank
        self.scaling = alpha / lora_rank
        self.mora_r = mora_rank  # one of mora rank elements in the list
        self.in_features = base_layer.weight.shape[0]  # 768
        self.out_features = base_layer.weight.shape[1]  # 2304

        # dropout
        self.dropout = nn.ModuleDict({
            'default': nn.Dropout(p=dropout)
        })

        # MoRA A and B matrices
        self.mora_A = nn.ModuleDict({
            'default': nn.Conv1d(self.mora_r, self.mora_r, bias=False, kernel_size=1)
        })
        # zero init for mora_A
        nn.init.zeros_(self.mora_A['default'].weight)
        self.mora_B = self.mora_A  # not for use

        # LoRA C and D matrices
        self.lora_C = nn.ModuleDict({
            'default': nn.Linear(self.in_features, self.lora_r, bias=False)
        })
        self.lora_D = nn.ModuleDict({
            'default': nn.Linear(self.lora_r, self.out_features, bias=False)
        })
        # Kaiming init for lora_C
        nn.init.kaiming_uniform_(self.lora_C['default'].weight, a=math.sqrt(5))
        # zero init for lora_D
        nn.init.zeros_(self.lora_D['default'].weight)

        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})

    def forward(self, x): 
        # Original output
        result = self.base_layer(x) 
        x = self.dropout['default'](x) 

        '''Process with LoRA'''
        lora_out_x = self.lora_D['default'](
            self.lora_C['default'](x)
        ) 

        '''Process with MoRA'''
        # apply compression before lora_A
        in_f, out_f = self.in_features, self.out_features
        r = self.mora_r

        sum_inter = in_f // r
        rb1 = in_f // r if in_f % r == 0 else in_f // r + 1

        if in_f % r != 0:
            pad_size = r - in_f % r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, r) 

        if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, r, 2).float() / r))
            t = torch.arange(rb1)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
            self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)

        rh_in_x = torch.cat((-in_x[..., r // 2:], in_x[..., :r // 2]), dim=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin 

        # rearrange features
        b, c, h, w = in_x.shape  # [16, 32, 3, 256]
        in_x = in_x.view(b, c*h, w).permute(0, 2, 1) 

        # apply mora_A
        mora_out_x = self.mora_A['default'](in_x) 
        mora_out_x = mora_out_x.view(b, c, h, w) 

        # apply decompression after lora_A
        mora_out_x = mora_out_x.view(*x.shape[:-1], -1)[..., :out_f] 
        if mora_out_x.shape[-1] < out_f:
            repeat_time = out_f // mora_out_x.shape[-1]
            if out_f % mora_out_x.shape[-1] != 0:
                repeat_time += 1
            mora_out_x = torch.cat([mora_out_x] * repeat_time, dim=-1)[..., :out_f] 

        return result + mora_out_x + (self.scaling * lora_out_x)

class VectorMoLoRAInitializer:
    def __init__(self, model, mora_base_rank=8, mora_rank_coefficients=None, lora_rank=None, lora_alpha=None, dropout=0.1):
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.mora_base_rank = mora_base_rank
        self.dropout = dropout
        self.mora_rank_coefficients = mora_rank_coefficients

    def calculate_mora_ranks(self):
        return [self.mora_base_rank * coeff for coeff in self.mora_rank_coefficients]

    def initialize(self):
        mora_ranks = self.calculate_mora_ranks()

        for param in self.model.transformer.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.model.transformer.h):
            layer_w_qkv = blk.attn.c_attn
            current_mora_rank = mora_ranks[t_layer_i]
            current_lora_rank = self.lora_rank[t_layer_i]
            current_alpha = self.lora_alpha[t_layer_i]
            blk.attn.c_attn = Vector_MoLoRA(base_layer=layer_w_qkv, lora_rank=current_lora_rank, alpha=current_alpha,
                                            mora_rank=current_mora_rank, dropout=self.dropout)

        print("Vector MoRA params initialized!")
        return self.model

class PitVQAGen(nn.Module):
    def __init__(self, mora_base_rank=8, mora_rank_coefficients=None, lora_rank=None, lora_alpha=None, dropout=0.1):
        super(PitVQAGen, self).__init__()

        if mora_rank_coefficients is None or lora_rank is None or lora_alpha is None:
            print('Wrong hyperparameters.')

        # visual encoder
        model_name = "google/vit-base-patch16-224-in21k"
        self.visual_encoder = ViTModel.from_pretrained(model_name)

        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder
        self.text_encoder = BlipTextModel.from_pretrained("Salesforce/blip-vqa-base")
        # copy weights
        original_weights = self.text_encoder.embeddings.word_embeddings.weight.data
        new_vocab_size = len(self.tokenizer)
        embedding_dim = self.text_encoder.embeddings.word_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        original_vocab_size = original_weights.shape[0]
        new_embeddings.weight.data[:original_vocab_size] = original_weights
        self.text_encoder.embeddings.word_embeddings = new_embeddings

        # gpt2 decoder
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt = VectorMoLoRAInitializer(self.gpt, mora_base_rank=mora_base_rank,
                                           mora_rank_coefficients=mora_rank_coefficients,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                                           dropout=dropout).initialize()

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        # visual encoder
        image = image.to(device)
        image_embeds = self.visual_encoder(image).last_hidden_state 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) 

        # multimodal encoder
        text_output = self.text_encoder(input_ids=qa_inputs_ids,
                                        attention_mask=qa_att_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state 

        # text decoder
        gpt_output = self.gpt(inputs_embeds=text_embeds,
                              encoder_attention_mask=qa_att_mask) 
        return gpt_output.logits
