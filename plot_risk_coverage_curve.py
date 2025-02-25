import os
import random

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import evaluate
import time
from nltk.translate.bleu_score import corpus_bleu

from model import PitVQAPlus
from dataloader import PitVQASentence

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    answers = []
    uncertainties = [[] for _ in range(len(questions))]  # track uncertainty until EOS per sample
    batch_size = len(questions)

    model.eval()
    with torch.no_grad():
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
        prompt_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding='longest',
            add_special_tokens=False
        )

        # prepare model inputs
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_length), device=device)

        orig_length = prompt_inputs['input_ids'].size(1)
        padded_input_ids[:, :orig_length] = prompt_inputs['input_ids']
        padded_attention_mask[:, :orig_length] = prompt_inputs['attention_mask']
        images = images.to(device)

        # initialize tensors to store generated tokens
        only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        # track which sequences have finished generating 
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # record each sample length (number of non-eos tokens)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(max_length):
            max_valid_lengths = max(valid_lengths).item()
            if max_valid_lengths == 65:  # early stop
                break

            logits = model(
                image=images, 
                qa_inputs_ids=padded_input_ids[:, :max_valid_lengths], 
                qa_att_mask=padded_attention_mask[:, :max_valid_lengths]
            )

            # get next token id
            last_valid_logits = logits[batch_indices, valid_lengths-1, :]
            next_token_ids = torch.argmax(last_valid_logits, dim=-1)

            # get probability for prediction
            probs = F.softmax(last_valid_logits, dim=-1)
            # compute entropy for uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Add small value to avoid log(0)
            # track uncertainty until EOS token for each sample
            for i in range(batch_size):
                if not finished[i]:  # only store uncertainty for unfinished samples
                    uncertainties[i].append(entropy[i].item())

            # check EOS
            is_eos = (next_token_ids == tokenizer.eos_token_id)
            finished = finished | is_eos

            # insert next token at valid_lengths position
            padded_input_ids[batch_indices, valid_lengths] = next_token_ids
            padded_attention_mask[batch_indices, valid_lengths] = 1
            valid_lengths += 1

            # append the selected tokens to the generated_ids
            only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1) 

            # if all sequences have finished, exit early
            if finished.all():
                break

        # decode the generated tokens into strings
        generated_ids_cpu = only_answer_ids.cpu().tolist()  # Move to CPU and convert to list for processing
        for i in range(batch_size):
            try:  # find the first eos_token_id to truncate the answer
                eos_index = generated_ids_cpu[i].index(tokenizer.eos_token_id)
                answer_ids = generated_ids_cpu[i][:eos_index]
            except ValueError:  # if eos_token_id is not found, use all generated tokens
                answer_ids = generated_ids_cpu[i]

            # decode the token IDs to a string, skipping special tokens
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)

    # compute per-sample (sentence) uncertainty by averaging entropies over all generated tokens for that sample
    samplewise_unc = [sum(unc) / len(unc) if len(unc) > 0 else 0.0 for unc in uncertainties]
    return answers, samplewise_unc


def validate(args, val_loader, model, tokenizer, device):
    references = []
    hypotheses = []
    samplewise_unc_all = []

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            images = images.to(device)
            generated_answers, samplewise_unc = batch_greedy_search(
                images,
                questions,
                model,
                tokenizer,
                max_length=args.seq_length,
                device=device
            )

            references.extend(answers)
            hypotheses.extend(generated_answers)
            samplewise_unc_all.extend(samplewise_unc)
    return references, hypotheses, samplewise_unc_all


def get_nlp_metrics(references, hypotheses):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    nltk_references = [[ref.split()] for ref in references]
    nltk_hypotheses = [hyp.split() for hyp in hypotheses]

    metrics = {}
    metrics["Bleu_1"] = corpus_bleu(nltk_references, nltk_hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
    metrics["Bleu_2"] = corpus_bleu(nltk_references, nltk_hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
    metrics["Bleu_3"] = corpus_bleu(nltk_references, nltk_hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
    metrics["Bleu_4"] = corpus_bleu(nltk_references, nltk_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    print("NLTK results: ")
    print(f"BLEU-1: {metrics['Bleu_1']:.6f} BLEU-2: {metrics['Bleu_2']:.6f} "
          f"BLEU-3: {metrics['Bleu_3']:.6f} BLEU-4: {metrics['Bleu_4']:.6f}")

    # compute HF metrics
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)

    print("HuggingFace Metrics Results:")
    print(f"Rouge-1: {results_rouge['rouge1']:.6f}")
    print(f"Rouge-L: {results_rouge['rougeL']:.6f}")
    print(f"Meteor: {results_meteor['meteor']:.6f}")

    samplewise_rouge = rouge.compute(predictions=hypotheses, references=references, use_aggregator=False)
    return samplewise_rouge['rougeL']

class HyperPara:
    def __init__(self):
        self.seq_length = 100
        self.workers = 8
        self.batch_size = 128

if __name__ == '__main__':
    
    random_seed = 42
    seed_everything(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get weights path
    vec_weight_path = '/path/to/weights/vec_mlr_st_saved_weights_Best.pth.tar'
    print(f'vec_weight_path: {vec_weight_path}')

    # prepare data
    args = HyperPara()
    val_seq = ['02', '06', '12', '13', '24']

    folder_head = r'/SAN/medic/CARES/mobarak/PitVQA/QA/3-sentence-simple-act/video_'
    folder_tail = '/*.txt'

    val_dataset = PitVQASentence(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # load vector-molora weights
    mora_base_rank = 8
    mora_coeff = [36, 36, 32, 32, 28, 28, 24, 24, 20, 20, 16, 16]
    lora_rank = [18, 18, 16, 16, 14, 14, 12, 12, 10, 10, 8, 8]
    lora_alpha = [18, 18, 16, 16, 14, 14, 12, 12, 10, 10, 8, 8]
    dropout = 0.1

    model = PitVQAPlus(mora_base_rank=mora_base_rank, mora_rank_coefficients=mora_coeff,
                      lora_rank=lora_rank, lora_alpha=lora_alpha, dropout=dropout)
    checkpoint = torch.load(vec_weight_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    st = time.time()
    references, hypotheses, samplewise_unc_all = validate(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)
    en = time.time()
    print('total time:', en-st)

    # compute BLEU, ROUGE and METEOR scores
    samplewise_rougeL = get_nlp_metrics(references, hypotheses)
    print(len(samplewise_unc_all), len(samplewise_rougeL))

    # ensure numpy arrays for easier manipulation
    uncertainty_array = np.array(samplewise_unc_all)
    rougeL_array = np.array(samplewise_rougeL)
    # get sorted indices based on uncertainty in descending order (highest uncertainty first)
    sorted_indices = np.argsort(-uncertainty_array)
    # sort RougeL scores based on sorted uncertainty indices
    sorted_rougeL = rougeL_array[sorted_indices]
    # define removal percentages
    removal_percentages = [0, 20, 40, 60, 80]

    print("Vector-MoLoRA - Average RougeL after removing samples with highest uncertainty:")
    avg_rougeL_by_refer = []
    for percent in removal_percentages:
        num_to_remove = int(len(sorted_rougeL) * (percent / 100))
        remaining_rougeL = sorted_rougeL[num_to_remove:]  # remove top N% uncertain samples
        avg_rougeL = np.mean(remaining_rougeL) if len(remaining_rougeL) > 0 else 0
        print(f"After removing top {percent}%: {avg_rougeL:.4f}")
        avg_rougeL_by_refer.append(avg_rougeL)

    # plot
    refer = np.array(['0', '0.2', '0.4', '0.6', '0.8'])
    avg_rougeL_by_refer = np.array(avg_rougeL_by_refer)
    
    ax = plt.gca()
    line1 = ax.plot(refer, avg_rougeL_by_refer.transpose(), marker='o', markersize=4, label='Vector-MoLoRA (Ours)')
    ax.legend()
    ax.set_ylabel('Performance (Rouge-L)')
    ax.set_xlabel("Refer to Clinicians\n(Exclusion Ratio of Highly Uncertain Samples)")
    ax.set_title('Risk Coverage Curve')
    plt.savefig(f"/path/to/file/risk_coverage_curve.pdf", bbox_inches='tight', dpi=1500)
