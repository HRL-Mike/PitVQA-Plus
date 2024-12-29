import os
import torch
import argparse
import torch.utils.data
import numpy as np
import random

from torch import nn
from utils import save_clf_checkpoint, adjust_learning_rate, AverageMeter
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from dataloader import PitVQASentence, EndoVis18VQAGen
from model import PitVQAGen

import evaluate
from nltk.translate.bleu_score import corpus_bleu

import warnings
warnings.filterwarnings('ignore')


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = AverageMeter()

    for i, (images, questions, answers) in enumerate(train_dataloader, 0):
        question_inputs = tokenizer(questions, padding="max_length", max_length=int(args.seq_length),
                                    return_tensors="pt", truncation=True)
        answer_inputs = tokenizer(answers, padding="max_length", max_length=int(args.seq_length),
                                  return_tensors="pt", truncation=True)

        # get logits and labels
        logits = model(image=images.to(device), question_inputs=question_inputs.to(device))
        labels = answer_inputs['input_ids'].to(device)

        # get shifted logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # compute loss
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
    print("Epoch: {}/{} Loss: {:.6f} AVG_Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    references = []
    hypotheses = []
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    model.eval()
    total_loss = AverageMeter()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            question_inputs = tokenizer(questions, padding="max_length", max_length=int(args.seq_length),
                                        return_tensors="pt", truncation=True)
            answer_inputs = tokenizer(answers, padding="max_length", max_length=int(args.seq_length),
                                      return_tensors="pt", truncation=True)

            # get logits and labels
            logits = model(image=images.to(device), question_inputs=question_inputs.to(device))
            labels = answer_inputs['input_ids'].to(device)

            # get shifted logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # compute loss
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss.update(loss.item())

            # generate predicted answer
            _, predicted = torch.max(logits, dim=-1)

            # decode references and predictions
            reference_answers = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predicted_answers = tokenizer.batch_decode(predicted, skip_special_tokens=True)

            # add references and hypotheses to lists
            for ref, hyp in zip(reference_answers, predicted_answers):
                references.append([ref.split()])
                hypotheses.append(hyp.split())

        ref_sentence = [' '.join(ref[0]) for ref in references] 
        hyp_sentence = [' '.join(hyp) for hyp in hypotheses]

        print(f"Epoch: {epoch}/{args.epochs} EVA LOSS: {total_loss.avg:.6f}")
        # compute 
        results_bleu = bleu.compute(predictions=hyp_sentence, references=ref_sentence)
        results_rouge = rouge.compute(predictions=hyp_sentence, references=ref_sentence)
        results_meteor = meteor.compute(predictions=hyp_sentence, references=ref_sentence)
        print("HF results: ")
        print(f"BLEU-4: {results_bleu['bleu']:.6f}")
        print(f"Rouge1: {results_rouge['rouge1']:.6f}, RougeL: {results_rouge['rougeL']:.6f}, "
              f"Meteor: {results_meteor['meteor']:.6f}")

        # Calculate BLEU_1~4
        metrics = {}
        metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
        metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
        metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
        metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        print("NLTK results: ")
        print(f"BLEU-1: {metrics['Bleu_1']:.6f} BLEU-2: {metrics['Bleu_2']:.6f} "
              f"BLEU-3: {metrics['Bleu_3']:.6f} BLEU-4: {metrics['Bleu_4']:.6f}")

    return metrics, results_bleu, results_rouge, results_meteor


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerGeneration')
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=60,   help='number of epochs to train for')
    parser.add_argument('--batch_size',     type=int,   default=32,   help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=32,   help='sequence length for question and answer')

    parser.add_argument('--mora_base_rank', type=int, default=8, help='MoRA base rank')
    parser.add_argument('--mora_coeff', type=int, nargs='+', default=[32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22], help='Vector-MoRA coefficient')

    parser.add_argument('--lora_rank', type=int, nargs='+', default=[16, 16, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6], help='Vector-LoRA rank')
    parser.add_argument('--lora_alpha', type=int, nargs='+', default=[64, 64, 56, 56, 48, 48, 40, 40, 32, 32, 24, 24], help='Vector-LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset', default='endo18', help='endo18 or open-pit')
    parser.add_argument('--lr',             type=float, default=0.00002,  help='0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/saved_weights_',  help='path to checkpoint')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arg()
    os.makedirs('./checkpoints/', exist_ok=True)

    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Random seed: {args.random_seed}')
    print(f'MoRA base rank: {args.mora_base_rank}')
    print(f'Sequence length: {args.seq_length}')
    print(f'Dropout: {args.dropout}')

    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0
    best_hf_bleu = 0.0
    best_rouge = 0.0
    best_meteor = 0.0

    # dataset preparation
    train_dataloader = None
    val_dataloader = None
    if args.dataset == 'endo18':
        # data split
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        # path to dataset folder
        folder_head = '/SAN/medic/CARES/mobarak/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Sentence/*.txt'
        # dataloader
        train_dataset = EndoVis18VQAGen(train_seq, folder_head, folder_tail)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)
        val_dataset = EndoVis18VQAGen(val_seq, folder_head, folder_tail)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

    if args.dataset == 'open-pit':
        # data split
        train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
        val_seq = ['02', '06', '12', '13', '24']
        # path to dataset folder
        folder_head = r'/SAN/medic/CARES/mobarak/PitVQA/QA/0-sentence-simple/video_'
        folder_tail = '/*.txt'
        # dataloader
        train_dataset = PitVQASentence(train_seq, folder_head, folder_tail)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)
        val_dataset = PitVQASentence(val_seq, folder_head, folder_tail)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

    # init tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f'MoRA coeff: {args.mora_coeff}')
    print(f'LoRA ranks: {args.lora_rank}')
    print(f'LoRA alpha: {args.lora_alpha}')
    model = PitVQAGen(mora_base_rank=args.mora_base_rank, mora_rank_coefficients=args.mora_coeff,
                      lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, dropout=args.dropout)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # train and validation
    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train
        train(args, train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer,
              epoch=epoch, tokenizer=tokenizer, device=device)
        # validation
        metrics, results_bleu, results_rouge, results_meteor = validate(args, val_loader=val_dataloader, model=model, criterion=criterion, 
                                                                        epoch=epoch, tokenizer=tokenizer, device=device)

        if results_bleu['precisions'][3] >= best_hf_bleu:
            print(f"Best HF BLEU-4 score, {results_bleu['precisions'][3]:.6f}")
            best_hf_bleu = results_bleu['precisions'][3]
        
        if results_rouge['rougeL'] >= best_rouge:
            print(f"Best RougeL score, {results_rouge['rougeL']:.6f}")
            best_rouge = results_rouge['rougeL']
        
        if results_meteor['meteor'] >= best_meteor:
            print(f"Best Meteor score, {results_meteor['meteor']:.6f}")
            best_meteor = results_meteor['meteor']

        if metrics["Bleu_4"] >= best_results[0]:
            epochs_since_improvement = 0
            best_results[0] = metrics["Bleu_4"]
            best_epoch[0] = epoch
            print(f'Best epoch: {epoch}, Best Bleu_4: {metrics["Bleu_4"]}')

            save_dir = args.checkpoint_dir + 'epoch_' + str(epoch)
            save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement,
                                model, optimizer, best_results[0], final_args=None)
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    print('End training.')
