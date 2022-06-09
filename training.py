import argparse
import csv
import gzip
import logging
import math
import random
import time

import numpy as np
import torch
from sentence_transformers import LoggingHandler, InputExample
from sentence_transformers import models
from sentence_transformers.models import Transformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

from loss import Loss
from modules import SentencesDataset, SentenceTransformer

start_time = time.time()

PRETRAINED_MODELS = ['bert-base-nli-cls-token',
                     'bert-base-nli-mean-tokens',
                     'bert-large-nli-cls-token',
                     'bert-large-nli-mean-tokens',
                     'roberta-base-nli-cls-token',
                     'roberta-base-nli-mean-tokens',
                     'roberta-large-nli-cls-token',
                     'roberta-large-nli-mean-tokens']

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--model_name', default='bert-base-uncased', type=str)
parser.add_argument('--pooling', default='cls', type=str)
parser.add_argument('--pooling2', default='mean', type=str)
parser.add_argument('--eval_step', default=50, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--T', default=1e-2, type=float)
parser.add_argument('--eps', default=0.1, type=float)
parser.add_argument('--lmin', default=0, type=int)
parser.add_argument('--lmax', default=-1, type=int)
parser.add_argument('--lamb', default=0.1, type=float)
parser.add_argument('--es', default=10, type=int)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--training', default=True, action='store_true')
parser.add_argument('--freeze', default=True, action='store_true')
parser.add_argument('--clone', default=True, action='store_true')
parser.add_argument('--disable_tqdm', default=True, action='store_true')
parser.add_argument('--obj', default='SG-OPT', type=str)
parser.add_argument('--device', default='cuda:0', type=str)

args = parser.parse_args()
for a in args.__dict__:
    print(f'{a}: {args.__dict__[a]}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.random.manual_seed(args.seed)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

sts_dataset_path = 'stsbenchmark.tsv.gz'

args_string = args.model_name + '-' + str(args.seed) + '-' + str(args.eps) + '-' + args.pooling + '-' +  str(args.lmin) + '-' + str(args.lmax)
logging.info(f'args_string: {args_string}')
model_save_path = f'output/{args_string}'

if args.model_name in PRETRAINED_MODELS:
    logging.info('Loading from SBERT')
    pretrained = SentenceTransformer(args.model_name)
    word_embedding_model = pretrained._first_module()
else:
    model_args = {'output_hidden_states': True, 'output_attentions': True}
    word_embedding_model = Transformer(args.model_name, model_args=model_args)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=args.pooling == 'mean' or args.pooling not in ['cls', 'max'],
    pooling_mode_cls_token=args.pooling == 'cls',
    pooling_mode_max_tokens=args.pooling == 'max')

modules = [word_embedding_model, pooling_model]
model = SentenceTransformer(modules=modules, name=args.model_name, device=args.device)

train_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf-8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples.append(InputExample(texts=[row['sentence1']]))
        train_samples.append(InputExample(texts=[row['sentence2']]))

train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
train_loss = Loss(model, args)

logging.info(f"Read eval dataset")
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.batch_size, name=f'stsb-dev', main_similarity=SimilarityFunction.COSINE)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=args.batch_size, name=f'stsb-test', main_similarity=SimilarityFunction.COSINE)

warmup_steps = math.ceil(len(train_dataset) * args.num_epochs / args.batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(train_objectives=[(train_dataloader, train_loss)],
          dev_evaluator=dev_evaluator,
          test_evaluator= None,
          epochs=args.num_epochs,
          optimizer_params={'lr': args.lr, 'correct_bias': True, 'weight_decay': args.weight_decay, 'betas': (0.9, 0.9)},
          evaluation_steps=args.eval_step,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          early_stopping_limit=args.es,
          disable_tqdm=args.disable_tqdm)

logging.info('Training finished.')

dev_score = dev_evaluator(model, output_path=model_save_path)
test_score = test_evaluator(model, output_path=model_save_path)
print(dev_score)
print(test_score)