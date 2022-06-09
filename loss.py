import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict

import copy
import logging

import numpy as np
import scipy.stats as stats

from sentence_transformers import models
from modules import Transformer, SentenceTransformer

def compute_entropy(probs):
    eps = torch.finfo(probs.dtype).eps
    ps_clamped = probs.clamp(min=eps, max=1 - eps)
    logits = torch.log(ps_clamped)
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * probs
    return -p_log_p.sum(-1)

class NTXentLossOriginal(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLossOriginal, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        self.batch_size = zis.size(0)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self._get_correlated_mask().type(torch.bool)].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / self.batch_size

class NTXentLossOpt1(NTXentLossOriginal):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLossOpt1, self).__init__(device, batch_size, temperature, use_cosine_similarity)

    def forward(self, cls, pooled):
        self.batch_size = cls.size(0)
        representations = torch.cat([cls, pooled], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        pos = torch.diag(similarity_matrix, self.batch_size)
        positives = pos.view(self.batch_size, 1)

        negatives = similarity_matrix[self._get_correlated_mask().type(torch.bool)].view(2 * self.batch_size, -1)[:self.batch_size]

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / self.batch_size


class NTXentLossOpt2(NTXentLossOriginal):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLossOpt2, self).__init__(device, batch_size, temperature, use_cosine_similarity)

    def forward(self, cls, pooled):
        self.batch_size = cls.size(0)
        representations = torch.cat([cls, pooled], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        pos = torch.diag(similarity_matrix, self.batch_size)
        positives = pos.view(self.batch_size, 1)

        negatives = similarity_matrix[self._get_correlated_mask().type(torch.bool)].view(2 * self.batch_size, -1)[:self.batch_size, self.batch_size:]

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / self.batch_size


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def euclidean(self, x, y):
        return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1).sqrt()

    def forward(self, cls, cont):
        """
        :param cls: (batch_size, hidden_size)
        :param cont: (batch_size, num_layers, hidden_size)
        :return:
        """
        batch_size = cls.size(0)
        num_layers = cont.size(1)

        positives, negatives = [], []
        for i in range(num_layers):
            # (batch_size, hidden_size) X (batch_size, hidden_size) -> (batch_size, batch_size)
            similarity_matrix = self.similarity_function(cls, cont[:, i])
            # add (batch_size, 1)
            positives.append(torch.diag(similarity_matrix))
            # (batch_size, batch_size - 1)
            neg_idx = (1 - torch.eye(batch_size)).bool()
            negatives.append(similarity_matrix[neg_idx].view(batch_size, -1))
        # (batch_size * num_layers, 1)
        positives = torch.cat(positives).view(-1, 1)

        # add other cls embeddings to negative samples
        # similarity_matrix = self.similarity_function(cls, cls)
        # (batch_size, batch_size - 1)
        # cls_negatives = similarity_matrix[(1 - torch.eye(batch_size)).bool()].view(batch_size, -1)
        # (batch_size * num_layers, batch_size - 1)
        # cls_negatives = torch.cat([cls_negatives] * num_layers, dim=0)

        # (batch_size, (batch_size - 1) * (num_layers (+ 1)))
        negatives = torch.cat(negatives, dim=1)
        # (batch_size * num_layers, (batch_size - 1) * (num_layers (+ 1)))
        negatives = torch.cat([negatives] * num_layers, dim=0)

        # (batch_size * num_layers, 1 + (batch_size - 1) * (num_layers (+ 1)))
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(batch_size * num_layers).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (batch_size * num_layers)


class Loss(nn.Module):
    def __init__(self, model, args):
        super(Loss, self).__init__()
        self.args = args
        config = model._first_module().auto_model.config
        self.config = config
        self.vocab_size = config.vocab_size

        if self.args.lmax == -1:
            self.args.lmax = config.num_hidden_layers + 1

        # class: SentenceTransformer
        self.model = model

        self.original = copy.deepcopy(model)
        self.original[0].eval()
        self.original_params = dict(self.original[0].named_parameters())
        for n, p in self.original_params.items():
            p.requires_grad = False

        if args.freeze:
            for n, p in self.model._first_module().auto_model.embeddings.named_parameters():
                p.requires_grad = False

        ph_hidden_size = 4096
        starting_hidden_size = config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(starting_hidden_size, ph_hidden_size),
            nn.GELU(),
            nn.Linear(ph_hidden_size, ph_hidden_size),
            nn.GELU())

        self.projection_head[0].weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.projection_head[0].bias.data.zero_()
        self.projection_head[2].weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.projection_head[2].bias.data.zero_()

        if self.args.obj == 'SG-OPT':
            self.loss = NTXentLoss
        elif self.args.obj == 'OPT1':
            self.loss = NTXentLossOpt1
        elif self.args.obj == 'OPT2':
            self.loss = NTXentLossOpt2
        else:
            self.loss = NTXentLossOriginal

        self.loss = self.loss(
            device=torch.device(self.args.device),
            batch_size=args.batch_size,
            temperature=args.T,
            use_cosine_similarity=True)

        self.sample_cnt = torch.zeros(config.num_hidden_layers + 1, dtype=torch.int)

    def compute_diff(self):
        diff = 0.0
        for n,p in self.model[0].named_parameters():
            diff += torch.norm(self.original_params[n] - p, p=2) ** 2
        return diff

    def mean_pooling(self, t, mask):
        return self.sum_pooling(t, mask) / mask.sum(2)

    def sum_pooling(self, t, mask):
        t = t * mask
        return t.sum(2)

    def max_pooling(self, t, mask):
        t[mask == 0] = -1e9
        return t.max(dim=2)[0]

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels):
        reps = []
        for i, sf in enumerate(sentence_features):
            if self.args.clone:
                ori = self.original(copy.deepcopy(sentence_features[i]))
            else:
                ori = self.model(copy.deepcopy(sentence_features[i]))
            target = self.model(copy.deepcopy(sentence_features[i]))

            sent_emb = target['sentence_embedding']
            batch_size = sent_emb.size(0)
            # (batch, n_layers, seq_len, hidden_size)
            intermediate = torch.stack([l for l in ori['all_layer_embeddings'][self.args.lmin:self.args.lmax]], dim=1)
            mask = ori['attention_mask'].unsqueeze(1).unsqueeze(-1).expand(intermediate.size()).float()
            # (batch, n_layers, hidden_size)
            pooled = getattr(self, f'{self.args.pooling2}_pooling')(intermediate, mask)
            reps.append({'sent_emb': sent_emb, 'pooled': pooled})

        sent_emb = reps[0]['sent_emb']
        if len(sentence_features) > 1 and self.args.obj == 'BT':
            pooled = reps[1]['sent_emb']
        elif self.args.obj in ['SG', 'OPT1', 'OPT2']:
            idx = torch.randint(self.args.lmin, self.args.lmax, (batch_size,))
            pooled = pooled[torch.arange(batch_size), idx]
        else:
            # pooled = torch.cat([reps[0]['pooled'], reps[1]['pooled']], dim=1)
            pooled = reps[0]['pooled']

        sent_emb = self.projection_head(sent_emb)
        if self.args.pooling == 'test':
            pooled = self.pre_projection_head(pooled)
        pooled = self.projection_head(pooled)
        loss1 = self.loss(sent_emb, pooled)
        if self.args.lamb > 0 :
            loss2 = self.compute_diff()
            return loss1 + self.args.lamb * loss2
        else:
            return loss1


