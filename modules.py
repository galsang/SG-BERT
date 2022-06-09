import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Optional
from zipfile import ZipFile

import numpy as np
import requests
import sentence_transformers
import sentence_transformers.models as models
import torch
import torch.nn as nn
import transformers
from sentence_transformers import __DOWNLOAD_SERVER__
from sentence_transformers import __version__
from sentence_transformers.datasets.EncodeDataset import EncodeDataset
from sentence_transformers.models import Pooling
from sentence_transformers.readers import InputExample
from sentence_transformers.util import import_from_string, batch_to_device, http_get
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.autonotebook import trange

class Transformer(models.Transformer):

    __module__ = 'sbert_modules.Transformer'

    def __init__(self, model_name_or_path: str, max_seq_length: int = 128,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None):
        super(Transformer, self).__init__(model_name_or_path, max_seq_length, model_args, cache_dir, tokenizer_args, do_lower_case)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.auto_model(**features)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens,
                         'cls_token_embeddings': cls_tokens,
                         'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})
            if self.auto_model.config.output_attentions:
                attentions = output_states[all_layer_idx+1]
                features.update({'attentions': attentions})

        return features

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json',
                            'sentence_roberta_config.json',
                            'sentence_distilbert_config.json',
                            'sentence_camembert_config.json',
                            'sentence_albert_config.json',
                            'sentence_xlm-roberta_config.json',
                            'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        config['model_args'] = {'output_hidden_states': True, 'output_attentions': True}

        return Transformer(model_name_or_path=input_path, **config)


class SentenceTransformer(sentence_transformers.SentenceTransformer):

    __module__ = 'sbert_modules.SentenceTransformer'

    def __init__(self, model_name_or_path=None, modules=None, device=None, name=None):
        self.encoder_name = name

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))
            model_path = model_name_or_path

            if not os.path.isdir(model_path) and not model_path.startswith('http://') and not model_path.startswith('https://'):
                logging.info("Did not find folder {}".format(model_path))

                if '\\' in model_path or model_path.count('/') > 1:
                    raise AttributeError("Path {} not found".format(model_path))

                model_path = __DOWNLOAD_SERVER__ + model_path + '.zip'
                logging.info("Try to download model from server: {}".format(model_path))

            if model_path.startswith('http://') or model_path.startswith('https://'):
                model_url = model_path
                folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250].rstrip('.zip')

                try:
                    from torch.hub import _get_torch_home
                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv('TORCH_HOME', os.path.join(
                            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
                default_cache_path = os.path.join(torch_cache_home, 'sentence_transformers')
                model_path = os.path.join(default_cache_path, folder_name)

                if not os.path.exists(model_path) or not os.listdir(model_path):
                    if model_url[-1] == "/":
                        model_url = model_url[:-1]
                    logging.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))

                    model_path_tmp = model_path.rstrip("/").rstrip("\\")+"_part"
                    try:
                        zip_save_path = os.path.join(model_path_tmp, 'model.zip')
                        http_get(model_url, zip_save_path)
                        with ZipFile(zip_save_path, 'r') as zip:
                            zip.extractall(model_path_tmp)
                        os.remove(zip_save_path)
                        os.rename(model_path_tmp, model_path)
                    except requests.exceptions.HTTPError as e:
                        shutil.rmtree(model_path_tmp)
                        if e.response.status_code == 404:
                            logging.warning('SentenceTransformer-Model {} not found. Try to create it from scratch'.format(model_url))
                            logging.warning('Try to create Transformer Model {} with mean pooling'.format(model_name_or_path))

                            model_path = None
                            transformer_model = Transformer(model_name_or_path)
                            pooling_model = Pooling(transformer_model.get_word_embedding_dimension())
                            modules = [transformer_model, pooling_model]

                        else:
                            raise e
                    except Exception as e:
                        shutil.rmtree(model_path)
                        raise e

            #### Load from disk
            if model_path is not None:
                logging.info("Load SentenceTransformer from folder: {}".format(model_path))

                if os.path.exists(os.path.join(model_path, 'config.json')):
                    with open(os.path.join(model_path, 'config.json')) as fIn:
                        config = json.load(fIn)
                        if config['__version__'] > __version__:
                            logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

                with open(os.path.join(model_path, 'modules.json')) as fIn:
                    contained_modules = json.load(fIn)

                modules = OrderedDict()
                for module_config in contained_modules:
                    if module_config['type'] in ['sentence_transformers.models.Transformer', 'sentence_transformers.models.BERT']:
                        module_config['type'] = 'sbert_modules.Transformer'
                    module_class = import_from_string(module_config['type'])
                    module = module_class.load(os.path.join(model_path, module_config['path']))
                    modules[module_config['name']] = module

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        nn.Sequential.__init__(self, modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

    def forward(self, input):
        for i, module in enumerate(self):
            if i > 0 and isinstance(module, models.Transformer):
                pass
            else:
                input = module(input)
        return input

    def fit(self, train_objectives, dev_evaluator, test_evaluator, epochs=1,
            steps_per_epoch=None, scheduler='WarmupLinear', warmup_steps=10000,
            optimizer_class=transformers.AdamW, optimizer_params={},
            weight_decay=0.01, evaluation_steps=0, output_path=None,
            save_best_model=True, max_grad_norm=1, use_amp=False, callback=None,
            output_path_ignore_not_empty=False, early_stopping_limit=5, disable_tqdm=False):
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = [(n,p) for n, p in list(loss_model.named_parameters()) if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        dev_score = self._eval_during_training(dev_evaluator, output_path, False, 0, 0, callback)

        skip_scheduler = False
        early_stopping_cnt = 0
        last_score = 0

        range_epoch = range(epochs) if disable_tqdm else trange(epochs, desc='Epoch')
        range_iter = range(steps_per_epoch) if disable_tqdm else trange(steps_per_epoch, desc="Iteration", smoothing=0.05)

        for epoch in range_epoch:
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in range_iter:
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        #logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self._target_device)

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    dev_score = self._eval_during_training(dev_evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                    if dev_score < last_score:
                        early_stopping_cnt += 1
                    if early_stopping_cnt >= early_stopping_limit:
                        logging.info('Early stopping!')
                        return
                    last_score = dev_score

            self._eval_during_training(dev_evaluator, output_path, save_best_model, epoch, training_steps, callback)
            if test_evaluator is not None:
                self._eval_during_training(test_evaluator, output_path, save_best_model, epoch, training_steps, callback)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

        return score


class SentencesDataset(Dataset):
    def __init__(self, examples: List[InputExample], model):
        self.model = model
        self.examples = examples
        self.n = 0
        for m in model:
            if isinstance(m, models.Transformer):
                self.n += 1
        self.label_type = torch.long if isinstance(self.examples[0].label, int) else torch.float

    def __getitem__(self, item):
        label = torch.tensor(self.examples[item].label, dtype=self.label_type)
        if self.examples[item].texts_tokenized is None:
            if self.n > 1:
                text = self.examples[item].texts[0]
                self.examples[item].texts_tokenized = [self.model[i].tokenize(text) for i in range(self.n)]
            else:
                self.examples[item].texts_tokenized = [self.model.tokenize(text) for text in self.examples[item].texts]
        return self.examples[item].texts_tokenized, label

    def __len__(self):
        return len(self.examples)
