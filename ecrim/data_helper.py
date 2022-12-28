
import pickle
import os
import re
import logging
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb
from buffer import Buffer
from utils import CAPACITY, BLOCK_SIZE, ForkedPdb
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    T5ForConditionalGeneration,
)
from typing import Dict, List
from a2t.base import Classifier, np_softmax
from collections import defaultdict
from dataclasses import dataclass

class SimpleListDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'rb') as fin:
                logging.info('Loading dataset...')
                self.dataset = pickle.load(fin)
        elif isinstance(source, list):
            self.dataset = source
        if not isinstance(self.dataset, list):
            raise ValueError('The source of SimpleListDataset is not a list.')
    def __getitem__(self, index):
        return self.dataset[index] 
    def __len__(self):
        return len(self.dataset)

class BlkPosInterface:
    def __init__(self, dataset):
        assert isinstance(dataset, SimpleListDataset)
        pdb.set_trace()
        self.d = {}
        self.dataset = dataset
        for bufs in dataset:
            for buf in bufs:
                for blk in buf:
                    assert blk.pos not in self.d
                    self.d[blk.pos] = blk
    def set_property(self, pos, key, value=None):
        blk = self.d[pos]
        if value is not None:
            setattr(blk, key, value)
        elif hasattr(blk, key):
            delattr(blk, key)
    def apply_changes_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                tmp = [
                    int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s 
                    for s in line.split()
                ]
                self.set_property(*tmp)
    def apply_changes_from_dir(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes_'):
                self.apply_changes_from_file(filename)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def collect_estimations_from_dir(self, tmp_dir):
        ret = []
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('estimations_'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        self.d[pos].estimation = estimation
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def build_random_buffer(self, num_samples):
        ForkedPdb().set_trace()
        n0, n1 = [int(s) for s in num_samples.split(',')][:2]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        logging.info('building buffers for introspection...')
        for qbuf, dbuf in tqdm(self.dataset):
            # 1. continous 
            lb = max_blk_num - len(qbuf)
            st = random.randint(0, max(0, len(dbuf) - lb * n0))
            for i in range(n0):
                buf = Buffer()
                buf.blocks = qbuf.blocks + dbuf.blocks[st + i * lb:st + (i+1) * lb]
                ret.append(buf) 
            # 2. pos + neg
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True) 
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf))) 
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf))) 
                buf = Buffer()
                buf.blocks = qbuf.blocks + selected_pblks + selected_nblks
                ret.append(buf.sort_())
        return SimpleListDataset(ret)

    def build_promising_buffer(self, num_samples):
        n2, n3 = [int(x) for x in num_samples.split(',')][2:]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        logging.info('building buffers for reasoning...')
        for qbuf, dbuf in tqdm(self.dataset):
            #1. retrieve top n2*(max-len(pos)) estimations into buf 2. cut
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            if len(pbuf) >= max_blk_num - len(qbuf):
                pbuf = pbuf.random_sample(max_blk_num - len(qbuf) - 1) 
            lb = max_blk_num - len(qbuf) - len(pbuf)
            estimations = torch.tensor([blk.estimation for blk in nbuf], dtype=torch.long)
            keeped_indices = estimations.argsort(descending=True)[:n2 * lb]
            selected_nblks = [blk for i, blk in enumerate(nbuf) if i in keeped_indices]
            while 0 < len(selected_nblks) < n2 * lb:
                selected_nblks = selected_nblks * (n2 * lb // len(selected_nblks) + 1)
            for i in range(n2):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + selected_nblks[i * lb: (i+1) * lb]
                ret.append(buf.sort_())
            for i in range(n3):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + random.sample(nbuf.blocks, min(len(nbuf), lb))
                ret.append(buf.sort_())
        return SimpleListDataset(ret)

def find_lastest_checkpoint(checkpoints_dir, epoch=False):
    lastest = (-1, '')
    if os.path.exists(checkpoints_dir):
        for shortname in os.listdir(checkpoints_dir):
            m = re.match(r'_ckpt_epoch_(\d+).+', shortname)
            if m is not None and int(m.group(1)) > lastest[0]:
                lastest = (int(m.group(1)), shortname)
    return os.path.join(checkpoints_dir, lastest[-1]) if not epoch else lastest[0]


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


class _NLIRelationClassifier(Classifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda=True,
        half=False,
        verbose=True,
        negative_threshold=0.95,
        negative_idx=0,
        max_activations=np.inf,
        valid_conditions=None,
        **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )
        # self.ent_pos = entailment_position
        # self.cont_pos = -1 if self.ent_pos == 0 else 0
        self.negative_threshold = negative_threshold
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.n_rel = len(labels)
        # for label in labels:
        #     assert '{subj}' in label and '{obj}' in label

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=False,
    ):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            sentences = [
                f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}."
                for label_template in self.labels
            ]
            batch.extend(sentences)

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch, multiclass=multiclass)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch, multiclass=multiclass)
            outputs.append(output)

        outputs = np.vstack(outputs)

        return outputs

    def _apply_negative_threshold(self, probs):
        activations = (probs >= self.negative_threshold).sum(-1).astype(np.int)
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00
        return probs

    def _apply_valid_conditions(self, probs, features: List[REInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
            axis=0,
        )
        probs = probs * mask_matrix

        return probs

    def predict(
        self,
        contexts: List[str],
        batch_size: int = 1,
        return_labels: bool = True,
        return_confidences: bool = False,
        topk: int = 1,
    ):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk]
        if return_labels:
            topics = self.idx2label(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        template_mapping: Dict[str, str],
        pretrained_model: str = "roberta-large-mnli",
        valid_conditions: Dict[str, list] = None,
        *args,
        **kwargs,
    ):

        self.template_mapping_reverse = defaultdict(list)
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)
        self.new_topics = list(self.template_mapping_reverse.keys())

        self.target_labels = labels
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in template_mapping.items():
            self.mapping[key].extend([self.new_labels2id[v] for v in value])

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            valid_conditions=None,
            **kwargs,
        )

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.target_labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack(
            [
                np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True)
                if label in self.mapping
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )
        outputs = np_softmax(outputs) if not multiclass else outputs

        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        return outputs