from concurrent.futures.thread import _threads_queues
import json
import random
from functools import partial
import pdb
from turtle import pd
import numpy as np
import redis
import sklearn
import torch
from eveliver import (Logger, load_model, tensor_to_obj)
from trainer import Trainer, TrainerCallback
from transformers import AutoTokenizer, BertModel
from matrix_transformer import Encoder as MatTransformer
from graph_encoder import Encoder as GraphEncoder
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from buffer import Buffer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME, contrastive_pair, check_htb_debug, complete_h_t_debug
from utils import complete_h_t, check_htb, check_htb_debug
from utils import CLS_TOKEN_ID, SEP_TOKEN_ID, H_START_MARKER_ID, H_END_MARKER_ID, T_END_MARKER_ID, T_START_MARKER_ID
import math
from torch.nn import CrossEntropyLoss
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from itertools import groupby
from pyg_graph import create_edges, create_graph, GCN, Attention, create_graph_single
from utils import DotProductSimilarity
from sentence_reordering import SentReOrdering
from sbert_wk import sbert
from itertools import product, combinations
def eval_performance(facts, pred_result):
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    rec = []
    correct = 0
    total = len(facts)
    #pdb.set_trace()
    for i, item in enumerate(sorted_pred_result):
        if (item['entpair'][0], item['entpair'][1], item['relation']) in facts:
            correct += 1
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))
    auc = sklearn.metrics.auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()
    return {'prec': np_prec.tolist(), 'rec': np_rec.tolist(), 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}

def expand(start, end, total_len, max_size):
    e_size = max_size - (end - start)
    _1 = start - (e_size // 2)
    _2 = end + (e_size - e_size // 2)
    if _2 - _1 <= total_len:
        if _1 < 0:
            _2 -= -1
            _1 = 0
        elif _2 > total_len:
            _1 -= (_2 - total_len)
            _2 = total_len
    else:
        _1 = 0
        _2 = total_len
    return _1, _2


def place_train_data(dataset):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, 'n/a', l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            for label in labels:
                if label != 'n/a':
                    ds = l2docs[label]
                    if 'n/a' in l2docs:
                        ds.extend(l2docs['n/a'])
                    bags.append([key, label, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + x[1])
    return bags


def place_dev_data(dataset, single_path):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags

def place_test_data(dataset, single_path):
    ep2d = dict()
    for data in dataset:
        key = data['h_id'] + '#' + data['t_id']
        doc1 = data['doc'][0]
        doc2 = data['doc'][1]
        label = 'n/a'
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags


def gen_c(tokenizer, passage, span, max_len, bound_tokens, d_start, d_end, no_additional_marker, mask_entity):
    ret = list() 
    ret.append(bound_tokens[0])
    for i in range(span[0], span[1]):
        if mask_entity:
            ret.append('[MASK]')
        else:
            ret.append(passage[i])
    ret.append(bound_tokens[1])
    prev = list()
    prev_ptr = span[0] - 1 
    while len(prev) < max_len:
        if prev_ptr < 0:
            break
        if not no_additional_marker and prev_ptr in d_end:
            prev.append(f'[unused{(d_end[prev_ptr] + 2) * 2 + 2}]')
        prev.append(passage[prev_ptr])
        if not no_additional_marker and prev_ptr in d_start:
            prev.append(f'[unused{(d_start[prev_ptr] + 2) * 2 + 1}]')
        prev_ptr -= 1
    nex = list()
    nex_ptr = span[1]
    while len(nex) < max_len:
        if nex_ptr >= len(passage):
            break
        if not no_additional_marker and nex_ptr in d_start:
            nex.append(f'[unused{(d_start[nex_ptr] + 2) * 2 + 1}]')
        nex.append(passage[nex_ptr])
        if not no_additional_marker and nex_ptr in d_end:
            nex.append(f'[unused{(d_end[nex_ptr] + 2) * 2 + 2}]')
        nex_ptr += 1
    prev.reverse()
    ret = prev + ret + nex
    return ret

def process(tokenizer, h, t, doc0, doc1):                    

    ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
    b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
    max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
    cnt, batches = 0, []
    d = []

    def fix_entity(doc, ht_markers, b_markers):
        markers = ht_markers + b_markers
        markers_pos = []
        if list(set(doc).intersection(set(markers))):
            for marker in markers:
                try:
                    pos = doc.index(marker)
                    markers_pos.append((pos, marker))
                except ValueError as e:
                    continue
        
        idx = 0
        while idx <= len(markers_pos)-1:
            try:
                assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) - int(markers_pos[idx+1][1].replace("[unused", "").replace("]", "")) == -1)
                entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                while "." in entity_name:
                    assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                idx += 2
            except:
                #pdb.set_trace()
                idx += 1
                continue
        return doc

    d0 = fix_entity(doc0, ht_markers, b_markers)
    d1 = fix_entity(doc1, ht_markers, b_markers)

    for di in [d0, d1]:
        d.extend(di)
    d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
    d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
    dbuf = Buffer()
    dbuf.blocks = d0_buf.blocks + d1_buf.blocks
    for blk in dbuf:
        if list(set(tokenizer.convert_tokens_to_ids(ht_markers)).intersection(set(blk.ids))):
            blk.relevance = 2
        elif list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
            blk.relevance = 1
        else:
            continue
    ret = []

    n0 = 1
    pbuf_ht, nbuf_ht = dbuf.filtered(lambda blk, idx: blk.relevance >= 2, need_residue=True) 
    pbuf_b, nbuf_b = nbuf_ht.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True) 

    for i in range(n0):
        _selected_htblks = random.sample(pbuf_ht.blocks, min(max_blk_num, len(pbuf_ht)))
        _selected_pblks = random.sample(pbuf_b.blocks, min(max_blk_num - len(_selected_htblks), len(pbuf_b)))
        _selected_nblks = random.sample(nbuf_b.blocks, min(max_blk_num - len(_selected_pblks) - len(_selected_htblks), len(nbuf_b)))
        buf = Buffer()
        buf.blocks = _selected_htblks + _selected_pblks + _selected_nblks
        ret.append(buf.sort_())
    ret[0][0].ids.insert(0, tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
    return ret[0]


def process_example_simp(h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity):
    max_len = 99999
    bert_max_len = 512
    doc1 = json.loads(redisd.get('codred-doc-' + doc1))
    doc2 = json.loads(redisd.get('codred-doc-' + doc2))
    v_h = None
    for entity in doc1['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
            v_h = entity
    assert v_h is not None
    v_t = None
    for entity in doc2['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
            v_t = entity
    assert v_t is not None
    d1_v = dict()
    for entity in doc1['entities']:
        if 'Q' in entity:
            d1_v[entity['Q']] = entity
    d2_v = dict()
    for entity in doc2['entities']:
        if 'Q' in entity:
            d2_v[entity['Q']] = entity
    ov = set(d1_v.keys()) & set(d2_v.keys())
    if len(ov) > 40:
        ov = set(random.choices(list(ov), k=40))
    ov = list(ov)
    ma = dict()
    for e in ov:
        ma[e] = len(ma)
    d1_start = dict()
    d1_end = dict()
    for entity in doc1['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d1_start[span[0]] = ma[entity['Q']]
                d1_end[span[1] - 1] = ma[entity['Q']]
    d2_start = dict()
    d2_end = dict()
    for entity in doc2['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d2_start[span[0]] = ma[entity['Q']]
                d2_end[span[1] - 1] = ma[entity['Q']]
    k1 = gen_c(tokenizer, doc1['tokens'], v_h['spans'][0], max_len, ['[unused1]', '[unused2]'], d1_start, d1_end, no_additional_marker, mask_entity)
    k2 = gen_c(tokenizer, doc2['tokens'], v_t['spans'][0], max_len, ['[unused3]', '[unused4]'], d2_start, d2_end, no_additional_marker, mask_entity)
    
    #pdb.set_trace()
    selected_rets = process(tokenizer, v_h['name'], v_t['name'], k1, k2)

    return selected_rets


def collate_fn(batch, args, relation2id, tokenizer, redisd, encoder, sbert_wk):
    #assert len(batch) == 1
    if batch[0][-1] == 'o':
        batch = batch[0]
        h, t = batch[0].split('#')
        r = relation2id[batch[1]]
        dps = batch[2]
        if len(dps) > 8:
            dps = random.choices(dps, k=8)
        dplabel = list()
        selected_rets = list()
        for doc1, doc2, l in dps:

            selected_ret = process_example_simp(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
            
            for s_blk in selected_ret:
                while(tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                    s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(".")
            dplabel.append(relation2id[l])
            selected_rets.append(selected_ret)
        dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
        rs_t = torch.tensor([r], dtype=torch.int64)

        

        selected_inputs = torch.zeros(4, len(dps), CAPACITY, dtype=torch.int64) 
        for dp, buf in enumerate(selected_rets):
            buf.export_01_turn(out=(selected_inputs[0, dp], selected_inputs[1, dp], selected_inputs[2, dp]))

        selected_ids = selected_inputs[0]
        selected_att_mask = selected_inputs[1]
        selected_token_type = selected_inputs[2]
        selected_labels = selected_inputs[3]

    else:
        examples = batch[0]
        h_len = tokenizer.max_len_sentences_pair // 2 - 2
        t_len = tokenizer.max_len_sentences_pair - tokenizer.max_len_sentences_pair // 2 - 2
        _input_ids = list()
        _token_type_ids = list()
        _attention_mask = list()
        _rs = list()
        selected_rets = list()
        for idx, example in enumerate(examples):
            doc = json.loads(redisd.get(f'dsre-doc-{example[0]}'))
            _, h_start, h_end, t_start, t_end, r = example
            if r in relation2id:
                r = relation2id[r]
            else:
                r = 'n/a'
            h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
            t_1, t_2 = expand(t_start, t_end, len(doc), t_len)
            h_tokens = doc[h_1:h_start] + ['[unused1]'] + doc[h_start:h_end] + ['[unused2]'] + doc[h_end:h_2]
            t_tokens = doc[t_1:t_start] + ['[unused3]'] + doc[t_start:t_end] + ['[unused4]'] + doc[t_end:t_2]
            h_name = doc[h_start:h_end]
            t_name = doc[t_start:t_end]
            h_token_ids = tokenizer.convert_tokens_to_ids(h_tokens)
            t_token_ids = tokenizer.convert_tokens_to_ids(t_tokens)
            selected_ret = process(tokenizer, " ".join(doc[h_start:h_end]), " ".join(doc[t_start:t_end]), h_tokens, t_tokens)
            for s_blk in selected_ret:
                while(tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                    s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(".")
            input_ids = tokenizer.build_inputs_with_special_tokens(h_token_ids, t_token_ids)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(h_token_ids, t_token_ids)
            obj = tokenizer._pad({'input_ids': input_ids, 'token_type_ids': token_type_ids}, max_length=args.seq_len, padding_strategy='max_length')
            _input_ids.append(obj['input_ids'])
            _token_type_ids.append(obj['token_type_ids'])
            _attention_mask.append(obj['attention_mask'])
            _rs.append(r)
            selected_rets.append(selected_ret)
        dplabel_t = torch.tensor(_rs, dtype=torch.long)
        rs_t = None
        r = None
        selected_inputs = torch.zeros(4, len(examples), CAPACITY, dtype=torch.int64)
        for ex, buf in enumerate(selected_rets):
            buf.export_01_turn(out=(selected_inputs[0, ex], selected_inputs[1, ex], selected_inputs[2, ex]))
        selected_ids = selected_inputs[0]
        selected_att_mask = selected_inputs[1]
        selected_token_type = selected_inputs[2]
        selected_labels = selected_inputs[3]
    return dplabel_t, rs_t, [r], selected_ids, selected_att_mask, selected_token_type, selected_labels, selected_rets


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd, encoder, sbert_wk):
    #assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]
    selected_rets = list()
    for doc1, doc2, l in dps:
        selected_ret = process_example_simp(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
        for s_blk in selected_ret:
            while(tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(".")
        selected_rets.append(selected_ret)

    selected_inputs = torch.zeros(4, len(dps), CAPACITY, dtype=torch.int64)
    for dp, buf in enumerate(selected_rets):
        buf.export_01_turn(out=(selected_inputs[0, dp], selected_inputs[1, dp], selected_inputs[2, dp]))

    selected_ids = selected_inputs[0]
    selected_att_mask = selected_inputs[1]
    selected_token_type = selected_inputs[2]
    selected_labels = selected_inputs[3]
    
    return h, rs, t, selected_ids, selected_att_mask, selected_token_type, selected_labels, selected_rets




class Codred(torch.nn.Module):
    def __init__(self, args, num_relations):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.predictor = torch.nn.Linear(self.bert.config.hidden_size, num_relations)
        weight = torch.ones(num_relations, dtype=torch.float32)
        weight[0] = 0.1
        self.d_model = 768
        self.reduced_dim = 256
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight, reduction='none')
        self.aggregator = args.aggregator
        self.no_doc_pair_supervision = args.no_doc_pair_supervision
        self.matt = MatTransformer(h = 8 , d_model = self.d_model , hidden_size = 1024 , num_layers = 4 , device = torch.device(0))

        self.graph_enc = GraphEncoder(h = 8 , d_model = self.d_model , hidden_size = 1024 , num_layers = 4)
        self.wu = nn.Linear(self.d_model , self.d_model)
        self.wv = nn.Linear(self.d_model , self.d_model)
        self.wi = nn.Linear(self.d_model , self.d_model)
        self.ln1 = nn.Linear(self.d_model , self.d_model)
        self.ln1_gnn = nn.Linear(2* self.d_model , self.d_model)
        self.dim_reduction = nn.Linear(self.d_model, self.reduced_dim)
        self.reduced_predictor = torch.nn.Linear(self.reduced_dim, num_relations)
        self.gamma = 2
        self.alpha = 0.25
        self.beta = 0.01
        self.d_k = 64
        self.num_relations = num_relations
        self.ent_emb = nn.Parameter(torch.zeros(2 , self.d_model))
        self.gnn = True
        self.norm = nn.LayerNorm(self.d_model)
        self.att_net = Attention(h=self.d_model, d_model=self.d_model)
        self.s_linear = torch.nn.Linear(self.d_model, 2)
        self.dotsim = DotProductSimilarity(scale_output=False)


    def forward(self, input_ids, token_type_ids, attention_mask, dplabel=None, rs=None, train=True):
        # input_ids: T(num_sentences, seq_len)
        # token_type_ids: T(num_sentences, seq_len)
        # attention_mask: T(num_sentences, seq_len)
        # rs: T(batch_size)
        # maps: T(batch_size, max_bag_size)
        # embedding: T(num_sentences, seq_len, embedding_size)
        bag_len, seq_len = input_ids.size()
        embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        # r_embedding: T(num_sentences, embedding_size)
        p_embedding = embedding[:, 0, :]
        if bag_len>8:
            print("bag_len:", bag_len)
        if rs is not None or not train:
            entity_mask, entity_span_list = self.get_htb(input_ids)
            h_embs = []
            t_embs = []
            b_embs = []
            dp_embs = []
            h_num = []
            t_num = []
            b_num = []
            for dp in range(0,bag_len):
                b_embs_dp = []
                try:
                    h_span = entity_span_list[dp][0]
                    t_span = entity_span_list[dp][1]
                    b_span_chunks = entity_span_list[dp][2]
                    h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1]+1], dim=0)[0]
                    t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1]+1], dim=0)[0]
                    h_embs.append(h_emb)
                    t_embs.append(t_emb)
                    for b_span in b_span_chunks:
                        b_emb = torch.max(embedding[dp, b_span[0]:b_span[1]+1], dim=0)[0]
                        b_embs_dp.append(b_emb)
                    if bag_len >= 16:
                        if len(b_embs_dp) > 3:
                            b_embs_dp = random.choices(b_embs_dp, k=3)
                    if bag_len >= 14:
                        if len(b_embs_dp) > 4:
                            b_embs_dp = random.choices(b_embs_dp, k=4)
                    elif bag_len >= 10:
                        if len(b_embs_dp) > 5:
                            b_embs_dp = random.choices(b_embs_dp, k=5)
                    else:
                        if len(b_embs_dp) > 8:
                            b_embs_dp = random.choices(b_embs_dp, k=8)
                        else:
                            b_embs_dp = b_embs_dp
                    b_embs.append(b_embs_dp)
                    h_num.append(1)
                    t_num.append(1)
                    b_num.append(len(b_embs_dp))
                    dp_embs.append(p_embedding[dp])
                except IndexError as e:
                    continue
            print(bag_len, b_num)
            htb_index = []
            htb_embs = []
            htb_start = [0]
            htb_end = []
            for h_emb, t_emb, b_emb in zip(h_embs, t_embs, b_embs):
                htb_embs.extend([h_emb,t_emb])
                htb_index.extend([1,2])
                htb_embs.extend(b_emb)
                htb_index.extend([3]*len(b_emb))
                htb_end.append(len(htb_index)-1)
                htb_start.append(len(htb_index))
            htb_start = htb_start[:-1]
            

            rel_mask = torch.ones(1,len(htb_index), len(htb_index)).to(embedding.device)
            try:
                htb_embs_t = torch.stack(htb_embs, dim=0).unsqueeze(0)
            except:
                print(input_ids)


            u = self.wu(htb_embs_t)
            v = self.wv(htb_embs_t)
            
            alpha = u.view(1, len(htb_index), 1, htb_embs_t.size()[-1]) + v.view(1, 1, len(htb_index), htb_embs_t.size()[-1]) 
            alpha = F.relu(alpha)

            rel_enco = F.relu(self.ln1(alpha))
            bs,es,es,d = rel_enco.size()

            rel_mask = torch.ones(1,len(htb_index), len(htb_index)).to(embedding.device)

            rel_enco_m = self.matt(rel_enco, rel_mask)
            h_pos = []
            t_pos = []
            for i, e_type in enumerate(htb_index):
                if e_type == 1:
                    h_pos.append(i)
                elif e_type == 2:
                    t_pos.append(i)
                else:
                    continue
            assert len(h_pos) == len(t_pos)
            rel_enco_m_ht = []
            for i,j in zip(h_pos, t_pos):
                rel_enco_m_ht.append(rel_enco_m[0][i][j])
            t_feature_m = torch.stack(rel_enco_m_ht)
            predict_logits = self.predictor(t_feature_m)
            ht_logits = predict_logits
            bag_logit = torch.max(ht_logits.transpose(0,1),dim=1)[0].unsqueeze(0)
            path_logit = ht_logits
        else:     # Inner doc
            entity_mask, entity_span_list = self.get_htb(input_ids) 
            path_logits = []
            ht_logits_flatten_list = []
            for dp in range(0,bag_len):
                h_embs = []
                t_embs = []
                b_embs = []
                try:
                    h_span = entity_span_list[dp][0]
                    t_span = entity_span_list[dp][1]
                    b_span_chunks = entity_span_list[dp][2]
                    h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1]+1], dim=0)[0]
                    t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1]+1], dim=0)[0]
                    h_embs.append(h_emb)
                    t_embs.append(t_emb)
                    for b_span in b_span_chunks:
                        b_emb = torch.max(embedding[dp, b_span[0]:b_span[1]+1], dim=0)[0]
                        b_embs.append(b_emb)
                    h_index = [1 for _ in h_embs]
                    t_index = [2 for _ in t_embs]
                    b_index = [3 for _ in b_embs]
                    htb_index = []
                    htb_embs = []
                    for idx, embs in zip([h_index, t_index, b_index], [h_embs, t_embs, b_embs]):
                        htb_index.extend(idx)
                        htb_embs.extend(embs)
                    rel_mask = torch.ones(1,len(htb_index), len(htb_index)).to(embedding.device)

                    htb_embs_t = torch.stack(htb_embs, dim=0).unsqueeze(0)

                    u = self.wu(htb_embs_t)
                    v = self.wv(htb_embs_t)
                    alpha = u.view(1, len(htb_index), 1, htb_embs_t.size()[-1]) + v.view(1, 1, len(htb_index), htb_embs_t.size()[-1])
                    alpha = F.relu(alpha)

                    rel_enco = F.relu(self.ln1(alpha))

                    rel_enco_m = self.matt(rel_enco , rel_mask)

                    t_feature = rel_enco_m
                    bs,es,es,d = rel_enco.size()

                    predict_logits = self.predictor(t_feature.reshape(bs,es,es,d))
                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    _ht_logits_flatten = ht_logits.reshape(1, -1, self.num_relations)
                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    path_logits.append(ht_logits)
                    ht_logits_flatten_list.append(_ht_logits_flatten)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
            try:
                path_logit = torch.stack(path_logits).reshape(1, 1, -1, self.num_relations).squeeze(0).squeeze(0)
            except Exception as e:
                print(e)
                pdb.set_trace()


        if dplabel is not None and rs is None:
            ht_logits_flatten = torch.stack(ht_logits_flatten_list).squeeze(1)
            ht_fixed_low = (torch.ones_like(ht_logits_flatten)*8)[:,:,0].unsqueeze(-1)
            y_true = torch.zeros_like(ht_logits_flatten)
            for idx, dpl in enumerate(dplabel):
                y_true[idx, 0, dpl.item()] = 1
            bag_logit = path_logit
            loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low+2, ht_fixed_low)
        elif rs is not None:
            _, prediction = torch.max(bag_logit, dim=1)
            if self.no_doc_pair_supervision:
                pass
            else:
                ht_logits_flatten = ht_logits.unsqueeze(1)
                y_true = torch.zeros_like(ht_logits_flatten)
                ht_fixed_low = (torch.ones_like(ht_logits_flatten)*8)[:,:,0].unsqueeze(-1)
                if rs.item() != 0:
                    for idx, dpl in enumerate(dplabel):
                        try:
                            y_true[idx, : , dpl.item()] = torch.ones_like(y_true[idx, : , dpl.item()])
                        except:
                            print("unmatched")
                            #pdb.set_trace()
                loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low+2, ht_fixed_low)
                
        else:
            ht_logits_flatten = ht_logits.unsqueeze(1)
            ht_fixed_low = (torch.ones_like(ht_logits_flatten)*8)[:,:,0].unsqueeze(-1)
            _, prediction = torch.max(bag_logit, dim=1)
            loss = None
        prediction = [] 
        return loss, prediction, bag_logit, ht_logits_flatten.transpose(0,1), (ht_fixed_low+2).transpose(0,1)


    def _multilabel_categorical_crossentropy(self, y_pred, y_true, cr_ceil, cr_low, ghm=True, r_dropout=True):
        # cr_low + 2 = cr_ceil
        y_pred = (1 - 2 * y_true) * y_pred  
        y_pred_neg = y_pred - y_true * 1e12  
        y_pred_pos = y_pred - (1 - y_true) * 1e12  
        y_pred_neg = torch.cat([y_pred_neg, cr_ceil], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, -cr_low], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return ((neg_loss + pos_loss + cr_low.squeeze(-1) - cr_ceil.squeeze(-1))).mean()

    def graph_encode(self , ent_encode , rel_encode , ent_mask , rel_mask):
        bs , ne , d = ent_encode.size()
        ent_encode = ent_encode + self.ent_emb[0].view(1,1,d)
        rel_encode = rel_encode + self.ent_emb[1].view(1,1,1,d)
        rel_encode , ent_encode = self.graph_enc(rel_encode , ent_encode , rel_mask , ent_mask)
        return rel_encode


    def get_htb(self, input_ids):
        htb_mask_list = []
        htb_list_batch = []
        for pi in range(input_ids.size()[0]):
            #pdb.set_trace()
            tmp = torch.nonzero(input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
            if tmp.size()[0] < input_ids.size()[0]:
                print(input_ids)
            try:
                h_starts = [i[0] for i in (input_ids[pi]==H_START_MARKER_ID).nonzero().detach().tolist()]
                h_ends = [i[0] for i in (input_ids[pi]==H_END_MARKER_ID).nonzero().detach().tolist()]
                t_starts = [i[0] for i in (input_ids[pi]==T_START_MARKER_ID).nonzero().detach().tolist()]
                t_ends = [i[0] for i in (input_ids[pi]==T_END_MARKER_ID).nonzero().detach().tolist()]
                if len(h_starts) == len(h_ends):
                    h_start = h_starts[0]
                    h_end = h_ends[0]
                else:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                                break
                if len(t_starts) == len(t_ends):
                    t_start = t_starts[0]
                    t_end = t_ends[0]
                else:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e
                                break
                if h_end-h_start<=0 or t_end-t_start<=0:
                    # print(h_starts)
                    # print(h_ends)
                    # print(t_starts)
                    # print(t_ends)
                    # pdb.set_trace()
                    if h_end-h_start<=0:
                        for h_s in h_starts:
                            for h_e in h_ends:
                                if 0 < h_e - h_s < 20:
                                    h_start = h_s
                                    h_end = h_e
                                    break
                    if t_end-t_start<=0:
                        for t_s in t_starts:
                            for t_e in t_ends:
                                if 0 < t_e - t_s < 20:
                                    t_start = t_s
                                    t_end = t_e
                                    break
                    if h_end-h_start<=0 or t_end-t_start<=0:
                        pdb.set_trace()
                        
                b_spans = torch.nonzero(torch.gt(torch.full(([input_ids.size()[1]]), 99).to(input_ids.device), input_ids[pi])).squeeze(0).squeeze(1).detach().tolist()
                token_len = input_ids[pi].nonzero().size()[0]
                b_spans = [i for i in b_spans if i <= token_len-1]
                assert len(b_spans) >= 4 
                for i in h_starts + h_ends + t_starts + t_ends:
                    b_spans.remove(i)
                h_span = [h_pos for h_pos in range(h_start, h_end+1)]
                t_span = [t_pos for t_pos in range(t_start, t_end+1)]
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(h_span).to(input_ids.device), 1)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(t_span).to(input_ids.device), 1)
            except:# dps＜8
                #pdb.set_trace()
                h_span = []
                t_span = []
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
                b_spans = []
            b_span_ = []
            if len(b_spans) > 0 and len(b_spans)%2==0:
                b_span_chunks = [b_spans[i:i+2] for i in range(0,len(b_spans),2)]
                b_span = []
                for span in b_span_chunks:
                    b_span.extend([b_pos for b_pos in range(span[0], span[1]+1)])
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span).to(input_ids.device), 1)
                b_span_.extend(b_span)
            elif len(b_spans) > 0 and len(b_spans)%2==1:
                b_span = []
                ptr = 0
                #pdb.set_trace()
                while(ptr<=len(b_spans)-1):
                    try:
                        if input_ids[pi][b_spans[ptr+1]] - input_ids[pi][b_spans[ptr]] == 1:
                            b_span.append([b_spans[ptr], b_spans[ptr+1]])
                            ptr += 2
                        else:
                            ptr += 1
                    except IndexError as e: 
                        ptr += 1 
                for bs in b_span:
                    #pdb.set_trace()
                    #ex_bs = range(bs[0], bs[1])
                    b_span_.extend(bs)
                    if len(b_span_)%2 != 0:
                        print(b_spans)
                b_span_chunks = [b_span_[i:i+2] for i in range(0,len(b_span_),2)]
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span_).to(input_ids.device), 1)
            else:
                b_span_ = []
                b_span_chunks = []
                b_mask = torch.zeros_like(input_ids[pi])
            htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0) #[3,512]
            htb_mask_list.append(htb_mask)
            htb_list_batch.append([h_span, t_span, b_span_chunks])
        # pdb.set_trace()
        htb_mask_batch = torch.stack(htb_mask_list,dim=0)
        return htb_mask_batch, htb_list_batch # 

def get_doc_entities(h, t, tokenizer, redisd, no_additional_marker, mask_entity, collec_doc1_titles, collec_doc2_titles):
    max_len = 99999
    bert_max_len = 512
    Doc1_tokens = []
    Doc2_tokens = []
    B_entities = []
    for doc1_title, doc2_title in zip(collec_doc1_titles, collec_doc2_titles):
        doc1 = json.loads(redisd.get('codred-doc-' + doc1_title))
        doc2 = json.loads(redisd.get('codred-doc-' + doc2_title))
        v_h = None
        for entity in doc1['entities']:
            if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
                v_h = entity
        assert v_h is not None
        v_t = None
        for entity in doc2['entities']:
            if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
                v_t = entity
        assert v_t is not None
        d1_v = dict()
        for entity in doc1['entities']:
            if 'Q' in entity:
                d1_v[entity['Q']] = entity
        d2_v = dict()
        for entity in doc2['entities']:
            if 'Q' in entity:
                d2_v[entity['Q']] = entity
        ov = set(d1_v.keys()) & set(d2_v.keys())
        if len(ov) > 40:
            ov = set(random.choices(list(ov), k=40)) 
        ov = list(ov)
        ma = dict()
        for e in ov:
            ma[e] = len(ma)
        B_entities.append(ma)
        
    # print(B_entities)

    return B_entities

class CodredCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_argument(self, parser):
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--aggregator', type=str, default='attention')
        parser.add_argument('--positive_only', action='store_true')
        parser.add_argument('--positive_ep_only', action='store_true')
        parser.add_argument('--no_doc_pair_supervision', action='store_true')
        parser.add_argument('--no_additional_marker', action='store_true')
        parser.add_argument('--mask_entity', action='store_true')
        parser.add_argument('--single_path', action='store_true')
        parser.add_argument('--dsre_only', action='store_true')
        parser.add_argument('--raw_only', action='store_true')
        parser.add_argument('--load_model_path', type=str, default=None)
        parser.add_argument('--train_file', type=str, default='../data/rawdata/train_dataset.json')
        parser.add_argument('--dev_file', type=str, default='../data/rawdata/dev_dataset.json')
        parser.add_argument('--test_file', type=str, default='../data/rawdata/test_dataset.json')
        parser.add_argument('--dsre_file', type=str, default='../data/dsre_train_examples.json')
        parser.add_argument('--model_name', type=str, default='bert')


    def load_model(self):
        relations = json.load(open('../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index
        with self.trainer.cache():
            reasoner = Codred(self.args, len(self.relations))
            if self.args.load_model_path:
                load_model(reasoner, self.args.load_model_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.sbert_wk = sbert(device='cuda')
        return reasoner

    def load_data(self):
        train_dataset = json.load(open(self.args.train_file))
        dev_dataset = json.load(open(self.args.dev_file))
        test_dataset = json.load(open(self.args.test_file))
        if self.args.positive_only:
            train_dataset = [d for d in train_dataset if d[3] != 'n/a']
            dev_dataset = [d for d in dev_dataset if d[3] != 'n/a']
            test_dataset = [d for d in test_dataset if d[3] != 'n/a']
        train_bags = place_train_data(train_dataset)
        dev_bags = place_dev_data(dev_dataset, self.args.single_path)
        test_bags = place_test_data(test_dataset, self.args.single_path)
        if self.args.positive_ep_only:
            train_bags = [b for b in train_bags if b[1] != 'n/a']
            dev_bags = [b for b in dev_bags if 'n/a' not in b[1]]
            test_bags = [b for b in test_bags if 'n/a' not in b[1]]
        self.dsre_train_dataset = json.load(open(self.args.dsre_file))
        self.dsre_train_dataset = [d for i, d in enumerate(self.dsre_train_dataset) if i % 10 == 0]
        d = list()
        for i in range(len(self.dsre_train_dataset) // 8):
            d.append(self.dsre_train_dataset[8 * i:8 * i + 8])
        if self.args.raw_only:
            pass
        elif self.args.dsre_only:
            train_bags = d
        else:
            d.extend(train_bags)
            train_bags = d
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True, db=0)
        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'], self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1, self.args.local_rank)
            self.test_logger = Logger(['test_mean_prec', 'test_f1', 'test_auc'], self.trainer.writer, 1, self.args.local_rank)
        return train_bags, dev_bags, test_bags

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, encoder = self.bert, sbert_wk=self.sbert_wk), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, encoder=self.bert, sbert_wk=self.sbert_wk), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd, encoder=self.bert, sbert_wk=self.sbert_wk)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            if inputs['rs'] is not None:
                _, prediction, logit, ht_logits_flatten, ht_threshold_flatten = outputs
                rs = extra['rs']
                if ht_logits_flatten is not None:
                    r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
                    if r_score>ht_threshold_flatten[0, 0, 0]:
                        prediction = [r_idx.item()]
                    else:
                        prediction = [0]

                for p, score, gold in zip(prediction, logit, rs):
                    self.train_logger.log(train_acc=1 if p == gold else 0)
                    if gold > 0:
                        self.train_logger.log(train_pos_acc=1 if p == gold else 0)
            else:
                _, prediction, logit, ht_logits_flatten, ht_threshold_flatten = outputs
                dplabel = inputs['dplabel']
                logit, dplabel = tensor_to_obj(logit, dplabel)
                prediction = []
                if ht_logits_flatten is not None:
                    r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
                    for dp_i, (r_s, r_i) in enumerate(zip(r_score, r_idx)):
                        if r_s > ht_threshold_flatten[dp_i, 0, 0]:
                            prediction.append(r_i.item())
                        else:
                            prediction.append(0)
                for p, l in zip(prediction, dplabel):
                    self.train_logger.log(train_dsre_acc=1 if p == l else 0)

    def on_train_epoch_end(self, epoch):
        #for k,v in self.train_logger.d:
        print(epoch, self.train_logger.d)
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        _, prediction, logit, ht_logits_flatten, ht_threshold_flatten = outputs
        r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
        eval_logit = torch.max(ht_logits_flatten,dim=1)[0]

        if r_score>ht_threshold_flatten[:, 0, 0]:
            prediction = [r_idx.item()]
        else:
            prediction = [0]
        h, t, rs = extra['h'], extra['t'], extra['rs']
        logit = tensor_to_obj(logit)
        self._prediction.append([prediction[0], eval_logit[0], h, t, rs])

    def on_dev_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        results = list()
        pred_result = list()
        facts = dict()
        for p, score, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        stat = eval_performance(facts, pred_result)
        with self.trainer.once():
            self.dev_logger.log(dev_mean_prec=stat['mean_prec'], dev_f1=stat['f1'], dev_auc=stat['auc'])
            json.dump(stat, open(f'output/dev-stat-dual-K1-{epoch}.json', 'w'))
            json.dump(results, open(f'output/dev-results-dual-K1-{epoch}.json', 'w'))
        return stat['f1']
    
    def on_test_epoch_start(self, epoch):
        self._prediction = list()
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        #_, prediction, logit = outputs
        #pdb.set_trace()
        _, prediction, logit, ht_logits_flatten, ht_threshold_flatten = outputs
        r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
        eval_logit = torch.max(ht_logits_flatten,dim=1)[0]

        if r_score>ht_threshold_flatten[0, 0, 0]:
            #prediction = [r_idx.item() + 1]
            prediction = [r_idx.item()]
        else:
            prediction = [0]
        h, t, rs = extra['h'], extra['t'], extra['rs']
        logit = tensor_to_obj(logit)
        self._prediction.append([prediction[0], eval_logit[0], h, t, rs])
        #self._prediction.append([prediction[0], logit[0], h, t, rs])

    def on_test_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        results = list()
        pred_result = list()
        facts = dict()
        out_results = list()
        coda_file = dict()
        coda_file['setting'] = 'closed'
        for p, score, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
                out_results.append({'h_id':str(h), "t_id":str(t), "relation": str(self.relations[i]), "score": float(score[i])})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        #stat = eval_performance(facts, pred_result)
        coda_file['predictions'] = out_results
        with self.trainer.once():
            json.dump(results, open(f'output/test-results-{epoch}.json', 'w'))
            json.dump(coda_file, open(f'output/test-codalab-results-{epoch}.json', 'w'))
        return True

    def process_train_data(self, data):
        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'rs':data[1],
            'dplabel': data[0],
            'train': True
        }
        return {'rs': data[2]}, selected_inputs, {'selected_rets': data[7]}

    def process_dev_data(self, data):
        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'train': False
        }
        return {'h': data[0], 'rs': data[1], 't': data[2]}, selected_inputs, {'selected_rets': data[7]}
    
    def process_test_data(self, data):

        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'train': False
        }
        return {'h': data[0], 'rs': data[1], 't': data[2]}, selected_inputs, {'selected_rets': data[7]}


def main():
    trainer = Trainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
