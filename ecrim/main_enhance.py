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

def if_h_t_complete(buffer):
        h_flag = False
        t_flag = False
        h_markers = [1,2]
        t_markers = [3,4]
        for ret in buffer:
            if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
                continue
            else:
                if ret.ids.index(1) < ret.ids.index(2):
                    h_flag = True
                else:
                    continue
        for ret in buffer:
            if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
                continue
            else:
                if ret.ids.index(3) < ret.ids.index(4):
                    t_flag = True
                else:
                    continue
        if h_flag and t_flag:
            return True
        else:
            return False

def bridge_entity_based_filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk, doc_entities, dps_count):
    alpha = 1
    beta = 0.1
    gamma = 0.01
    K = 16
    def complete_h_t(all_buf, filtered_buf):
        h_markers = [1,2]
        t_markers = [3,4]
        for blk_id, blk in enumerate(filtered_buf.blocks):
            if blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
                if list(set(blk.ids).intersection(set(h_markers))) == [H_START_MARKER_ID]:
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new)<=63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                elif list(set(blk.ids).intersection(set(h_markers))) == [H_END_MARKER_ID]:
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0]!=CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
            elif blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers: 
                #pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == H_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == H_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends): 
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new)<=63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends): 
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0]!=CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

                else:   
                    if blk.ids.index(H_END_MARKER_ID) > blk.ids.index(H_START_MARKER_ID):
                        pass
                    elif blk.ids.index(H_END_MARKER_ID) < blk.ids.index(H_START_MARKER_ID):
                        first_end_marker = blk.ids.index(H_END_MARKER_ID)
                        second_start_marker = blk.ids.index(H_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(H_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        if len(new)<=63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        # print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
                if list(set(blk.ids).intersection(set(t_markers))) == [T_START_MARKER_ID]:
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new)<=63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(t_markers))) == [T_END_MARKER_ID]:
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p = complementary.index(T_START_MARKER_ID)
                    marker_p_start = complementary.index(T_START_MARKER_ID)
                    if blk.ids[0]!=CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                            
                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers:
                #pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == T_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == T_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new)<=63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(3)
                    if blk.ids[0]!=CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                else: 
                    if blk.ids.index(T_END_MARKER_ID) > blk.ids.index(T_START_MARKER_ID):
                        pass
                    elif blk.ids.index(T_END_MARKER_ID) < blk.ids.index(T_START_MARKER_ID):
                        first_end_marker = blk.ids.index(T_END_MARKER_ID)
                        second_start_marker = blk.ids.index(T_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(T_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        if len(new)<=63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID:
                            filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
                        else:
                            continue
                        # print(filtered_buf[blk_id].ids)
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID and filtered_buf[blk_id].ids[0] not in [1,3]:
                if len(filtered_buf[blk_id].ids) <=63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    #pdb.set_trace()
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
            elif filtered_buf[blk_id].ids[0] in [1,3]:
                if len(filtered_buf[blk_id].ids) <=63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                    #pdb.set_trace()
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID or filtered_buf[blk_id].ids[-1] != SEP_TOKEN_ID:
                pdb.set_trace()
            else:
                pass
        return filtered_buf
        
        
    def detect_h_t(tokenizer, buffer):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        h_blocks = []
        t_blocks = []
        for blk in buffer:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                h_blocks.append(blk)
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                t_blocks.append(blk)
            else:
                continue
        return h_blocks, t_blocks
    
    def if_h_t_complete(buffer):
        h_flag = False
        t_flag = False
        h_markers = [1,2]
        t_markers = [3,4]
        for ret in buffer:
            if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
                continue
            else:
                if ret.ids.index(1) < ret.ids.index(2):
                    h_flag = True
                else: 
                    if len(list(set(ret.ids).intersection(set([2])))) > len(list(set(ret.ids).intersection(set([1])))):
                        h_flag = True
                    else:
                        continue
        for ret in buffer:
            if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
                continue
            else:
                if ret.ids.index(3) < ret.ids.index(T_END_MARKER_ID):
                    t_flag = True
                else:
                    if len(list(set(ret.ids).intersection(set([T_END_MARKER_ID])))) > len(list(set(ret.ids).intersection(set([3])))): 
                        t_flag = True
                    else:
                        continue
        if h_flag and t_flag:
            return True
        else:
            return False
    
    def co_occur_graph(tokenizer, h, t, d0, d1, doc_entities, alpha, beta, gamma, dps_count):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []
        d = []

        for di in [d0, d1]:
            d.extend(di)
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
        dbuf = Buffer()
        dbuf.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf.blocks:
            if blk.ids[0]!= CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids
            
        
        co_occur_pair = []
        for blk in dbuf:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))) and list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idx = list(set([math.ceil(int(b_m)/2) for b_m in list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))[0]
                co_occur_pair.append((1, b_idx, blk.pos))
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))) and list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idx = list(set([math.ceil(int(b_m)/2) for b_m in list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))[0]
                co_occur_pair.append((2, b_idx, blk.pos))
            elif list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idxs = list(set([math.ceil(int(b_m)/2) for b_m in list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))
                if len(b_idxs)>=2: 
                    pairs = combinations(b_idxs, 2) 
                else:
                    pairs = []
                for pair in pairs:
                    co_occur_pair.append((pair[0], pair[1], blk.pos))
            else:
                continue
       
        h_co = list((filter(lambda pair: pair[0] == 1, co_occur_pair)))
        t_co = list((filter(lambda pair: pair[0] == 2, co_occur_pair)))
        b_co = list((filter(lambda pair: pair[0] > 2 , co_occur_pair)))

        score_b = dict()
        s1 = dict()
        s2 = dict()
        s3 = dict()

        for entity_id in range(1, math.ceil((len(b_markers))/2) + 2):
            s1[entity_id] = 0 
            s2[entity_id] = 0 
            s3[entity_id] = 0 
            score_b[entity_id] = 0

        for pair in co_occur_pair:
            if pair[0] <= 2:
                s1[pair[1]] = 1
        
        for pair in b_co:
            if s1[pair[0]] == 1:
                s2[pair[1]] += 1

            if s1[pair[1]] == 1:
                s2[pair[0]] += 1

        
        bridge_ids = {doc_entities[dps_count][key]: key for key in doc_entities[dps_count].keys()}
        for idx in range(len(doc_entities)):
            if idx == dps_count:
                continue
            else:
                ent_ids = doc_entities[idx].keys()
                for k,v in bridge_ids.items():
                    if v in ent_ids:
                        s3[k+3] += 1
                    else:
                        continue

        for entity_id in range(1, math.ceil((len(b_markers))/2) + 2):
            score_b[entity_id] += alpha * s1[entity_id] + beta * s2[entity_id] + gamma * s3[entity_id]
        # pdb.set_trace()
        return score_b
    
    def get_block_By_sentence_score(tokenizer, h, t, d0, d1, score_b, K):
        #pdb.set_trace()
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []
        d = []

        score_b_positive = [(k,v) for k,v in score_b.items() if v>0]
        score_b_positive_ids = []
        for b in score_b_positive:
            b_id = b[0]
            b_score = b[1]
            score_b_positive_ids.append(2*b_id-1)
            score_b_positive_ids.append(2*b_id)


        
        #pdb.set_trace()
        for di in [d0, d1]:
            d.extend(di)
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
        dbuf_all = Buffer()
        dbuf_all.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf_all.blocks:
            if blk.ids[0]!= CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                blk.h_flag = 1
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                blk.t_flag = 1
        
        
        for blk in dbuf_all:
            if len(list(set(score_b_positive_ids).intersection(set(blk.ids)))) > 0:
                blk_bridge_marker_ids = list(set(score_b_positive_ids).intersection(set(blk.ids)))
                blk_bridge_ids = list(set([math.ceil(int(b_m_id)/2) for b_m_id in blk_bridge_marker_ids]))
                for b_id in blk_bridge_ids:
                    blk.relevance += score_b[b_id]
                #print(blk.pos, blk.relevance)
            else:
                blk.relevance = 0
                continue
        
        #pdb.set_trace()
        for blk in dbuf_all:
            if blk.h_flag == 1 or blk.t_flag==1:
                blk.relevance += 1
            else:
                continue
            
        block_scores = dict()
        for blk in dbuf_all:
            block_scores[blk.pos] = blk.relevance
            #print(blk.pos, blk.relevance)
        
        block_scores = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)

        try:
            score_threshold = block_scores[K][1]
        except IndexError as e:
            h_blocks = []
            t_blocks = []
            if not if_h_t_complete(dbuf_all):
                #pdb.set_trace()
                dbuf_all = complete_h_t(dbuf_all, dbuf_all)
                if not if_h_t_complete(dbuf_all):
                    pdb.set_trace()
                    dbuf_all = complete_h_t_debug(dbuf_all, dbuf_all)
            else:
                pass
            for blk in dbuf_all:
                if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                    h_blocks.append(blk)
                elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                    t_blocks.append(blk)
                else:
                    continue
            return h_blocks, t_blocks, dbuf_all, dbuf_all
        
        score_highest = block_scores[0][1]
        if score_threshold > 0 or score_highest > 0:
            p_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance > score_threshold, need_residue=True)
            e_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance == score_threshold, need_residue=True)
        else:
            p_buf, e_buf = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)
 
        if len(p_buf)+len(e_buf) == K: 
            dbuf_filtered = p_buf + e_buf
        elif len(p_buf)+len(e_buf) < K:
            _, rest_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance < score_threshold, need_residue=True)
            dbuf_filtered = p_buf + e_buf + random.sample(rest_buf, K - len(p_buf) - len(e_buf))
            assert len(dbuf_filtered) <= K
        else:
            try:
                highest_blk_id = sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].pos
            except:
                if score_threshold > 0 or score_highest > 0:
                    highest_blk_id = sorted(e_buf, key=lambda x: x.relevance, reverse=True)[0].pos
                else:
                    detect_h_t(tokenizer, dbuf_filtered)
            e_buf_selected_blocks = []
            try:
                if sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].relevance > 0:
                    e_buf_distance = dict()
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = abs(e.pos - highest_blk_id)
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K-len(p_buf)]] 
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            except:
                if e_buf[0].relevance > 0:
                    e_buf_distance = dict()
                    ht_buf, _ = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in ht_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K-len(p_buf)]] 
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
                else:
                    e_buf_distance = dict()
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in p_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K-len(p_buf)]] 
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            dbuf_blocks = p_buf.blocks + e_buf_selected_blocks
            dbuf_filtered = Buffer()
            for block in dbuf_blocks:
                dbuf_filtered.insert(block)

        h_blocks = []
        t_blocks = []
        for blk in dbuf_filtered:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                h_blocks.append(blk)
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                t_blocks.append(blk)
            else:
                continue
        if len(h_blocks) == 0 or len(t_blocks) == 0:
            new_dbuf = Buffer()
            ori_dbuf_all_blocks = sorted(dbuf_all.blocks, key=lambda x:x.relevance * 0.01 + (x.h_flag + x.t_flag), reverse=True)
            ori_dbuf_filtered_blocks = sorted(dbuf_filtered.blocks, key=lambda x:x.relevance * 0.01 + (x.h_flag + x.t_flag), reverse=True)
            if len(h_blocks) == 0:
                candi_h_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.h_flag:
                        candi_h_blocks.append(blk)
                    else:
                        continue
                h_blocks.append(random.choice(candi_h_blocks))
                new_dbuf.insert(h_blocks[0])
            if len(t_blocks) == 0:
                candi_t_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.t_flag:
                        candi_t_blocks.append(blk)
                        #break
                    else:
                        continue
                t_blocks.append(random.choice(candi_t_blocks))
                new_dbuf.insert(t_blocks[0])
            for ori_blk in ori_dbuf_filtered_blocks:
                if len(new_dbuf) <= K-1:
                    new_dbuf.insert(ori_blk)
                else:
                    break
            dbuf_filtered = new_dbuf

        h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
        all_block_pos = [blk.pos for blk in dbuf_filtered]
        if len(set(all_block_pos).intersection(set(h_t_block_pos))) != len(set(h_t_block_pos)):
            if len(set(all_block_pos).intersection(set(h_t_block_pos))) < len(set(h_t_block_pos)):
                h_blocks = [blk for blk in dbuf_filtered if blk.h_flag == 1]
                t_blocks = [blk for blk in dbuf_filtered if blk.t_flag == 1]
                h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
                assert len(set(all_block_pos).intersection(set(h_t_block_pos))) == len(set(h_t_block_pos))
            else:
                pdb.set_trace()
        if not if_h_t_complete(dbuf_filtered):
            dbuf_filtered = complete_h_t(dbuf_all, dbuf_filtered)
            if not if_h_t_complete(dbuf_filtered):
                pdb.set_trace()
                dbuf_filtered = complete_h_t_debug(dbuf_all, dbuf_filtered)
            
        else:
            pass
        return  h_blocks, t_blocks, dbuf_filtered, dbuf_all



    score_b = co_occur_graph(tokenizer, h, t, doc0, doc1, doc_entities, alpha, beta, gamma, dps_count)
    h_blocks, t_blocks, dbuf, dbuf_all = get_block_By_sentence_score(tokenizer, h, t, doc0, doc1, score_b, K)
    if len(h_blocks) == 0 or len(t_blocks) == 0:
        pdb.set_trace()
    h_t_flag = False
    dbuf_concat = []
    for blk in dbuf:
        dbuf_concat.extend(blk.ids)
    h_t_flag = check_htb(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag)
    if not h_t_flag:
        pdb.set_trace()
        h_t_flag = check_htb_debug(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag)
    else:
        pass
    return h_blocks, t_blocks, dbuf, dbuf_all


def sent_Filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk, doc_entities, dps_count):

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
                idx += 1
                continue
        return doc

    ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
    b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
    doc0 = fix_entity(doc0, ht_markers, b_markers)
    doc1 = fix_entity(doc1, ht_markers, b_markers)

    h_blocks, t_blocks, dbuf, dbuf_all = bridge_entity_based_filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk, doc_entities, dps_count)


    sentence_blocks = dbuf.blocks
    block_pos = [blk.pos for blk in dbuf]
    order_start_blocks = [blk.pos for blk in h_blocks]
    order_end_blocks = [blk.pos for blk in t_blocks]
    if len(order_start_blocks)==0  or len(order_end_blocks)==0:
        pdb.set_trace()

    doc_0_blks = [blk for blk in sentence_blocks if blk.docid==0]
    doc_1_blks = [blk for blk in sentence_blocks if blk.docid==1]

    doc_0_sentences = [tokenizer.convert_ids_to_tokens(blk.ids) for blk in doc_0_blks]
    doc_1_sentences = [tokenizer.convert_ids_to_tokens(blk.ids) for blk in doc_1_blks]

    try:
        order_starts = [block_pos.index(pos) for pos in order_start_blocks]
        order_ends = [block_pos.index(pos) for pos in order_end_blocks]
    except:
        pdb.set_trace()

    for s in doc_0_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')
    for s in doc_1_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')
    #pdb.set_trace()
    sro = SentReOrdering(doc_0_sentences, doc_1_sentences, encoder=encoder, device='cuda', tokenizer=tokenizer, h=h, t=t, sbert_wk=sbert_wk)
    orders = sro.semantic_based_sort(order_starts, order_ends)
    # for order in orders:
        # print(order)
    selected_buffers = []
    for order in orders:
        selected_buffer = Buffer()
        if len(order)<=8:
            for od in order:
                try:
                    selected_buffer.insert(sentence_blocks[od])
                except Exception as e:
                    pdb.set_trace()
        else:
            print(order)
            
            o_scores = dict()
            for o in order[1:-1]:
                o_scores[o] = sentence_blocks[o].relevance
            o_scores = sorted(o_scores.items(),key=lambda s:s[1], reverse=True)
            while len(order)>8:
                lowest_score = o_scores[-1][1]
                removable = list((filter(lambda o_score: o_score[1] == lowest_score, o_scores)))
                if len(removable) >= 1:
                    random.shuffle(removable)
                    remove_o = removable[0][0]
                order.remove(remove_o)
                o_scores.remove((remove_o,lowest_score))
            assert len(order) <= 8
            for od in order:
                try:
                    selected_buffer.insert(sentence_blocks[od])
                except Exception as e:
                    pdb.set_trace()
        selected_buffers.append(selected_buffer)
    #pdb.set_trace()
    return selected_buffers, dbuf_all


def process_example_ReoS(h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity, encoder, sbert_wk, doc_entities, dps_count):
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
    
    selected_order_rets, dbuf_all = sent_Filter(tokenizer, v_h['name'], v_t['name'], k1, k2, encoder, sbert_wk, doc_entities, dps_count)

    if len(selected_order_rets) == 0:
        print("SELECTION FAIL")
        pdb.set_trace()
        return []
    else:
        pass
    #pdb.set_trace()
    h_flag = False
    t_flag = False
    h_markers = [1,2]
    t_markers = [3,4]
    for selected_order_ret in selected_order_rets:
        for ret in selected_order_ret:
            if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
                continue
            else:
                h_flag = True
        for ret in selected_order_ret:
            if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
                continue
            else:
                t_flag = True
        if h_flag and t_flag:
            pass
        else:
            pdb.set_trace()
            completed_selected_ret = complete_h_t(dbuf_all, selected_order_ret)
            if not if_h_t_complete(completed_selected_ret):
                completed_selected_ret = complete_h_t_debug(dbuf_all, selected_order_ret)
            selected_order_ret, dbuf_all = sent_Filter(tokenizer, v_h['name'], v_t['name'], k1, k2, encoder, sbert_wk, doc_entities, dps_count)
            selected_order_rets[0] = completed_selected_ret
    return selected_order_rets[0]

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
        collec_doc1_titles = [doc1 for doc1,_,_ in dps]
        collec_doc2_titles = [doc2 for _,doc2,_ in dps]
        doc_entities = get_doc_entities(h, t, tokenizer, redisd, args.no_additional_marker, args.mask_entity, collec_doc1_titles, collec_doc2_titles)
        dps_count = 0
        for doc1, doc2, l in dps:
            selected_ret = process_example_ReoS(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, encoder, sbert_wk, doc_entities, dps_count)
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
        
        collec_doc1_titles = [doc1 for doc1,_,_ in dps]
        collec_doc2_titles = [doc2 for _,doc2,_ in dps]
        doc_entities = get_doc_entities(h, t, tokenizer, redisd, args.no_additional_marker, args.mask_entity, collec_doc1_titles, collec_doc2_titles)
        dps_count = 0
        selected_ret = process_example_ReoS(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity, encoder, sbert_wk, doc_entities, dps_count)
    
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
        self.gcn = GCN(hidden_channels=self.d_model, dim_node_features=self.d_model, num_classes=2)


    def forward(self, input_ids, token_type_ids, attention_mask, dplabel=None, rs=None, train=True):
        bag_len, seq_len = input_ids.size()
        embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
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
            graph_output_r = []
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
            t_feature = torch.stack(rel_enco_m_ht)
            
            # Get multi instance level Graph Features
            dp_rel_enco = []
            for dp in range(0,bag_len):
                b_ptr = 0
                htb_embs_list = [h_embs[dp], t_embs[dp]] + b_embs[dp]
                htb_embs_dp = torch.stack(htb_embs_list)
                ee = self.wi(htb_embs_dp)
                htb_index_dp = [1,2] + [3 for _ in range(b_num[dp])]
                rel_enco_dp = ee.view(1, len(htb_index_dp), 1, htb_embs_dp.size()[-1]) + ee.view(1, 1, len(htb_index_dp), htb_embs_dp.size()[-1]) 
                rel_enco_dp = F.relu(rel_enco_dp)
                b_ptr += b_num[dp]
                dp_rel_enco.append(rel_enco_dp)

            r_nodes = []
            r_node_features = []
            r_ht_embs = []
            seq_mask = [[1]*i + [0]*(512-i) for i in (torch.sum(input_ids!=0, dim=-1))]
            for p_idx, r_emb in enumerate(dp_rel_enco):
                r_num = r_emb.view(r_emb.size()[0], -1, r_emb.size()[-1]).size()[1]
                r_nodes.append([i for i in range(r_num)])
                #for r in r_emb:
                r_node_features.append(r_emb.view(r_emb.size()[0], -1, r_emb.size()[-1]).squeeze(0))
                r_ht_embs.append(r_emb[0, 0, 0, :])
            #r_embs = torch.stack(r_node_features)
            r_embs = torch.concat(r_node_features).to(p_embedding.device)
            p_embs = p_embedding
            #pdb.set_trace()
            r_ht_embs = torch.stack(r_ht_embs).unsqueeze(1).to(p_embedding.device)
            p_mask = torch.tensor(seq_mask).to(p_embedding.device)
            r_mask = torch.tensor([1]* r_ht_embs.size()[0]).unsqueeze(-1).unsqueeze(-1).to(p_embedding.device)
            #pdb.set_trace()
            attentive_tokens_embedding = self.att_net(embedding, r_ht_embs, p_mask, r_mask)

            seq_index = torch.nonzero(torch.where(input_ids == 102, torch.tensor(102).to(p_embedding.device), torch.tensor(0).to(p_embedding.device))).tolist()
            # seq_embs = []
            dp_set = set(map(lambda x:x[0], seq_index))
            # seq_index_dp = [[y[1] for y in seq_index if y[0]==x] for x in dp_set]
            # for dp in range(input_ids.size()[0]):
            #     seq_emb_dp = []
            #     for idx, pos in enumerate(seq_index_dp[dp]):
            #         if idx==0:
            #             start_pos = 0
            #             end_pos = pos
            #             seq_emb_dp.append(torch.mean(attentive_tokens_embedding[dp][start_pos+1:end_pos+1], dim=0)) # 不包括CLS，包括SEP
            #         else:
            #             start_pos = seq_index_dp[dp][idx-1]
            #             end_pos = pos
            #             seq_emb_dp.append(torch.mean(attentive_tokens_embedding[dp][start_pos-1:end_pos+1], dim=0)) # start_pos-1是去掉上一句的SEQ，end_pos+1是包括自己的SEP
            #     if len(seq_emb_dp)<8:
            #         ex_seq = random.choices(seq_emb_dp, k=8-len(seq_emb_dp))
            #         seq_emb_dp.extend(ex_seq)
            #     else:
            #         pass
            #     assert len(seq_emb_dp)==8
            #     seq_embs.append(torch.stack(seq_emb_dp))
            
            # sentence_embedding = torch.stack(seq_embs)
            token_logits = self.s_linear(attentive_tokens_embedding)
            gumbel_logits = torch.nn.functional.gumbel_softmax(token_logits[:,:,0], tau=0.2)
            attentive_gumbel_embedding = torch.sum((attentive_tokens_embedding.permute(2,0,1)*gumbel_logits).permute(1,2,0), dim=1)
            #sentence_gumbel_embedding = torch.sum((sentence_embedding.permute(2,0,1)*torch.nn.functional.gumbel_softmax(self.s_linear(sentence_embedding)[:,:,0], tau=0.2)).permute(1,2,0), dim=1)
            #bridge_nodes = (sentence_gumbel_embedding.view(1, sentence_gumbel_embedding.size()[0], sentence_gumbel_embedding.size()[1])+sentence_gumbel_embedding.view(sentence_gumbel_embedding.size()[0], 1, sentence_gumbel_embedding.size()[1])).view(-1,sentence_gumbel_embedding.size()[-1])
            bridge_nodes = (attentive_gumbel_embedding.view(1, attentive_gumbel_embedding.size()[0], attentive_gumbel_embedding.size()[1])+attentive_gumbel_embedding.view(attentive_gumbel_embedding.size()[0], 1, attentive_gumbel_embedding.size()[1])).view(-1,attentive_gumbel_embedding.size()[-1])

            g = create_graph(r_nodes, r_embs, p_embs, bridge_nodes)
            print(g)
            graph_output = self.gcn(g.x, g.edge_index, torch.tensor([g.x.size(0)], device=p_embedding.device))
            nodes_offset = p_embs.size()[0]
            # p_nodes = graph_output[:dps]
            # r_nodes = graph_output[dps:-(dps*dps)]
            # b_nodes = graph_output[-(dps*dps):]
            for r_node in r_nodes:
                graph_output_r.append(graph_output[nodes_offset:nodes_offset+len(r_node)])
                nodes_offset += len(r_node)

            bs,es,es,d = rel_enco_m.size()
            g_feature = torch.stack([graph_output_r[i][0] for i in range(len(graph_output_r))])
            if self.gnn:
                predict_logits = self.predictor(self.norm(t_feature+ g_feature))
            else:
                predict_logits = self.predictor(t_feature)


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

                    ee = self.wi(htb_embs_t)
                    htb_index_dp = [1,2] + [3 for _ in range(len(b_embs))]
                    rel_enco_dp = ee.view(1, len(htb_index_dp), 1, htb_embs_t.size()[-1]) + ee.view(1, 1, len(htb_index_dp), htb_embs_t.size()[-1]) 
                    rel_enco_dp = F.relu(rel_enco_dp)
                    r_nodes = [[i for i in range(len(htb_index)*len(htb_index))]]
                    r_embs = rel_enco_dp.view(-1,self.d_model).squeeze(0)
                    p_embs = p_embedding[dp].unsqueeze(0)
                    g = create_graph_single(r_nodes, r_embs, p_embs)
                    #print(g)
                    graph_output = self.gcn(g.x, g.edge_index, torch.tensor([g.x.size(0)], device=p_embedding.device))
                    nodes_offset = p_embs.size()[0]
                    g_feature = graph_output[nodes_offset:nodes_offset+len(r_nodes[0])]


                    if self.gnn:
                        predict_logits = self.predictor(self.norm(t_feature.reshape(bs,es,es,d) + g_feature.reshape(bs,es,es,d)))
                    else:
                        predict_logits = self.predictor(t_feature.reshape(bs,es,es,d))

                    predict_logits = self.predictor(t_feature.reshape(bs,es,es,d))

                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    _ht_logits_flatten = ht_logits.reshape(1, -1, self.num_relations)

                    ht_logits = predict_logits[0][:len(h_index), len(h_index):len(h_index)+len(t_index)]
                    path_logits.append(ht_logits)
                    ht_logits_flatten_list.append(_ht_logits_flatten)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
            #pdb.set_trace()
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
                #for i in [h_start, h_end, t_start, t_end]:
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
                    b_span_.extend(bs)
                    if len(b_span_)%2 != 0:
                        print(b_spans)
                b_span_chunks = [b_span_[i:i+2] for i in range(0,len(b_span_),2)]
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span_).to(input_ids.device), 1)
            else:
                b_span_ = []
                b_span_chunks = []
                b_mask = torch.zeros_like(input_ids[pi])
            htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0)
            htb_mask_list.append(htb_mask)
            htb_list_batch.append([h_span, t_span, b_span_chunks])
        htb_mask_batch = torch.stack(htb_mask_list,dim=0)
        return htb_mask_batch, htb_list_batch 

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
        _, prediction, logit, ht_logits_flatten, ht_threshold_flatten = outputs
        r_score, r_idx = torch.max(torch.max(ht_logits_flatten,dim=1)[0], dim=-1)
        eval_logit = torch.max(ht_logits_flatten,dim=1)[0]

        if r_score>ht_threshold_flatten[0, 0, 0]:
            prediction = [r_idx.item()]
        else:
            prediction = [0]
        h, t, rs = extra['h'], extra['t'], extra['rs']
        logit = tensor_to_obj(logit)
        self._prediction.append([prediction[0], eval_logit[0], h, t, rs])

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
