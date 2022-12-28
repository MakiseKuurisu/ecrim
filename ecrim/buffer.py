import torch
from copy import copy
from transformers import AutoTokenizer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME
import random
from bisect import bisect_left
from itertools import chain
import pdb
class Block:
    """Similar to CogLTX(https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf).
    """
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    def __init__(self, ids, pos, blk_type=1, **kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type
        self.relevance = 0
        self.estimation = 0
        self.entail_score = 0
        self.docid = 0
        self.h_flag = 0
        self.t_flag = 0
        self.__dict__.update(kwargs)
    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)
    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type
    def __len__(self):
        return len(self.ids)
    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))

class Buffer:
    @staticmethod
    def split_document_into_blocks(d, tokenizer, cnt=0, hard=True, properties=None, docid=0):
        ret = Buffer()
        updiv = lambda a,b: (a - 1) // b + 1
        if hard:
            for sid, tsen in enumerate(d):
                psen = properties[sid] if properties is not None else []
                num = updiv(len(tsen), BLOCK_SIZE) # cls
                bsize = updiv(len(tsen), num)
                for i in range(num):
                    st, en = i * bsize, min((i + 1) * bsize, len(tsen))
                    cnt += 1
                    tmp = tsen[st: en] + [tokenizer.sep_token]
                    # inject properties into blks
                    tmp_kwargs = {}
                    for p in psen:
                        if len(p) == 2:
                            tmp_kwargs[p[0]] = p[1]
                        elif len(p) == 3:
                            if st <= p[1] < en:
                                tmp_kwargs[p[0]] = (p[1] - st, p[2])
                        else:
                            raise ValueError('Invalid property {}'.format(p))
                    ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))
        else:
            # d is only a list of tokens, not split. 
            # properties are also a list of tuples.
            end_tokens = {'\n':0, '.':1, '?':1, '!':1, ',':2}
            for k, v in list(end_tokens.items()):
                end_tokens['Ġ' + k] = v
            sen_cost, break_cost = 4, 8
            poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
            poses.insert(0, (-1, 0))
            if poses[-1][0] < len(d) - 1:
                poses.append((len(d) - 1, 0))
            x = 0
            while x < len(poses) - 1:
                if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                    poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
                x += 1

            best = [(0, 0)]
            for i, (p, cost) in enumerate(poses):
                if i == 0:
                    continue
                best.append((-1, 100000)) 
                for j in range(i-1, -1, -1): 
                    if p - poses[j][0] > BLOCK_SIZE: 
                        break
                    value = best[j][1] + cost + sen_cost
                    if value < best[i][1]:
                        best[i] = (j, value)
                assert best[i][0] >= 0
            intervals, x = [], len(poses) - 1 
            while x > 0: 
                l = poses[best[x][0]][0 ]
                intervals.append((l + 1, poses[x][0] + 1))
                x = best[x][0] 
            if properties is None:
                properties = []
            for st, en in reversed(intervals):
                # copy from hard version
                cnt += 1
                tmp = d[st: en] + [tokenizer.sep_token]
                # inject properties into blks
                tmp_kwargs = {}
                for p in properties:
                    if len(p) == 2:
                        tmp_kwargs[p[0]] = p[1]
                    elif len(p) == 3:
                        if st <= p[1] < en:
                            tmp_kwargs[p[0]] = (p[1] - st, p[2])
                    else:
                        raise ValueError('Invalid property {}'.format(p))
                ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))
        for blk in ret.blocks:
            blk.docid = docid
        return ret, cnt 

    def __init__(self):
        self.blocks = []

    def __add__(self, buf):
        ret = Buffer()
        ret.blocks = self.blocks + buf.blocks
        return ret

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b)+'\n' for b in self.blocks])
        
    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def merge(self, buf):
        ret = Buffer()
        t1, t2 = 0, 0
        while t1 < len(self.blocks) or t2 < len(buf):
            if t1 < len(self.blocks) and (t2 >= len(buf) or self.blocks[t1] < buf.blocks[t2]):
                ret.blocks.append(self.blocks[t1])
                t1 += 1
            else:
                ret.blocks.append(buf.blocks[t2])
                t2 += 1
        return ret
    
    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret
            
    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret
    
    def sort_(self):
        self.blocks.sort()
        return self

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > CAPACITY:
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret

    def export(self, device=None, length=None, out=None):
        if out is None:
            if length is None:
                total_length = self.calc_size()
                if total_length > CAPACITY:
                    raise ValueError('export inputs larger than capacity')
            else:
                total_length = length * len(self.blocks)
            ids, att_masks, type_ids = torch.zeros(3, total_length, dtype=torch.long, device=device)
        else: # must be zeros and big enough
            ids, att_masks, type_ids = out
            att_masks.zero_()
        t = 0
        for b in self.blocks:
            try:
                ids[t:t + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device)
            except:
                #pdb.set_trace()
                ids[t-1 :t-1 + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device) 
                #pdb.set_trace()
            # if b.blk_type == 1:
            #     type_ids[t:w] = 1 # sentence B
            att_masks[t:t + len(b)] = 1 # attention_mask
            t += len(b) if length is None else length
        return ids, att_masks, type_ids
    
    def export_01_turn(self, device=None, length=None, out=None):
        if out is None:
            if length is None:
                total_length = self.calc_size()
                if total_length > CAPACITY:
                    raise ValueError('export inputs larger than capacity')
            else:
                total_length = length * len(self.blocks)
            ids, att_masks, type_ids = torch.zeros(3, total_length, dtype=torch.long, device=device)
        else: # must be zeros and big enough
            ids, att_masks, type_ids = out
            att_masks.zero_()
        t = 0
        for b in self.blocks:
            try:
                ids[t:t + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device) # id
            except:
                #pdb.set_trace()
                print("capacity:", 512 - t, "blk_length:", len(b))
                try:
                    ids[t-1 :t-1 + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device) 
                except: 
                    
                    print(ids, len(ids))
                    print(b.ids, len(b.ids))
                    try:
                        ids[t : -1] = torch.tensor(b.ids, dtype=torch.long, device=device)[:512 - t - 1]
                        ids[-1] = torch.tensor([102], dtype=torch.long, device=device)
                    except Exception as e:
                        print(e)
                        pdb.set_trace()
                #pdb.set_trace()
            # if b.blk_type == 1:
            #     type_ids[t:w] = 1 # sentence B
            att_masks[t:t + len(b)] = 1 # attention_mask
            t += len(b) if length is None else length
        sentences = []
        sentences_with_sep = []
        ptr = 0
        ids_list = ids.tolist()
        for i in range(len(ids_list)):
            if ids_list[i] == 102:
                sentences.append(ids_list[ptr:i])
                sentences_with_sep.append(ids_list[ptr:i+1])
                ptr = i+1
        sentences[-1].append(102)
        s_ptr = 0
        for s_idx in range(len(sentences_with_sep)):
            type_ids[s_ptr:s_ptr + len(sentences_with_sep[s_idx])] = torch.tensor([s_idx%2]* len(sentences_with_sep[s_idx]), dtype=torch.long, device=device)
            s_ptr += len(sentences_with_sep[s_idx])
        return ids, att_masks, type_ids

    def export_01_doc(self, device=None, length=None, out=None):
        if out is None:
            if length is None:
                total_length = self.calc_size()
                if total_length > CAPACITY:
                    raise ValueError('export inputs larger than capacity')
            else:
                total_length = length * len(self.blocks)
            ids, att_masks, type_ids = torch.zeros(3, total_length, dtype=torch.long, device=device)
        else: # must be zeros and big enough
            ids, att_masks, type_ids = out
            att_masks.zero_()
        t = 0
        
        #pdb.set_trace()
        doc0_ids = []
        doc1_ids = []
        for b in self.blocks:
            if b.docid == 0:
                doc0_ids.extend(b.ids)
            elif b.docid == 1:
                doc1_ids.extend(b.ids)
        #pdb.set_trace()
        ids[:len(doc0_ids)] = torch.tensor(doc0_ids, dtype=torch.long, device=device)
        try:        
            ids[len(doc0_ids):len(doc0_ids)+len(doc1_ids)] = torch.tensor(doc1_ids, dtype=torch.long, device=device)
        except:
            pdb.set_trace()
            print(doc1_ids)
            doc1_ids.reverse()
            doc1_ids.remove(102)
            doc1_ids.remove(102)
            doc1_ids.reverse()
            doc1_ids.append(102) 
            ids[len(doc0_ids):len(doc0_ids)+len(doc1_ids)] = torch.tensor(doc1_ids, dtype=torch.long, device=device)
        type_ids[:len(doc0_ids)] = torch.tensor([0]*len(doc0_ids), dtype=torch.long, device=device)
        type_ids[len(doc0_ids):len(doc0_ids)+len(doc1_ids)] = torch.tensor([1]*len(doc1_ids), dtype=torch.long, device=device)
        return ids, att_masks, type_ids

    def export_as_batch(self, device, length=BLOCK_SIZE+1, add_cls=False):
        ids, att_masks, type_ids = self.export(device, length, add_cls=add_cls)
        return ids.view(-1, length), att_masks.view(-1, length), type_ids.view(-1, length)

    def export_relevance(self, device, length=None, dtype=torch.long, out=None):
        if out is None:
            total_length = self.calc_size() if length is None else length * len(self.blocks)
            relevance = torch.zeros(total_length, dtype=dtype, device=device)
        else:
            relevance = out
        t = 0
        for b in self.blocks:
            w = t + (len(b) if length is None else length)
            if b.relevance >= 1:
                relevance[t: w] = 1
            t = w
        return relevance

def buffer_collate(batch): 
    return batch
