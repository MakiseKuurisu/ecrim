import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
import pdb
from tqdm import tqdm
from itertools import groupby
from topological_sort import Graph
import random
class SentReOrdering():
    def __init__(self, doc1_sentences, doc2_sentences, encoder, device, tokenizer, h, t, sbert_wk):
        self.encoder = encoder
        self.doc1_sentences = doc1_sentences
        self.doc2_sentences = doc2_sentences
        self.device = device
        self.max_len = 512
        self.tokenizer = tokenizer
        self.h = h
        self.t = t
        self.sentences = self.doc1_sentences + self.doc2_sentences
        self.sbert = sbert_wk

    def pair_encoding(self):
        pairs = []
        for i,sent_1 in tqdm(enumerate(self.sentences)):
            for j,sent_2 in tqdm(enumerate(self.sentences)):
                if sent_1 != sent_2:
                    pair_len = len(sent_1) + len(sent_2) + 4
                    input_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                    token_type_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                    attention_mask = torch.zeros(1, self.max_len, dtype=torch.long)
                    token_type_ids[0][:pair_len] = torch.tensor([[0] * (len(sent_1)+2) + [1] * (len(sent_2)+2)]).unsqueeze(0)
                    input_ids[0][:pair_len] = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sent_1 + ['[SEP]'] + ['CLS'] + sent_2 + ['[SEP]'])).unsqueeze(0)
                    attention_mask[0][:pair_len] = torch.ones([1] * pair_len)
                    pair_encoding = self.encoder(input_ids, attention_mask, token_type_ids)[0]
                    sent_1_embedding = pair_encoding[0, :len(sent_1)+2][0]
                    sent_2_embedding = pair_encoding[0, len(sent_1)+2:len(sent_1)+len(sent_2)+4][0]
                    similarity = F.cosine_similarity(sent_1_embedding, sent_2_embedding, dim=0)
                    F.cosine_similarity(sent_1_embedding.unsqueeze(0), sent_2_embedding.unsqueeze(0), dim=0)
                    pairs.append((i, j, similarity.item()))
            else:
                continue
        return pairs
    
    def half_pair_encoding(self):
        pairs_1_with_2 = []
        pairs_2_with_1 = []
        for i,sent_1 in enumerate(self.doc1_sentences):
            for j,sent_2 in enumerate(self.doc2_sentences):
                pair_len = len(sent_1) + len(sent_2) + 4
                input_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                token_type_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                attention_mask = torch.zeros(1, self.max_len, dtype=torch.long)
                token_type_ids[0][:pair_len] = torch.tensor([[0] * (len(sent_1)+2) + [1] * (len(sent_2)+2)]).unsqueeze(0)
                input_ids[0][:pair_len] = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sent_1 + ['[SEP]'] + ['CLS'] + sent_2 + ['[SEP]'])).unsqueeze(0)
                attention_mask[0][:pair_len] = torch.ones([1] * pair_len)
                pair_encoding = self.encoder(input_ids, attention_mask, token_type_ids)[0]
                sent_1_embedding = pair_encoding[0, :len(sent_1)+2][0]
                sent_2_embedding = pair_encoding[0, len(sent_1)+2:len(sent_1)+len(sent_2)+4][0]
                similarity = F.cosine_similarity(sent_1_embedding, sent_2_embedding, dim=0)
                F.cosine_similarity(sent_1_embedding.unsqueeze(0), sent_2_embedding.unsqueeze(0), dim=0)
                pairs_1_with_2.append((i, j, similarity.item()))
        for i,sent_1 in enumerate(self.doc2_sentences):
            for j,sent_2 in enumerate(self.doc1_sentences):
                pair_len = len(sent_1) + len(sent_2) + 4
                input_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                token_type_ids = torch.zeros(1, self.max_len, dtype=torch.long)
                attention_mask = torch.zeros(1, self.max_len, dtype=torch.long)
                token_type_ids[0][:pair_len] = torch.tensor([[0] * (len(sent_1)+2) + [1] * (len(sent_2)+2)]).unsqueeze(0)
                input_ids[0][:pair_len] = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sent_1 + ['[SEP]'] + ['CLS'] + sent_2 + ['[SEP]'])).unsqueeze(0)
                attention_mask[0][:pair_len] = torch.ones([1] * pair_len)
                pair_encoding = self.encoder(input_ids, attention_mask, token_type_ids)[0]
                sent_1_embedding = pair_encoding[0, :len(sent_1)+2][0]
                sent_2_embedding = pair_encoding[0, len(sent_1)+2:len(sent_1)+len(sent_2)+4][0]
                similarity = F.cosine_similarity(sent_1_embedding, sent_2_embedding, dim=0)
                F.cosine_similarity(sent_1_embedding.unsqueeze(0), sent_2_embedding.unsqueeze(0), dim=0)
                pairs_2_with_1.append((i, j, similarity.item()))        

        return pairs_1_with_2, pairs_2_with_1

    def sentence_ordering(self):
        pairs = self.pair_encoding()
        # start
        Selected = []
        pair_start = [p for p in pairs if p[0]==0].sort(reverse=True)[0]
        Selected.append(pair_start[1])
        pair_next = [p for p in pairs if p[0]==Selected[-1]].sort(reverse=True)[0]
        score_max = 0
        pdb.set_trace()
        while(len(Selected)<=8):
            pair_next = [p for p in pairs if p[0]==Selected[-1]].sort(reverse=True)[0]
            score_max = 0
            for p_n in pair_next:
                if p_n[2] > score_max:
                    score_max = p_n[2]
                    candidate = p_n[1]
                else:
                    continue
            Selected.append(candidate)
        return Selected


    def half_ordering(self):
        Insert_2_to_1 = []
        Insert_1_to_2 = []
        pairs_1_with_2, pairs_2_with_1 = self.half_pair_encoding()
        doc1_num = len(self.doc1_sentences)
        doc2_num = len(self.doc2_sentences)
        Selected = []
        for s_2_idx in range(doc2_num):
            s_2_map = list((filter(lambda pair: pair[1] == s_2_idx, pairs_1_with_2)))
            head_idx = sorted(s_2_map, key=lambda sims: sims[2], reverse=True)[0][0]
            Insert_2_to_1.append((s_2_idx, '->', head_idx))
        for s_1_idx in range(doc1_num):
            s_1_map = list((filter(lambda pair: pair[1] == s_1_idx, pairs_2_with_1)))
            head_idx = sorted(s_1_map, key=lambda sims: sims[2], reverse=True)[0][0]
            Insert_1_to_2.append((s_1_idx, '->', head_idx))
        to_be_removed_2_to_1 = []
        to_be_removed_1_to_2 = []
        for i_2_to_1 in Insert_2_to_1:
            for i_1_to_2 in Insert_1_to_2:
                if i_2_to_1[0]==i_1_to_2[2] and i_2_to_1[2]==i_1_to_2[0]: # symmetric
                    Selected.append(self.doc1_sentences[i_2_to_1[2]])
                    Selected.append(self.doc2_sentences[i_1_to_2[2]])
                    to_be_removed_2_to_1.append(i_2_to_1)
                    to_be_removed_1_to_2.append(i_1_to_2)

        for tb_r_2_1 in to_be_removed_2_to_1:
            Insert_2_to_1.remove(tb_r_2_1)
        for tb_r_1_2 in to_be_removed_1_to_2:
            Insert_1_to_2.remove(tb_r_1_2)
        pdb.set_trace()
        max_score = 0
        chain = []
        for rest_pair in Insert_1_to_2:
            s_1_score = 0
            s_1_idx = rest_pair[0]
            s_1_map = list((filter(lambda pair: pair[0] == s_1_idx, pairs_1_with_2)))
            for mp in s_1_map:
                s_1_score += mp[2]
            if s_1_score > max_score:
                max_score = s_1_score
                chain_start = s_1_idx
        chain.append(chain_start)
        return Selected

    
    def half_sbert_encoding(self):
        pairs_1_with_2 = []
        pairs_2_with_1 = []
        for i, sent_prior in enumerate(self.doc1_sentences):
            for j, sent_later in enumerate(self.doc2_sentences):
                similarity = self.sbert.pair_sims(" ".join(sent_prior), " ".join(sent_later))
                pairs_1_with_2.append((i, j, similarity.item()))
        for i, sent_prior in enumerate(self.doc2_sentences):
            for j, sent_later in enumerate(self.doc1_sentences):
                similarity = self.sbert.pair_sims(" ".join(sent_prior), " ".join(sent_later))
                pairs_2_with_1.append((i, j, similarity.item()))
        return pairs_1_with_2, pairs_2_with_1        

    def sbert_encoding(self):
        pairs = []
        sentences = self.doc1_sentences + self.doc2_sentences
        for i, sent_prior in enumerate(sentences):
            for j, sent_later in enumerate(sentences):
                if i!=j:
                    similarity = self.sbert.pair_sims(" ".join(sent_prior), " ".join(sent_later))
                    pairs.append((i, j, similarity.item()))
                    pairs.append((j, i, similarity.item()))
        return pairs
    
    def peer_encoding(self, h_idx):
        pairs = []
        sentences = self.doc1_sentences + self.doc2_sentences
        sent_prior = sentences[h_idx]
        for j, sent_later in tqdm(enumerate(sentences)):
            if j!=h_idx:
                similarity = self.sbert.pair_sims(" ".join(sent_prior), " ".join(sent_later))
                pairs.append((h_idx, j, similarity.item()))
            else:
                pairs.append((h_idx, j, 0))
        return pairs
                

    def generate_edges(self):
        Edge_2_to_1 = []
        Edge_1_to_2 = []
        pairs_1_with_2, pairs_2_with_1 = self.half_sbert_encoding()
        doc1_num = len(self.doc1_sentences)
        doc2_num = len(self.doc2_sentences)
        for s_2_idx in range(doc2_num):
            s_2_map = list((filter(lambda pair: pair[1] == s_2_idx, pairs_1_with_2)))
            head_idx = sorted(s_2_map, key=lambda sims: sims[2], reverse=True)[0][0]
            Edge_2_to_1.append((s_2_idx, '->', head_idx))
        for s_1_idx in range(doc1_num):
            s_1_map = list((filter(lambda pair: pair[1] == s_1_idx, pairs_2_with_1)))
            head_idx = sorted(s_1_map, key=lambda sims: sims[2], reverse=True)[0][0]
            Edge_1_to_2.append((s_1_idx, '->', head_idx))
        return Edge_2_to_1, Edge_1_to_2

    def topo_sort(self):
        doc1_num = len(self.doc1_sentences)
        doc2_num = len(self.doc2_sentences)
        Edge_2_to_1, Edge_1_to_2 = self.generate_edges()
        Edge_1_to_1 = [(i, '->', i+1) for i in range(doc1_num-1)]
        Edge_2_to_2 = [(i, '->', i+1) for i in range(doc2_num-1)]
        nvert = doc1_num + doc2_num
        g = Graph(nvert)
        for edge in Edge_2_to_1: 
            pos_start = edge[2]
            pos_end   = edge[0] + doc1_num
            g.addEdge(pos_start, pos_end, 1)
        for edge in Edge_1_to_2:
            pos_start = edge[2] + doc1_num
            pos_end = edge[0]
            g.addEdge(pos_start, pos_end, 1) 
        for edge in Edge_1_to_1: 
            pos_s2 = edge[0] 
            pos_s1 = edge[2]
            g.addEdge(pos_s2, pos_s1, 1)
        for edge in Edge_2_to_2:
            pos_s1 = edge[0] + doc1_num
            pos_s2 = edge[2] + doc1_num
            g.addEdge(pos_s1, pos_s2, 1) 
        while g.isCyclic():
            g.isCyclic()
        order = g.topologicalSort()
        return order

    def all_sort(self, starts, ends):
        s_e_pairs = []
        for start in starts:
            for end in ends:
                s_e_pairs.append((start, end))
        pairs = self.sbert_encoding()
        chains = []
        
        for s_e_pair in s_e_pairs:
            start = s_e_pair[0]
            end = s_e_pair[1]
            chain = []
            chain.append(start)
            peers = list((filter(lambda pair: pair[0] == start and pair[1] not in chain, pairs)))
            next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
            while next_blk != end:
                chain.append(next_blk)
                peers = list((filter(lambda pair: pair[0] == next_blk and pair[1] not in chain, pairs)))
                next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
            chain.append(next_blk)
            chains.append(chain)
        return chains

    def dynamic_sort(self, starts, ends):
        s_e_pairs = []
        for start in starts:
            for end in ends:
                s_e_pairs.append((start, end))
        pairs = self.sbert_encoding()
        chains = []
        
        for s_e_pair in s_e_pairs:
            start = s_e_pair[0]
            end = s_e_pair[1]
            chain = []
            chain.append(start)
            pairs = self.peer_encoding(start)
            #pdb.set_trace()
            peers = list((filter(lambda pair: pair[0] == start and pair[1] not in chain, pairs)))
            next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
            while next_blk != end:
                chain.append(next_blk)
                pairs = self.peer_encoding(next_blk)
                peers = list((filter(lambda pair: pair[0] == next_blk and pair[1] not in chain, pairs)))
                next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
            chain.append(next_blk)
            if chain not in chains:
                chains.append(chain)
            else:
                continue
        return chains
    
    def semantic_based_sort(self, starts, ends):
        start = random.choice(starts)
        chain = []
        chain.append(start)
        pairs = self.peer_encoding(start)
        peers = list((filter(lambda pair: pair[0] == start and pair[1] not in chain, pairs)))
        next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
        while next_blk not in ends:
            chain.append(next_blk)
            pairs = self.peer_encoding(next_blk)
            peers = list((filter(lambda pair: pair[0] == next_blk and pair[1] not in chain, pairs)))
            next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
        chain.append(next_blk)
        rest = list(set([i for i in range(len(self.sentences))]) - set(chain))
        random.shuffle(rest)
        if len(chain) < 8:
            chain.extend(rest[:8-len(chain)])
        else:
            pass
        return [chain]


    def unsemantic_based_sort(self, starts, ends):
        start = random.choice(starts)
        end = random.choice(ends)
        
        chain = []
        chain.append(start)

        sentences = self.doc1_sentences + self.doc2_sentences
        s_ids = list(set([idx for idx,_ in enumerate(sentences)]).difference(set([start,end])))
        random.shuffle(s_ids)
        others = min(6, len(s_ids)-2)
        chain.extend(s_ids[:others+1])
        chain.append(end)
        return [chain]

    def bidirection_sort(self, starts, ends):
        s_e_pairs = []
        for start in starts:
            for end in ends:
                s_e_pairs.append((start, end))
        chains = []
        
        for s_e_pair in s_e_pairs:
            start = s_e_pair[0]
            end = s_e_pair[1]
            chain_head = []
            chain_tail = []
            chain_head.append(start)
            chain_tail.append(end)

            pairs_head = self.peer_encoding(start)
            pairs_tail = self.peer_encoding(end)

            peers_head = list((filter(lambda pair: pair[0] == start and pair[1] not in chain_head, pairs_head)))
            next_blk_head = sorted(peers_head, key=lambda sims: sims[2], reverse=True)[0][1]
            peers_tail = list((filter(lambda pair: pair[0] == end and pair[1] not in chain_tail, pairs_tail)))
            next_blk_tail = sorted(peers_tail, key=lambda sims: sims[2], reverse=True)[0][1]
            while next_blk_head != end and len(chain_head)<4:
                chain_head.append(next_blk_head)
                pairs_head = self.peer_encoding(next_blk_head)
                peers_head = list((filter(lambda pair: pair[0] == next_blk_head and pair[1] not in chain_head, pairs_head)))
                next_blk_head = sorted(peers_head, key=lambda sims: sims[2], reverse=True)[0][1]
            while next_blk_tail != start and len(chain_tail)<4:
                chain_tail.append(next_blk_tail)
                pairs_tail = self.peer_encoding(next_blk_tail)
                peers_tail = list((filter(lambda pair: pair[0] == next_blk_tail and pair[1] not in chain_tail, pairs_tail)))
                next_blk_tail = sorted(peers_tail, key=lambda sims: sims[2], reverse=True)[0][1]
            if next_blk_head==end or next_blk_tail==start:
                if next_blk_head==end:
                    chain_head.append(next_blk_head)
                    chain = chain_head
                if next_blk_tail==start:
                    chain_tail.append(next_blk_tail)
                    chain_tail.reverse()
                    chain = chain_tail
            else:
                chain_tail.reverse()
                chain = merge_chain(chain_head=chain_head, chain_tail=chain_tail)
            chains.append(chain)
        return chains

    def threeSent(self, starts, ends, co_occur):
        def consecutive_path(starts, ends):
            chain = []
            for start in starts:
                chain.append(start)
                pairs = self.peer_encoding(start)
                peers = list((filter(lambda pair: pair[0] == start and pair[1] not in chain, pairs)))
                next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
                while next_blk not in ends and len(chain)<=2:
                    chain.append(next_blk)
                    pairs = self.peer_encoding(next_blk)
                    peers = list((filter(lambda pair: pair[0] == next_blk and pair[1] not in chain, pairs)))
                    next_blk = sorted(peers, key=lambda sims: sims[2], reverse=True)[0][1]
                chain.append(next_blk)
                if len(set(chain).intersection(set(ends))) > 0 :
                    break
                else:
                    chain = []
                    continue
            return chain
        def multihop_path(starts, ends, co_occur):
            ori_co_occur = [i for i in co_occur]
            path = []
            edges_tuple = []
            start_pos = {}
            end_pos = {}
            start_edges = list((filter(lambda co: co[0]==1 and co[2] in starts, co_occur)))
            end_edges = list((filter(lambda co: co[0]==2 and co[2] in ends, co_occur)))
            for s_ed in start_edges:
                start_pos[s_ed[2]]=s_ed[1]
                edges_tuple.append((1, s_ed[1]))
            for e_ed in end_edges:
                end_pos[e_ed[2]]=e_ed[1]
                edges_tuple.append((2, e_ed[1]))
            co_occur = list(set(co_occur).difference(set(start_edges)).difference(set(end_edges)))
            if len(start_edges)>0 and len(end_edges)>0:
                next_set = []
                pre_set = []
                next_pos = {}
                pre_pos = {}
                for s_ed in start_edges:
                    next_set.append(s_ed[1])
                    next_pos[s_ed[2]]=s_ed[1]
                    edges_tuple.append((s_ed[0], s_ed[1]))
                next_set = set(next_set)
                for e_ed in end_edges:
                    pre_set.append(e_ed[1])
                    pre_pos[e_ed[2]]=e_ed[1]
                    edges_tuple.append((e_ed[0], e_ed[1]))
                pre_set = set(pre_set)
            while(len(next_set.intersection(pre_set))==0 and len(co_occur)>0):
                start_edges = list((filter(lambda co: co[0] in list(next_set), co_occur)))
                end_edges = list((filter(lambda co: co[0] in list(pre_set), co_occur)))
                co_occur = list(set(co_occur).difference(set(start_edges)).difference(set(end_edges)))
                next_set = list(next_set)
                pre_set = list(pre_set)
                for s_ed in start_edges:
                    next_set.append(s_ed[1])
                    next_pos[s_ed[2]]=s_ed[1]
                    edges_tuple.append((s_ed[0], s_ed[1]))
                next_set = set(next_set)
                for e_ed in end_edges:
                    pre_set.append(e_ed[1])
                    pre_pos[e_ed[2]]=e_ed[1]
                    edges_tuple.append((e_ed[0], e_ed[1]))
                pre_set = set(pre_set)
            entity_chain = merge_chain(list(next_set)+[1], [2]+list(pre_set))
            print(entity_chain)
            edges_tuple = list(set(edges_tuple))
            path_edges = list((filter(lambda co: co[0] in entity_chain and co[1] in entity_chain, edges_tuple)))
            path_triplet = list((filter(lambda e: (e[0], e[1]) in path_edges, ori_co_occur)))
            path = [o[2] for o in path_triplet]
            return path

        def default_path(starts, ends, co_occur, max_pos):
            chain = starts + ends
            if len(chain) >= 8:
                chain = [starts[0]] + [ends[0]] + random.choices(list(set(starts+ends).difference(set([starts[0]] + [ends[0]]))), k=6)
            else:
                while(len(chain)<=7):
                    offset = [-3,-2,-1,1,2,3]
                    ex_blk = random.choices(chain,k=1)[0] + random.choices(offset, k=1)[0]
                    if 0<=ex_blk<=max_pos-1 and ex_blk not in chain:
                        chain.append(ex_blk)
                    else:
                        continue
            return chain

        c_path = consecutive_path(starts, ends)
        s_e, e_e = multihop_path(starts, ends, co_occur)
        d_path = default_path(starts, ends, co_occur, len(self.sentences))
        if len(c_path)>0:
            path = c_path
        elif len(d_path)>0:
            path = d_path
        else:
            path = []
        print(path)
        return [path]

def merge_chain(chain_head, chain_tail):
    overlap_blk = list(set(chain_head).intersection(set(chain_tail)))
    if len(overlap_blk) >= 1:
        #pdb.set_trace()
        print(overlap_blk)
    for ov in overlap_blk:
        chain_tail.remove(ov)
    merged = chain_head + chain_tail
    return merged
