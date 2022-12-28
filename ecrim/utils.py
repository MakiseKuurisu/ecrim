CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
H_START_MARKER_ID = 1
H_END_MARKER_ID = 2
T_START_MARKER_ID = 3
T_END_MARKER_ID = 4
from prometheus_client import Enum
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Optional
def convert_caps(s):
    # ret = []
    # for word in s.split():
    #     if word[0].isupper():
    #         ret.append('<pad>')
    #     ret.append(word)
    # return ' '.join(ret).lower()    
    return s.lower()

import pdb
import sys
import math
class DotProductSimilarity(nn.Module):
 
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
 
    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class contrastive_pair:
    def __init__(self, rep, label):
        self.rep = rep
        self.label = torch.tensor(label, device=rep.device)
        self.n = rep.size()[0]
        self.T = 0.5
        self.device = rep.device

    def n_pair_loss(self):
        #pdb.set_trace()
        similarity_matrix = F.cosine_similarity(self.rep.unsqueeze(1), self.rep.unsqueeze(0), dim=2)
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (self.label.expand(self.n, self.n).eq(self.label.expand(self.n, self.n).t()))
        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask

        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = (torch.ones(self.n ,self.n) - torch.eye(self.n, self.n )).to(self.device)

        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.T)

        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0


        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix


        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim


        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(self.n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        if self.label.nonzero().sum().item()==0 or self.label.nonzero().sum().item()==self.n*(self.n-1)/2:
            sim_sum = sim_sum + torch.eye(self.n,self.n).to(self.device)
        loss = torch.div(sim , sim_sum)


        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim + loss + torch.eye(self.n, self.n).to(self.device)

        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        loss = torch.sum(torch.sum(loss, dim=1) )/(2*self.n)  #将所有数据都加起来除以2n

        return loss


class generate_embedding():

    def __init__(self, embed_method, masks):
        # Select from embedding methods
        switcher = {
            'ave_last_hidden': self.ave_last_hidden,
            'CLS': self.CLS,
            'dissecting': self.dissecting,
            'ave_one_layer': self.ave_one_layer,
        }
        
        self.masks = masks
        self.embed = switcher.get(embed_method, 'Not a valide method index.')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 'ave_last_hidden': self.ave_last_hidden,
    def ave_last_hidden(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
        
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1,:,:]
            embedding.append(np.mean(hidden_state_sen[:sent_len,:], axis=0))

        embedding = np.array(embedding)
        return embedding


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 'ave_last_hidden': self.ave_last_hidden,
    def ave_one_layer(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
        
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][params['layer_start'],:,:]
            embedding.append(np.mean(hidden_state_sen[:sent_len,:], axis=0))

        embedding = np.array(embedding)
        return embedding


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 'CLS': self.CLS,
    def CLS(self, params, all_layer_embedding):
        """
            CLS vector as embedding
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
        
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1,:,:]
            embedding.append(hidden_state_sen[0])

        embedding = np.array(embedding)
        return embedding

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 'dissecting': self.dissecting,
    def dissecting(self, params, all_layer_embedding):
        """
            dissecting deep contextualized model
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
        all_layer_embedding = np.array(all_layer_embedding)[:,params['layer_start']:,:,:] # Start from 4th layers output

        embedding = []
        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index,:,:unmask_num[sent_index],:]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:,token_index,:]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(params, token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = np.array(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(params, sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        embedding = np.array(embedding)

        return embedding

    def unify_token(self, params, token_feature):
        """
            Unify Token Representation
        """
        window_size = params['context_window_size']

        alpha_alignment = np.zeros(token_feature.shape[0])
        alpha_novelty = np.zeros(token_feature.shape[0])
        
        for k in range(token_feature.shape[0]):
         
            left_window = token_feature[k-window_size:k,:]
            right_window = token_feature[k+1:k+window_size+1,:]
            window_matrix = np.vstack([left_window, right_window, token_feature[k,:][None,:]])
            
            Q, R = np.linalg.qr(window_matrix.T) # This gives negative weights

            q = Q[:, -1]
            r = R[:, -1]
            alpha_alignment[k] = np.mean(normalize(R[:-1,:-1],axis=0),axis=1).dot(R[:-1,-1]) / (np.linalg.norm(r[:-1]))
            alpha_alignment[k] = 1/(alpha_alignment[k]*window_matrix.shape[0]*2)
            alpha_novelty[k] = abs(r[-1]) / (np.linalg.norm(r))
            
        
        # Sum Norm
        alpha_alignment = alpha_alignment / np.sum(alpha_alignment) # Normalization Choice
        alpha_novelty = alpha_novelty / np.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        
        alpha = alpha / np.sum(alpha) # Normalize
        
        out_embedding = token_feature.T.dot(alpha)
        

        return out_embedding

    def unify_sentence(self, params, sentence_feature, one_sentence_embedding):
        """
            Unify Sentence By Token Importance
        """
        sent_len = one_sentence_embedding.shape[0]

        var_token = np.zeros(sent_len)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:,token_index,:]
            sim_map = cosine_similarity(token_feature)
            var_token[token_index] = np.var(sim_map.diagonal(-1))

        var_token = var_token / np.sum(var_token)

        sentence_embedding = one_sentence_embedding.T.dot(var_token)

        return sentence_embedding

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)

class PosEmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(PosEmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)

        if pretrained:
            self.load_pretrained(pretrained, mapping)
        self.embedding.weight.requires_grad = not freeze

        self.drop = nn.Dropout(dropout)

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds

def PosMap(distance):
    if distance == 0:
        return 0
    elif distance <= 2:
        return 1
    elif distance <= 4:
        return 2
    elif distance <= 8:
        return 3
    elif distance <= 16:
        return 4
    elif distance <= 32:
        return 5
    elif distance <= 64:
        return 6
    elif distance <= 128:
        return 7
    elif distance <= 256:
        return 8
    elif distance <= 512:
        return 9
    else:
        return -1


def complete_h_t(all_buf, filtered_buf):
        h_markers = [1,2]
        t_markers = [3,4]
        for blk_id, blk in enumerate(filtered_buf.blocks):
            if blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
                if list(set(blk.ids).intersection(set(h_markers))) == [1]:  #向后补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(h_markers))) == [2]: #向前补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(1)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)
            elif blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers: # 2在1前
                pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == 1:
                        markers_starts.append(i)
                    elif id == 2:
                        markers_ends.append[i]
                    else:
                        continue
                if len(markers_starts) > len(markers_ends): # 1比2多，向后补1个2
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends): #2比1多，向前补1个1
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(1)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)

                else:
                    if blk.ids.index(2) > blk.ids.index(1):
                        pass
                    elif blk.ids.index(2) < blk.ids.index(1):
                        #pdb.set_trace()
                        first_end_marker = blk.ids.index(2)
                        second_start_marker = blk.ids.index(1)
                        # 向后补或向前补都行
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(2)
                        if 101 in complementary:
                            complementary.remove(101)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        filtered_buf[blk_id].ids = new[-len(old):] + [102]
                        print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
                if list(set(blk.ids).intersection(set(t_markers))) == [3]:  #向后补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(t_markers))) == [4]: #向前补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p = complementary.index(3)
                    marker_p_start = complementary.index(3)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                            
                        except Exception as e:
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers: # 4在3前
                pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == 3:
                        markers_starts.append(i)
                    elif id == 4:
                        markers_ends.append[i]
                    else:
                        continue
                if len(markers_starts) > len(markers_ends): # 3比4多，向后补1个4
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends): #4比3多，向前补1个3
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(3)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)
                else: # 3和4一样多，但4在3前
                    if blk.ids.index(4) > blk.ids.index(3):
                        pass
                    elif blk.ids.index(4) < blk.ids.index(3):
                        #pdb.set_trace()
                        first_end_marker = blk.ids.index(4)
                        second_start_marker = blk.ids.index(3)
                        # 向后补或向前补都行
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(4)
                        if 101 in complementary:
                            complementary.remove(101)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        filtered_buf[blk_id].ids = new[-len(old):] + [102]
                        print(filtered_buf[blk_id].ids)
        return filtered_buf

def complete_h_t_debug(all_buf, filtered_buf):
        h_markers = [1,2]
        t_markers = [3,4]
        pdb.set_trace()
        for blk_id, blk in enumerate(filtered_buf.blocks):
            if blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
                if list(set(blk.ids).intersection(set(h_markers))) == [1]:  #向后补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(h_markers))) == [2]: #向前补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(1)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)
            elif blk.h_flag==1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers: # 2在1前
                #pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == 1:
                        markers_starts.append(i)
                    elif id == 2:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends): # 1比2多，向后补1个2
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends): #2比1多，向前补1个1
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(1)
                    if blk.ids[0]!=101 and blk.ids[0]!=2:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    elif blk.ids[0]!=2:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 1:
                                complementary = [1]
                            else:
                                complementary = [1]
                    else:
                        complementary = all_buf[blk.pos-2].ids[marker_p_start:-1]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)

                else:
                    if blk.ids.index(2) > blk.ids.index(1):
                        pass
                    elif blk.ids.index(2) < blk.ids.index(1):
                        #pdb.set_trace()
                        first_end_marker = blk.ids.index(2)
                        second_start_marker = blk.ids.index(1)
                        # 向后补或向前补都行
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(2)
                        if 101 in complementary:
                            complementary.remove(101)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        filtered_buf[blk_id].ids = new[-len(old):] + [102]
                        print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
                if list(set(blk.ids).intersection(set(t_markers))) == [3]:  #向后补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(t_markers))) == [4]: #向前补
                    #pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p = complementary.index(3)
                    marker_p_start = complementary.index(3)
                    if blk.ids[0]!=101:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                            
                        except Exception as e:
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)

            elif blk.t_flag==1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers: # 4在3前
                #pdb.set_trace()
                markers_starts= []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == 3:
                        markers_starts.append(i)
                    elif id == 4:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends): # 3比4多，向后补1个4
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends): #4比3多，向前补1个3
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(3)
                    if blk.ids[0]!=101 and blk.ids[0]!=4:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    elif blk.ids[0]!=4:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == 102:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            #pdb.set_trace()
                            if complementary[-2] == 3:
                                complementary = [3]
                            else:
                                complementary = [3]
                    else:
                        complementary = all_buf[blk.pos-2].ids[marker_p_start:-1]
                    if blk.ids[0] != 101:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(101)
                        new = [101] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [102]
                    print(filtered_buf[blk_id].ids)
                else: # 3和4一样多，但4在3前
                    if blk.ids.index(4) > blk.ids.index(3):
                        pass
                    elif blk.ids.index(4) < blk.ids.index(3):
                        #pdb.set_trace()
                        first_end_marker = blk.ids.index(4)
                        second_start_marker = blk.ids.index(3)
                        # 向后补或向前补都行
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(4)
                        if 101 in complementary:
                            complementary.remove(101)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        filtered_buf[blk_id].ids = new[-len(old):] + [102]
                        print(filtered_buf[blk_id].ids)
        return filtered_buf


def check_htb(input_ids, h_t_flag):
        htb_mask_list = []
        htb_list_batch = []
        for pi in range(input_ids.size()[0]):
            #pdb.set_trace()
            tmp = torch.nonzero(input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
            if tmp.size()[0] < input_ids.size()[0]:
                print(input_ids)
            try:
                h_starts = [i[0] for i in (input_ids[pi]==1).nonzero().detach().tolist()]
                h_ends = [i[0] for i in (input_ids[pi]==2).nonzero().detach().tolist()]
                t_starts = [i[0] for i in (input_ids[pi]==3).nonzero().detach().tolist()]
                t_ends = [i[0] for i in (input_ids[pi]==4).nonzero().detach().tolist()]
                if len(h_starts) == len(h_ends):
                    h_start = h_starts[0]
                    h_end = h_ends[0]
                else:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                if len(t_starts) == len(t_ends):
                    t_start = t_starts[0]
                    t_end = t_ends[0]
                else:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e
                if h_end-h_start<=0 or t_end-t_start<=0:
                    print("H/T INDEX ERROR")
                    print(h_starts)
                    print(h_ends)
                    print(t_starts)
                    print(t_ends)
                    #pdb.set_trace()
                    if h_end-h_start<=0:
                        for h_s in h_starts:
                            for h_e in h_ends:
                                if 0 < h_e - h_s < 20:
                                    h_start = h_s
                                    h_end = h_e
                    if t_end-t_start<=0:
                        for t_s in t_starts:
                            for t_e in t_ends:
                                if 0 < t_e - t_s < 20:
                                    t_start = t_s
                                    t_end = t_e
                    
                    if h_end-h_start<=0 or t_end-t_start<=0:
                        pdb.set_trace()

                b_spans = torch.nonzero(torch.gt(torch.full(([input_ids.size()[1]]), 99).to(input_ids.device), input_ids[pi])).squeeze(0).squeeze(1).detach().tolist()
                token_len = input_ids[pi].nonzero().size()[0]
                b_spans = [i for i in b_spans if i <= token_len-1]
                assert len(b_spans) >= 4 #
                #for i in [h_start, h_end, t_start, t_end]:
                for i in h_starts + h_ends + t_starts + t_ends:
                    b_spans.remove(i)
                h_span = [h_pos for h_pos in range(h_start, h_end+1)]
                t_span = [t_pos for t_pos in range(t_start, t_end+1)]
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(h_span).to(input_ids.device), 1)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(t_span).to(input_ids.device), 1)
            except:# dps＜8 从而填充0的会导致索引错误,这里也填充为0向量
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
                    except IndexError as e: # 最后一个导致数组越界
                        #pdb.set_trace()
                        ptr += 1 # 末尾是一个孤立的marker，ptr+1跳出循环
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
        bag_len = input_ids.size()[0]
        for dp in range(0,bag_len):
            try:
                h_span = htb_list_batch[dp][0]   #[3,4,5,6,7]
                t_span = htb_list_batch[dp][1]   #[71,72,73,74]
                if h_span == [] or t_span == []:
                    #print("fail detecting h/t")
                    #h_t_flag = False
                    #pdb.set_trace()
                    # flag do not change
                    pass
                else:
                    #pdb.set_trace()
                    h_t_flag = True
                    #print("H/T Detected")
            except Exception as e:
                print(e)
                pdb.set_trace()
        return h_t_flag # entity_mask = T[8,3,512] h_mask = T[8, 512] t_mask = T[8,512] b_mask = T[8,512]

def check_htb_debug(input_ids, h_t_flag):
        htb_mask_list = []
        htb_list_batch = []
        pdb.set_trace()
        for pi in range(input_ids.size()[0]):
            #pdb.set_trace()
            tmp = torch.nonzero(input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
            if tmp.size()[0] < input_ids.size()[0]:
                print(input_ids)
            try:
                h_starts = [i[0] for i in (input_ids[pi]==1).nonzero().detach().tolist()]
                h_ends = [i[0] for i in (input_ids[pi]==2).nonzero().detach().tolist()]
                t_starts = [i[0] for i in (input_ids[pi]==3).nonzero().detach().tolist()]
                t_ends = [i[0] for i in (input_ids[pi]==4).nonzero().detach().tolist()]
                if len(h_starts) == len(h_ends):
                    h_start = h_starts[0]
                    h_end = h_ends[0]
                else:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                if len(t_starts) == len(t_ends):
                    t_start = t_starts[0]
                    t_end = t_ends[0]
                else:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e
                if h_end-h_start<=0 or t_end-t_start<=0:
                    # 出现了[1,325] + [321,339]的情况
                    print("H/T INDEX ERROR")
                    print(h_starts)
                    print(h_ends)
                    print(t_starts)
                    print(t_ends)
                    pdb.set_trace()
                    if h_end-h_start<=0:
                        for h_s in h_starts:
                            for h_e in h_ends:
                                if 0 < h_e - h_s < 20:
                                    h_start = h_s
                                    h_end = h_e
                    if t_end-t_start<=0:
                        for t_s in t_starts:
                            for t_e in t_ends:
                                if 0 < t_e - t_s < 20:
                                    t_start = t_s
                                    t_end = t_e
                    pdb.set_trace()
                b_spans = torch.nonzero(torch.gt(torch.full(([input_ids.size()[1]]), 99).to(input_ids.device), input_ids[pi])).squeeze(0).squeeze(1).detach().tolist()
                token_len = input_ids[pi].nonzero().size()[0]
                b_spans = [i for i in b_spans if i <= token_len-1]
                assert len(b_spans) >= 4 #
                for i in [h_start, h_end, t_start, t_end]:
                    b_spans.remove(i)
                h_span = [h_pos for h_pos in range(h_start, h_end+1)]
                t_span = [t_pos for t_pos in range(t_start, t_end+1)]
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(h_span).to(input_ids.device), 1)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(t_span).to(input_ids.device), 1)
            except:# dps＜8 从而填充0的会导致索引错误,这里也填充为0向量
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
                    except IndexError as e: # 最后一个导致数组越界
                        #pdb.set_trace()
                        ptr += 1 # 末尾是一个孤立的marker，ptr+1跳出循环
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
        bag_len = input_ids.size()[0]
        for dp in range(0,bag_len):
            try:
                h_span = htb_list_batch[dp][0]   #[3,4,5,6,7]
                t_span = htb_list_batch[dp][1]   #[71,72,73,74]
                if h_span == [] or t_span == []:
                    #print("fail detecting h/t")
                    #h_t_flag = False
                    #pdb.set_trace()
                    # flag do not change
                    pass
                else:
                    #pdb.set_trace()
                    h_t_flag = True
                    #print("H/T Detected")
            except Exception as e:
                print(e)
                pdb.set_trace()
        return h_t_flag # entity_mask = T[8,3,512] h_mask = T[8, 512] t_mask = T[8,512] b_mask = T[8,512]


def fix_entity_(doc, ht_markers, b_markers):
    #pdb.set_trace()
    """Temporarily replaces the "." inner the entity with "|", replaces the "," inner the entity with "~"
    to prevent the entity from being split in two during the block split process.
    
    Args:
        doc: .
        ht_markers: a list of markers that denotes target entities.
        b_markers: a list of markers that denotes bridge entities.

    Returns:
        doc: modified doc.
    """
    markers = ht_markers + b_markers
    markers_pos = []
    m_idx = 0
    if list(set(doc).intersection(set(markers))):
        while m_idx <= len(doc)-1:
            if doc[m_idx] in markers:
                markers_pos.append((m_idx, doc[m_idx]))
                m_idx += 1
            else:
                m_idx += 1

    
    idx = 0
    end_tokens = {'\n':0, '.':1, '?':1, '!':1, ',':2}
    while idx <= len(markers_pos)-1:
        try:
            assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) - int(markers_pos[idx+1][1].replace("[unused", "").replace("]", "")) == -1)
            entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
            if "." in entity_name and "," in entity_name:
                #pdb.set_trace()
                while "." in entity_name:
                    #print(entity_name)
                    #pdb.set_trace()
                    assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    print(doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]])
                    entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                while "," in entity_name:
                    #print(entity_name)
                    #pdb.set_trace()
                    assert doc[markers_pos[idx][0] + entity_name.index(",") + 1] == ","
                    doc[markers_pos[idx][0] + entity_name.index(",") + 1] = "~"
                    print(doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]])
                    entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                idx += 2
            elif "." in entity_name:
                while "." in entity_name:
                    #print(entity_name)
                    #pdb.set_trace()
                    assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    print(doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]])
                    entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                idx += 2
            elif "," in entity_name:
                while "," in entity_name:
                    #print(entity_name)
                    #pdb.set_trace()
                    assert doc[markers_pos[idx][0] + entity_name.index(",") + 1] == ","
                    doc[markers_pos[idx][0] + entity_name.index(",") + 1] = "~"
                    print(doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]])
                    entity_name = doc[markers_pos[idx][0]+1: markers_pos[idx + 1][0]]
                idx += 2
            else:
                idx += 2
        except:
            #pdb.set_trace()
            idx += 1
            continue
    return doc


def fix_entity(doc, ht_markers, b_markers):
    """Temporarily replaces the "." inner the entity with "|"
    to prevent the entity from being split in two during the block split process.
    
    Args:
        doc: .
        ht_markers: a list of markers that denotes target entities.
        b_markers: a list of markers that denotes bridge entities.

    Returns:
        doc: modified doc.
    """
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