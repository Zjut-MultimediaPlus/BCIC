import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
#import misc.utils as utils
import os
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class AttentionLayer(nn.Module):
  def __init__(self, conv_channels, embed_dim):
    super(AttentionLayer, self).__init__()
    self.in_projection = Linear(conv_channels, embed_dim)
    self.out_projection = Linear(embed_dim, conv_channels)
    self.bmm = torch.bmm

  def forward(self, x, wordemb, imgsfeats):
    residual = x

    x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)

    b, c, f_h, f_w = imgsfeats.size()
    y = imgsfeats.view(b, c, f_h*f_w)

    x = self.bmm(x, y)

    sz = x.size()
    x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
    x = x.view(sz)
    attn_scores = x

    y = y.permute(0, 2, 1)

    x = self.bmm(x, y)

    s = y.size(1)
    x = x * (s * math.sqrt(1.0 / s))

    x = (self.out_projection(x) + residual) * math.sqrt(0.5)

    return x, attn_scores

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)    


class convcap_G(nn.Module):
  
    #def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=0.1):
    def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=0.1):
        super(convcap_G, self).__init__()
        self.nimgfeats = 2048
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout

        self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)    # Linear(9221, 512)
        self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

        self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
        self.resproj = Linear(nfeats*2, self.nfeats, dropout=dropout)

        n_in = 2 * self.nfeats
        n_out = self.nfeats
        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = 5
        self.pad = self.kernel_size - 1
        for i in range(self.n_layers):
            self.convs.append(Conv1d(n_in, 2*n_out, self.kernel_size, self.pad, dropout))
            if(self.is_attention):
                self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out

        self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
        self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

        '''
        self.input_encoding_size = 512
        self.rnn_size = 512
        self.drop_prob_lm = 0.5
        self.fc_feat_size = 2048
        self.att_feat_size = 2048
        self.att_hid_size = 512
        self.seq_per_img = 5
        self.index_eval = 0
        self.use_rela = False
        self.vocab_size = 14964
        self.use_bn = False

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.embed2vis = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))

        self.rela_sbj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_obj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_attr_fc = nn.Sequential(nn.Linear(self.rnn_size*2, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)    ## nn.Linear(512, 512)
        '''

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        return fc_feats, att_feats

    def prepare_rela_feats(self, rela_data):
        """
        Change relationship index (one-hot) to relationship features, or change relationship
        probability to relationship features.
        :param rela_matrix:
        :param rela_masks:
        :return: rela_features, [N_img*5, N_rela_max, rnn_size]
        """
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        rela_feats_size = rela_matrix.size()
        N_att = rela_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        rela_feats = torch.zeros([rela_feats_size[0], rela_feats_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            if N_rela>0:
                rela_index = rela_matrix[img_id*seq_per_img,:N_rela,2].cuda().long()
                rela_feats_temp = self.embed(rela_index)
                rela_feats_temp = self.embed2vis(rela_feats_temp)
                rela_feats[img_id*seq_per_img:(img_id+1)*seq_per_img,:N_rela,:] = rela_feats_temp
        rela_data['rela_feats'] = rela_feats
        return rela_data

    def rela_graph_gfc(self, rela_data):
        """
        :param att_feats: roi features of each bounding box, [N_img*5, N_att_max, rnn_size]
        :param rela_feats: the embeddings of relationship, [N_img*5, N_rela_max, rnn_size]
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param att_masks: attention masks, [N_img*5, N_att_max].
                            For each row, the sum of that row is the total number
                            of roi poolings.
        :param attr_matrix: attribute matrix,[N_img*5, N_attr_max, N_attr_each_max]
                            N_img is the batch size, N_attr_max is the maximum number
                            of attributes of one mini-batch, N_attr_each_max is the
                            maximum number of attributes of each objects in that mini-batch
        :param attr_masks: attribute masks, [N_img*5, N_attr_max, N_attr_each_max]
                            the sum of attr_masks[img_id*5,:,0] is the number of objects
                            which own attributes, the sum of attr_masks[img_id*5, obj_id, :]
                            is the number of attribute that object has
        :return: att_feats_new: new roi features
                 rela_feats_new: new relationship embeddings
                 attr_feats_new: new attribute features
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_matrix = rela_data['rela_matrix']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_matrix = rela_data['attr_matrix']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        attr_masks_size = attr_masks.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        att_feats_new = att_feats.clone()
        rela_feats_new = rela_feats.clone()
        attr_feats_new = torch.zeros([attr_masks_size[0], attr_masks_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            #N_box = torch.sum(att_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            #N_box = int(N_box)
            #box_num = np.ones([N_box,])
            rela_num = np.ones([N_rela,])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                sub_id = int(sub_id)
                #box_num[sub_id] += 1.0
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                obj_id = int(obj_id)
                #box_num[obj_id] += 1.0
                rela_id = i
                rela_num[rela_id] += 1.0
                sub_feat_use = att_feats[img_id * seq_per_img, sub_id, :]
                obj_feat_use = att_feats[img_id * seq_per_img, obj_id, :]
                rela_feat_use = rela_feats[img_id * seq_per_img, rela_id, :]

                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, sub_id, :] += \
                    self.rela_sbj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, obj_id, :] += \
                    self.rela_obj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, rela_id, :] += \
                    self.rela_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))

            N_obj_attr = torch.sum(attr_masks[img_id * seq_per_img, :, 0])
            N_obj_attr = int(N_obj_attr)
            for i in range(N_obj_attr):
                attr_obj_id = int(attr_matrix[img_id * seq_per_img, i, 0])
                obj_feat_use = att_feats[img_id * seq_per_img, int(attr_obj_id), :]
                N_attr_each = torch.sum(attr_masks[img_id * seq_per_img, i, :])
                for j in range(N_attr_each-1):
                    attr_index = attr_matrix[img_id * seq_per_img, i, j+1].cuda().long()
                    attr_feat_use = self.embed(attr_index)
                    attr_feat_use = self.embed2vis(attr_feat_use)
                    attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] += \
                        self.rela_attr_fc( torch.cat((attr_feat_use, obj_feat_use)) )
                attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] = \
                    attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :]/(float(N_attr_each)-1)


            # for i in range(N_box):
            #     att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] = \
            #         att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i]/box_num[i]
            for i in range(N_rela):
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] = \
                    rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :]/rela_num[i]

        rela_data['att_feats'] = att_feats_new
        rela_data['rela_feats'] = rela_feats_new
        rela_data['attr_feats'] = attr_feats_new
        return rela_data  

    def merge_rela_att(self, rela_data):
        """
        merge attention features (roi features) and relationship features together
        :param att_feats: [N_att, N_att_max, rnn_size]
        :param att_masks: [N_att, N_att_max]
        :param rela_feats: [N_att, N_rela_max, rnn_size]
        :param rela_masks: [N_att, N_rela_max]
        :return: att_feats_new: [N_att, N_att_new_max, rnn_size]
                 att_masks_new: [N_att, N_att_new_max]
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_feats = rela_data['attr_feats']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        N_att_new_max = -1
        for img_id in range(int(N_img)):
            N_att_new_max = \
            max(N_att_new_max,torch.sum(rela_masks[img_id * seq_per_img, :]) +
                torch.sum(att_masks[img_id * seq_per_img, :]) + torch.sum(attr_masks[img_id * seq_per_img,:,0]))
        att_masks_new = torch.zeros([N_att, int(N_att_new_max)]).cuda()
        att_feats_new = torch.zeros([N_att, int(N_att_new_max), self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = int(torch.sum(rela_masks[img_id * seq_per_img, :]))
            N_box = int(torch.sum(att_masks[img_id * seq_per_img, :]))
            N_attr = int(torch.sum(attr_masks[img_id * seq_per_img,:,0]))
            att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :] = \
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :]
            if N_rela > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela, :] = \
                    rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_rela, :]
            if N_attr > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela: N_box + N_rela + N_attr, :] = \
                    attr_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_attr, :]
            att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box] = 1
            if N_rela > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela] = 1
            if N_attr > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela:N_box + N_rela + N_attr] = 1

        rela_data['att_feats_new'] = att_feats_new
        rela_data['att_masks_new'] = att_masks_new
        return rela_data  

    #def forward(self, fc_feats, att_feats, att_masks, rela_data, use_rela, imgsfeats, imgsfc7, wordclass):
    def forward(self, imgsfeats, imgsfc7, wordclass, rela_data):

        # caption word -> (100, 512, 15)
        attn_buffer = None
        wordemb = self.emb_0(wordclass)              ## Embedding(9221, 512)  -> (100, 15, 512)
        wordemb = self.emb_1(wordemb)                ## Linear(512, 512)      -> (100, 15, 512)
        x = wordemb.transpose(2, 1)                  ## (100, 15, 512)        -> (100, 512, 15)
        batchsize, wordembdim, maxtokens = x.size()

        y = F.relu(self.imgproj(imgsfc7))
        y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
        x = torch.cat([x, y], 1)


        for i, conv in enumerate(self.convs):
          
          if(i == 0):
            x = x.transpose(2, 1)
            residual = self.resproj(x)
            residual = residual.transpose(2, 1)
            x = x.transpose(2, 1)
          else:
            residual = x

          x = F.dropout(x, p=self.dropout, training=self.training)

          x = conv(x)
          x = x[:,:,:-self.pad]

          x = F.glu(x, dim=1)

          if (self.is_attention):
            attn = self.attention[i]
            x = x.transpose(2, 1)
            x, attn_buffer = attn(x, wordemb, imgsfeats)
            x = x.transpose(2, 1)

          x = (x+residual)*math.sqrt(.5)

        x = x.transpose(2, 1)
      
        x = self.classifier_0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier_1(x)

        x = x.transpose(2, 1)

        return x, attn_buffer


class convcap_D(nn.Module):

    # def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=0.1):
    def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=0.1):
        super(convcap_D, self).__init__()
        self.nimgfeats = 2048
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout

        self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)  # Linear(9221, 512)
        self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

        self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
        self.resproj = Linear(nfeats * 2, self.nfeats, dropout=dropout)

        n_in = 2 * self.nfeats
        n_out = self.nfeats
        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = 5
        self.pad = self.kernel_size - 1
        for i in range(self.n_layers):
            self.convs.append(Conv1d(n_in, 2 * n_out, self.kernel_size, self.pad, dropout))
            if (self.is_attention):
                self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out

        self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
        self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

        '''
        self.input_encoding_size = 512
        self.rnn_size = 512
        self.drop_prob_lm = 0.5
        self.fc_feat_size = 2048
        self.att_feat_size = 2048
        self.att_hid_size = 512
        self.seq_per_img = 5
        self.index_eval = 0
        self.use_rela = False
        self.vocab_size = 14964
        self.use_bn = False

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(self.drop_prob_lm))
        self.embed2vis = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(self.drop_prob_lm))

        self.rela_sbj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(self.drop_prob_lm))
        self.rela_obj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(self.drop_prob_lm))
        self.rela_rela_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
        self.rela_attr_fc = nn.Sequential(nn.Linear(self.rnn_size * 2, self.rnn_size),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
        self.rela_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)  ## nn.Linear(512, 512)
        '''

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        return fc_feats, att_feats

    def prepare_rela_feats(self, rela_data):
        """
        Change relationship index (one-hot) to relationship features, or change relationship
        probability to relationship features.
        :param rela_matrix:
        :param rela_masks:
        :return: rela_features, [N_img*5, N_rela_max, rnn_size]
        """
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        rela_feats_size = rela_matrix.size()
        N_att = rela_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        rela_feats = torch.zeros([rela_feats_size[0], rela_feats_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            if N_rela > 0:
                rela_index = rela_matrix[img_id * seq_per_img, :N_rela, 2].cuda().long()
                rela_feats_temp = self.embed(rela_index)
                rela_feats_temp = self.embed2vis(rela_feats_temp)
                rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_rela, :] = rela_feats_temp
        rela_data['rela_feats'] = rela_feats
        return rela_data

    def rela_graph_gfc(self, rela_data):
        """
        :param att_feats: roi features of each bounding box, [N_img*5, N_att_max, rnn_size]
        :param rela_feats: the embeddings of relationship, [N_img*5, N_rela_max, rnn_size]
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param att_masks: attention masks, [N_img*5, N_att_max].
                            For each row, the sum of that row is the total number
                            of roi poolings.
        :param attr_matrix: attribute matrix,[N_img*5, N_attr_max, N_attr_each_max]
                            N_img is the batch size, N_attr_max is the maximum number
                            of attributes of one mini-batch, N_attr_each_max is the
                            maximum number of attributes of each objects in that mini-batch
        :param attr_masks: attribute masks, [N_img*5, N_attr_max, N_attr_each_max]
                            the sum of attr_masks[img_id*5,:,0] is the number of objects
                            which own attributes, the sum of attr_masks[img_id*5, obj_id, :]
                            is the number of attribute that object has
        :return: att_feats_new: new roi features
                 rela_feats_new: new relationship embeddings
                 attr_feats_new: new attribute features
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_matrix = rela_data['rela_matrix']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_matrix = rela_data['attr_matrix']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        attr_masks_size = attr_masks.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        att_feats_new = att_feats.clone()
        rela_feats_new = rela_feats.clone()
        attr_feats_new = torch.zeros([attr_masks_size[0], attr_masks_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            # N_box = torch.sum(att_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            # N_box = int(N_box)
            # box_num = np.ones([N_box,])
            rela_num = np.ones([N_rela, ])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                sub_id = int(sub_id)
                # box_num[sub_id] += 1.0
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                obj_id = int(obj_id)
                # box_num[obj_id] += 1.0
                rela_id = i
                rela_num[rela_id] += 1.0
                sub_feat_use = att_feats[img_id * seq_per_img, sub_id, :]
                obj_feat_use = att_feats[img_id * seq_per_img, obj_id, :]
                rela_feat_use = rela_feats[img_id * seq_per_img, rela_id, :]

                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, sub_id, :] += \
                    self.rela_sbj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, obj_id, :] += \
                    self.rela_obj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, rela_id, :] += \
                    self.rela_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))

            N_obj_attr = torch.sum(attr_masks[img_id * seq_per_img, :, 0])
            N_obj_attr = int(N_obj_attr)
            for i in range(N_obj_attr):
                attr_obj_id = int(attr_matrix[img_id * seq_per_img, i, 0])
                obj_feat_use = att_feats[img_id * seq_per_img, int(attr_obj_id), :]
                N_attr_each = torch.sum(attr_masks[img_id * seq_per_img, i, :])
                for j in range(N_attr_each - 1):
                    attr_index = attr_matrix[img_id * seq_per_img, i, j + 1].cuda().long()
                    attr_feat_use = self.embed(attr_index)
                    attr_feat_use = self.embed2vis(attr_feat_use)
                    attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] += \
                        self.rela_attr_fc(torch.cat((attr_feat_use, obj_feat_use)))
                attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] = \
                    attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] / (float(N_attr_each) - 1)

            # for i in range(N_box):
            #     att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] = \
            #         att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i]/box_num[i]
            for i in range(N_rela):
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] = \
                    rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] / rela_num[i]

        rela_data['att_feats'] = att_feats_new
        rela_data['rela_feats'] = rela_feats_new
        rela_data['attr_feats'] = attr_feats_new
        return rela_data

    def merge_rela_att(self, rela_data):
        """
        merge attention features (roi features) and relationship features together
        :param att_feats: [N_att, N_att_max, rnn_size]
        :param att_masks: [N_att, N_att_max]
        :param rela_feats: [N_att, N_rela_max, rnn_size]
        :param rela_masks: [N_att, N_rela_max]
        :return: att_feats_new: [N_att, N_att_new_max, rnn_size]
                 att_masks_new: [N_att, N_att_new_max]
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_feats = rela_data['attr_feats']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        N_att_new_max = -1
        for img_id in range(int(N_img)):
            N_att_new_max = \
                max(N_att_new_max, torch.sum(rela_masks[img_id * seq_per_img, :]) +
                    torch.sum(att_masks[img_id * seq_per_img, :]) + torch.sum(attr_masks[img_id * seq_per_img, :, 0]))
        att_masks_new = torch.zeros([N_att, int(N_att_new_max)]).cuda()
        att_feats_new = torch.zeros([N_att, int(N_att_new_max), self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = int(torch.sum(rela_masks[img_id * seq_per_img, :]))
            N_box = int(torch.sum(att_masks[img_id * seq_per_img, :]))
            N_attr = int(torch.sum(attr_masks[img_id * seq_per_img, :, 0]))
            att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :] = \
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :]
            if N_rela > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela, :] = \
                    rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_rela, :]
            if N_attr > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela: N_box + N_rela + N_attr,
                :] = \
                    attr_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_attr, :]
            att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box] = 1
            if N_rela > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela] = 1
            if N_attr > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img,
                N_box + N_rela:N_box + N_rela + N_attr] = 1

        rela_data['att_feats_new'] = att_feats_new
        rela_data['att_masks_new'] = att_masks_new
        return rela_data

        # def forward(self, fc_feats, att_feats, att_masks, rela_data, use_rela, imgsfeats, imgsfc7, wordclass):

    def forward(self, imgsfeats, imgsfc7, wordclass, rela_data):

        # caption word -> (100, 512, 15)
        attn_buffer = None
        wordemb = self.emb_0(wordclass)  ## Embedding(9221, 512)  -> (100, 15, 512)
        wordemb = self.emb_1(wordemb)  ## Linear(512, 512)      -> (100, 15, 512)
        x = wordemb.transpose(2, 1)  ## (100, 15, 512)        -> (100, 512, 15)
        batchsize, wordembdim, maxtokens = x.size()

        y = F.relu(self.imgproj(imgsfc7))
        y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
        x = torch.cat([x, y], 1)

        ####
        '''
        att_masks = None
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
        if use_rela == 1:
            rela_data['att_feats'] = att_feats
            rela_data['att_masks'] = att_masks
            rela_data = self.prepare_rela_feats(rela_data)
            rela_data = self.rela_graph_gfc(rela_data)
            rela_data = self.merge_rela_att(rela_data)
        else:
            rela_data['att_feats_new'] = fc_feats
            rela_data['att_masks_new'] = att_masks

        att_feats_rela = rela_data['att_feats_new']

        p_att_feats_rela = att_feats_rela.unsqueeze(2).expand((batchsize, self.nfeats, maxtokens))
        #x = torch.cat([x, p_att_feats_rela], 1)
        ####
        '''

        for i, conv in enumerate(self.convs):

            if (i == 0):
                x = x.transpose(2, 1)
                residual = self.resproj(x)
                residual = residual.transpose(2, 1)
                x = x.transpose(2, 1)
            else:
                residual = x

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x)
            x = x[:, :, :-self.pad]

            x = F.glu(x, dim=1)

            if (self.is_attention):
                attn = self.attention[i]
                x = x.transpose(2, 1)
                x, attn_buffer = attn(x, wordemb, imgsfeats)
                x = x.transpose(2, 1)

            x = (x + residual) * math.sqrt(.5)

        x = x.transpose(2, 1)

        x = self.classifier_0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier_1(x)

        x = x.transpose(2, 1)

        return x, attn_buffer
