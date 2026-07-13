# -*- coding: utf-8 -*-
# @Time   : 2022/3/25 13:38
# @Author : HaoJun Qin
# @Email  : 18697951462@163.com

r"""
SimpleX
################################################

Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

Reference code:
    https://github.com/xue-pai/TwoToweRS
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_

from recbole.model.init import xavier_normal_initialization, xavier_normal_initialization_linearonly
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import time
import numpy as np
from recbole.model.general_recommender.lightgcn import Mask

class SimpleX(GeneralRecommender):
    r"""SimpleX is a simple, unified collaborative filtering model.

    SimpleX presents a simple and easy-to-understand model. Its advantage lies
    in its loss function, which uses a larger number of negative samples and
    sets a threshold to filter out less informative samples, it also uses
    relative weights to control the balance of positive-sample loss
    and negative-sample loss.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # Get user history interacted items
        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        self.reg_weight = config["reg_weight"]
        self.aggregator = config["aggregator"]
        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention or self_attention"
            )
        self.history_len = torch.max(self.history_item_len, dim=0)
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 4096
            # self.middle_dim = self.embedding_size
            self.middle_dim = (self.openai_dim + self.embedding_size) // 2
            # self.middle_dim2 = (self.middle_dim + self.embedding_size) // 2
            self.pretrained_user_emb = dataset.useremb_feat['user_emb']
            self.pretrained_item_emb = dataset.itememb_feat['item_emb']
            zeroline = torch.zeros((1, self.pretrained_user_emb.shape[1]))
            self.pretrained_user_emb = torch.cat((zeroline, self.pretrained_user_emb))
            self.pretrained_item_emb = torch.cat((zeroline, self.pretrained_item_emb))
            # self.pretrained_user_emb = dataset.user_feat['user_emb']
            # self.pretrained_item_emb = dataset.item_feat['item_emb']
            self.merge_embed = config['merge_embed']
            # self.sample_num = config['sample_num']
            self.beta = config['beta']
            self.tau = config['tau']
            
            self.user_ori_embed = torch.nn.Embedding.from_pretrained(self.pretrained_user_emb)
            self.item_ori_embed = torch.nn.Embedding.from_pretrained(self.pretrained_item_emb)
            freeze(self.user_ori_embed)
            freeze(self.item_ori_embed)
            if self.merge_embed != 'cl-gen':
                self.user_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.middle_dim, self.middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.embedding_size)
                )
                self.item_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.middle_dim, self.middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.embedding_size)
                )
            else:
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.openai_dim)
                )
                self.mask_ratio = config['mask_ratio']
                self.masker = Mask(self.mask_ratio, self.embedding_size, config)
            if self.merge_embed != 'none':
                self.model_user_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_users, embedding_dim=self.embedding_size
                )
                self.model_item_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_items, embedding_dim=self.embedding_size, padding_idx=0
                )
                self.model_item_embedding.weight.data[0, :] = 0
        else:
            # user embedding matrix
            self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
            # item embedding matrix
            self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
            # get the mask
            self.item_emb.weight.data[0, :] = 0
        # edit end
        
        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_size, 1, bias=False)
        # dropout
        self.dropout = nn.Dropout(0.1)
        self.require_pow = config["require_pow"]
        # l2 regularization loss
        self.reg_loss = EmbLoss()

        # edit
        if self.use_llm_embed:
            self.apply(xavier_normal_initialization_linearonly)
            if self.merge_embed != 'none':
                xavier_normal_(self.model_user_embedding.weight.data)
                xavier_normal_(self.model_item_embedding.weight.data)
        else:
            # parameters initialization
            self.apply(xavier_normal_initialization)
        # edit end
        
        # get the mask
        # self.item_emb.weight.data[0, :] = 0

    def get_UI_aggregation(self, user_e, history_item_e, history_len):
        r"""Get the combined vector of user and historically interacted items

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            history_item_e (torch.Tensor): History item's feature vector,
                shape: [user_num, max_history_len, embedding_size]
            history_len (torch.Tensor): User's history length, shape: [user_num]

        Returns:
            torch.Tensor: Combined vector of user and item sequences, shape: [user_num, embedding_size]
        """
        if self.aggregator == "mean":
            pos_item_sum = history_item_e.sum(dim=1)
            # [user_num, embedding_size]
            out = pos_item_sum / (history_len + 1.0e-10).unsqueeze(1)
        elif self.aggregator in ["user_attention", "self_attention"]:
            # [user_num, max_history_len, embedding_size]
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                # [user_num, max_history_len]
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
            elif self.aggregator == "self_attention":
                # [user_num, max_history_len]
                attention = self.W_q(key).squeeze(2)
            e_attention = torch.exp(attention)
            mask = (history_item_e.sum(dim=-1) != 0).int()
            e_attention = e_attention * mask
            # [user_num, max_history_len]
            attention_weight = e_attention / (
                e_attention.sum(dim=1, keepdim=True) + 1.0e-10
            )
            # [user_num, embedding_size]
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
        # Combined vector of user and item sequences
        out = self.UI_map(out)
        g = self.gamma
        UI_aggregation_e = g * user_e + (1 - g) * out
        return UI_aggregation_e

    def get_cos(self, user_e, item_e):
        r"""Get the cosine similarity between user and item

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            item_e (torch.Tensor): Item's feature vector,
                shape: [user_num, item_num, embedding_size]

        Returns:
            torch.Tensor: Cosine similarity between user and item, shape: [user_num, item_num]
        """
        user_e = F.normalize(user_e, dim=1)
        # [user_num, embedding_size, 1]
        user_e = user_e.unsqueeze(2)
        item_e = F.normalize(item_e, dim=2)
        UI_cos = torch.matmul(item_e, user_e)
        return UI_cos.squeeze(2)
    
    # edit
    def get_cl_loss(self, model_embed, llm_embed):
        cl_loss = 0.0
        llm_embed2 = llm_embed.T
        # dots = torch.mm(model_embed, llm_embed2) # multiply
        
        dots = model_embed @ llm_embed.T
        if self.merge_embed == 'cl-cos':
            # mod1 = torch.sqrt(model_embed.square().sum(-1, keepdim=True))
            # mod2 = torch.sqrt(llm_embed.square().sum(-1, keepdim=True))
            mod1 = model_embed.norm(dim=-1, keepdim=True)
            mod2 = llm_embed.norm(dim=-1, keepdim=True)
            # mod1 = mod1.unsqueeze(1)
            # mod2 = mod2.unsqueeze(0)
            # moddot = mod1.T @ mod2
            moddot = mod1 @ mod2.T
            # moddot = torch.mm(mod1, mod2)
            dots /= moddot # cosine
        
        dots /= self.tau
        numerators = torch.exp(dots.diag())
        denominators = torch.exp(dots).sum(-1)
        cl_loss = torch.log(numerators / denominators).sum() / (-dots.shape[0])
        return cl_loss
    
    # edit
    def get_ssl_con_loss(self, x, y, temp=1.0):
        x = F.normalize(x)
        y = F.normalize(y)
        mole = torch.exp(torch.sum(x * y, dim=1) / temp)
        deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)
        return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()   

    def forward(self, user, pos_item, history_item, history_len, neg_item_seq):
        r"""Get the loss

        Args:
            user (torch.Tensor): User's id, shape: [user_num]
            pos_item (torch.Tensor): Positive item's id, shape: [user_num]
            history_item (torch.Tensor): Id of historty item, shape: [user_num, max_history_len]
            history_len (torch.Tensor): History item's length, shape: [user_num]
            neg_item_seq (torch.Tensor): Negative item seq's id, shape: [user_num, neg_seq_len]

        Returns:
            torch.Tensor: Loss, shape: []
        """
        # edit
        cl_loss = None
        if self.use_llm_embed:
            if self.merge_embed == 'none':
                self.user_emb = self.user_linear(self.user_ori_embed.weight)
                self.item_emb = self.item_linear(self.item_ori_embed.weight)
            elif self.merge_embed != 'cl-gen':
                self.llm_user_embedding = self.user_linear(self.user_ori_embed.weight)
                self.llm_item_embedding = self.item_linear(self.item_ori_embed.weight)
                model_user_embeddings = self.model_user_embedding.weight
                model_item_embeddings = self.model_item_embedding.weight
                llm_user_embeddings = self.llm_user_embedding
                llm_item_embeddings = self.llm_item_embedding
                if self.merge_embed == 'add':
                    self.user_emb = model_user_embeddings + llm_user_embeddings # merge_func
                    self.item_emb = model_item_embeddings + llm_item_embeddings
                elif self.merge_embed[0:2] == 'cl' and self.merge_embed != 'cl-gen':
                    cur_model_user_embed = model_user_embeddings[self.batch_user]
                    cur_model_item_embed = model_item_embeddings[self.batch_item]
                    cur_llm_user_embed = llm_user_embeddings[self.batch_user]
                    cur_llm_item_embed = llm_item_embeddings[self.batch_item]
                    user_cl_loss = self.get_cl_loss(cur_model_user_embed, cur_llm_user_embed)
                    item_cl_loss = self.get_cl_loss(cur_model_item_embed, cur_llm_item_embed)
                    cl_loss = user_cl_loss + item_cl_loss
                    # self.user_emb = model_user_embeddings + llm_user_embeddings
                    # self.item_emb = model_item_embeddings + llm_item_embeddings
                    self.user_emb = model_user_embeddings
                    self.item_emb = model_item_embeddings
            else:
                self.llm_user_embedding = self.user_ori_embed.weight
                self.llm_item_embedding = self.item_ori_embed.weight
                temp_embeds = torch.concat([self.model_user_embedding.weight, self.model_item_embedding.weight], axis=0)
                masked_embeds, self.seeds = self.masker(temp_embeds)
                self.model_user_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[:self.n_users])
                self.model_item_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[self.n_users:])
                self.user_emb = self.model_user_embedding.weight
                self.item_emb = self.model_item_embedding.weight
        
        if not self.use_llm_embed:
            # [user_num, embedding_size]
            user_e = self.user_emb(user)
            # [user_num, embedding_size]
            pos_item_e = self.item_emb(pos_item)
            # [user_num, max_history_len, embedding_size]
            history_item_e = self.item_emb(history_item)
            # [nuser_num, neg_seq_len, embedding_size]
            # neg_item_seq_e = self.item_emb(neg_item_seq)
        else:
            user_e = self.user_emb[user]
            pos_item_e = self.item_emb[pos_item]
            history_item_e = self.item_emb[history_item]
        # edit end

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)
        UI_aggregation_e = self.dropout(UI_aggregation_e)

        pos_cos = self.get_cos(UI_aggregation_e, pos_item_e.unsqueeze(1))
        # neg_cos = self.get_cos(UI_aggregation_e, neg_item_seq_e)

        # CCL loss
        pos_loss = torch.relu(1 - pos_cos)
        # neg_loss = torch.relu(neg_cos - self.margin)
        # neg_loss = neg_loss.mean(1, keepdim=True) * self.negative_weight
        # CCL_loss = (pos_loss + neg_loss).mean()
        CCL_loss = (pos_loss).mean()

        # l2 regularization loss
        reg_loss = self.reg_loss(
            user_e,
            pos_item_e,
            history_item_e,
            # neg_item_seq_e,
            require_pow=self.require_pow,
        )
        
        # edit
        if self.use_llm_embed and self.merge_embed == 'cl-gen':
            model_all_embed = torch.concat([self.user_emb, self.item_emb], axis=0)
            llm_all_embed = torch.concat([self.llm_user_embedding, self.llm_item_embedding], axis=0)
            enc_embeds = model_all_embed[self.seeds]
            prf_embeds = llm_all_embed[self.seeds]
            enc_embeds = self.mlp(enc_embeds)
            cl_loss = self.get_ssl_con_loss(enc_embeds, prf_embeds, self.tau)

        loss = CCL_loss + self.reg_weight * reg_loss.sum()
        # edit
        if cl_loss is not None:
            loss += cl_loss * self.beta
        # edit end
        return loss

    def calculate_loss(self, interaction):
        r"""Data processing and call function forward(), return loss

        To use SimpleX, a user must have a historical transaction record,
        a pos item and a sequence of neg items. Based on the RecBole
        framework, the data in the interaction object is ordered, so
        we can get the data quickly.
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID] edit, using mse loss, not bpr loss

        # get the sequence of neg items
        # neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        
        # neg_item_seq = neg_item_seq.T
        # user_number = int(len(user) / self.neg_seq_len)
        # user's id
        # user = user[0:user_number]
        # historical transaction record
        history_item = self.history_item_id[user]
        # positive item's id
        # pos_item = pos_item[0:user_number]
        # history_len
        history_len = self.history_item_len[user]
        
        # print(user) # 一个user的history item就是训练集里面的
        # print(pos_item)
        # print(history_item)
        # print(user.shape, pos_item.shape, history_item.shape)
        # temp_dict, index_dict = {}, {}
        # for i in range(user.shape[0]):
        #     if int(user[i]) not in temp_dict.keys():
        #         temp_dict[int(user[i])] = []
        #         index_dict[int(user[i])] = i
        #     temp_dict[int(user[i])].append(int(pos_item[i]))
        # for i in temp_dict.keys():
        #     temp_dict[i] = sorted(temp_dict[i], key=lambda x:x, reverse=True)
        # print(temp_dict[38])
        # his = history_item[index_dict[38]].tolist()
        # his = sorted(his, key=lambda x:x, reverse=True)
        # print(his)
        # time.sleep(1000)
        
        # edit
        user_ids, item_ids = list(set(user.tolist())), list(set(pos_item.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)

        # loss = self.forward(user, pos_item, history_item, history_len, neg_item_seq)
        loss = self.forward(user, pos_item, history_item, history_len, None)
        # edit end
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]
        test_item = interaction[self.ITEM_ID]

        if not self.use_llm_embed:
            # [user_num, embedding_size]
            user_e = self.user_emb(user)
            # [user_num, embedding_size]
            test_item_e = self.item_emb(test_item)
            # [user_num, max_history_len, embedding_size]
            history_item_e = self.item_emb(history_item)
        else:
            user_e = self.user_emb[user]
            test_item_e = self.item_emb[test_item]
            history_item_e = self.item_emb[history_item]

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_cos = self.get_cos(UI_aggregation_e, test_item_e.unsqueeze(1))
        return UI_cos.squeeze(1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_aggregation_e = F.normalize(UI_aggregation_e, dim=1)
        all_item_emb = self.item_emb.weight
        all_item_emb = F.normalize(all_item_emb, dim=1)
        UI_cos = torch.matmul(UI_aggregation_e, all_item_emb.T)
        return UI_cos
