# -*- coding: utf-8 -*-
# @Time   : 2023/04/12
# @Author : Wanli Yang
# @Email  : 2013774@mail.nankai.edu.cn

r"""
LightGCL
################################################
Reference:
    Xuheng Cai et al. "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation" in ICLR 2023.

Reference code:
    https://github.com/HKUDS/LightGCL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization, xavier_uniform_initialization_linearonly
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from recbole.model.general_recommender.lightgcn import Mask

class LightGCL(GeneralRecommender):
    r"""LightGCL is a GCN-based recommender model.

    LightGCL guides graph augmentation by singular value decomposition (SVD) to not only
    distill the useful information of user-item interactions but also inject the global
    collaborative context into the representation alignment of contrastive learning.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        # load parameters info
        self.embed_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.temp = config["temp"]
        self.lambda_1 = config["lambda1"]
        self.lambda_2 = config["lambda2"]
        self.q = config["q"]
        self.act = nn.LeakyReLU(0.5)
        self.reg_loss = EmbLoss()
        self.LABEL = config["LABEL_FIELD"]

        # get the normalized adjust matrix
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())

        # perform svd reconstruction
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self.q)
        self.u_mul_s = svd_u @ (torch.diag(s))
        self.v_mul_s = svd_v @ (torch.diag(s))
        del s
        self.ut = svd_u.T
        self.vt = svd_v.T
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 4096
            self.middle_dim = (self.openai_dim + self.embed_dim) // 2
            # self.middle_dim2 = (self.middle_dim + self.embed_dim) // 2
            self.pretrained_user_emb = dataset.useremb_feat['user_emb']
            self.pretrained_item_emb = dataset.itememb_feat['item_emb']
            zeroline = torch.zeros((1, self.pretrained_user_emb.shape[1]))
            self.pretrained_user_emb = torch.cat((zeroline, self.pretrained_user_emb))
            self.pretrained_item_emb = torch.cat((zeroline, self.pretrained_item_emb))
            self.merge_embed = config['merge_embed']
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
                    torch.nn.Linear(self.middle_dim, self.embed_dim)
                )
                self.item_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.middle_dim, self.middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.embed_dim)
                )
            else:
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.openai_dim)
                )
                self.mask_ratio = config['mask_ratio']
                self.masker = Mask(self.mask_ratio, self.embed_dim, config)
            if self.merge_embed != 'none':
                self.model_user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_dim)))
                self.model_item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_dim)))
        if (not self.use_llm_embed) or self.merge_embed == 'none':
            self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_dim)))
            self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_dim)))

        # self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_dim)))
        # self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_dim)))
        self.E_u_list = [None] * (self.n_layers + 1)
        self.E_i_list = [None] * (self.n_layers + 1)
        try:
            self.E_u_list[0] = self.E_u_0
            self.E_i_list[0] = self.E_i_0
        except:
            self.E_u_list[0] = self.model_user_embedding
            self.E_i_list[0] = self.model_item_embedding
        self.Z_u_list = [None] * (self.n_layers + 1)
        self.Z_i_list = [None] * (self.n_layers + 1)
        self.G_u_list = [None] * (self.n_layers + 1)
        self.G_i_list = [None] * (self.n_layers + 1)
        
        try:
            self.G_u_list[0] = self.E_u_0
            self.G_i_list[0] = self.E_i_0
        except:
            self.G_u_list[0] = self.model_user_embedding
            self.G_i_list[0] = self.model_item_embedding

        self.E_u = None
        self.E_i = None
        self.restore_user_e = None
        self.restore_item_e = None

        if self.use_llm_embed:
            self.apply(xavier_uniform_initialization_linearonly)
        else:
            self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def create_adjust_matrix(self):
        r"""Get the normalized interaction matrix of users and items.

        Returns:
            coo_matrix of the normalized interaction matrix.
        """
        ratings = np.ones_like(self._user, dtype=np.float32)
        matrix = sp.csr_matrix(
            (ratings, (self._user, self._item)),
            shape=(self.n_users, self.n_items),
        ).tocoo()
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)
        for i in range(len(matrix.data)):
            matrix.data[i] = matrix.data[i] / pow(rowD[matrix.row[i]] * colD[matrix.col[i]], 0.5)
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix):
        r"""Convert coo_matrix to tensor.

        Args:
            matrix (scipy.coo_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        indices = torch.from_numpy(
            np.vstack((matrix.row, matrix.col)).astype(np.int64))
        values = torch.from_numpy(matrix.data)
        shape = torch.Size(matrix.shape)
        x = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(self.device)
        return x

    def sparse_dropout(self, matrix, dropout):
        if dropout == 0.0:
            return matrix
        indices = matrix.indices()
        values = F.dropout(matrix.values(), p=dropout)
        size = matrix.size()
        return torch.sparse.FloatTensor(indices, values, size)
    
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

    def forward(self):
        # edit
        cl_loss = None
        if self.use_llm_embed:
            if self.merge_embed == 'none':
                self.E_u_list[0] = self.user_linear(self.user_ori_embed.weight)
                self.E_i_list[0] = self.item_linear(self.item_ori_embed.weight)
            elif self.merge_embed != 'cl-gen':
                self.llm_user_embedding = self.user_linear(self.user_ori_embed.weight)
                self.llm_item_embedding = self.item_linear(self.item_ori_embed.weight)
                if self.merge_embed == 'add':
                    self.E_u_list[0] = self.model_user_embedding + self.llm_user_embedding
                    self.E_i_list[0] = self.model_item_embedding + self.llm_item_embedding
                elif self.merge_embed[0:2] == 'cl':
                    cur_model_user_embed = self.model_user_embedding[self.batch_user]
                    cur_model_item_embed = self.model_item_embedding[self.batch_item]
                    cur_llm_user_embed = self.llm_user_embedding[self.batch_user]
                    cur_llm_item_embed = self.llm_item_embedding[self.batch_item]
                    user_cl_loss = self.get_cl_loss(cur_model_user_embed, cur_llm_user_embed)
                    item_cl_loss = self.get_cl_loss(cur_model_item_embed, cur_llm_item_embed)
                    cl_loss = user_cl_loss + item_cl_loss
                    self.E_u_list[0] = self.model_user_embedding
                    self.E_i_list[0] = self.model_item_embedding
            else:
                self.llm_user_embedding = self.user_ori_embed.weight
                self.llm_item_embedding = self.item_ori_embed.weight
                temp_embeds = torch.concat([self.model_user_embedding, self.model_item_embedding], axis=0)
                masked_embeds, self.seeds = self.masker(temp_embeds)
                self.model_user_embedding = torch.nn.Parameter(torch.nn.Embedding.from_pretrained(masked_embeds[:self.n_users]).weight)
                self.model_item_embedding = torch.nn.Parameter(torch.nn.Embedding.from_pretrained(masked_embeds[self.n_users:]).weight)
                self.E_u_list[0] = self.model_user_embedding
                self.E_i_list[0] = self.model_item_embedding
        
        for layer in range(1, self.n_layers + 1):
            # GNN propagation
            self.Z_u_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout),
                                              self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                                              self.E_u_list[layer - 1])
            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        # aggregate across layer
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i, cl_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        # neg_item_list = interaction[self.NEG_ITEM_ID]
        label = interaction[self.LABEL]
        
        user_ids, item_ids = list(set(user_list.tolist())), list(set(pos_item_list.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)
        
        E_u_norm, E_i_norm, cl_loss = self.forward()
        if self.use_llm_embed and self.merge_embed == 'cl-gen':
            model_all_embed = torch.concat([E_u_norm, E_i_norm], axis=0)
            llm_all_embed = torch.concat([self.llm_user_embedding, self.llm_item_embedding], axis=0)
            enc_embeds = model_all_embed[self.seeds]
            prf_embeds = llm_all_embed[self.seeds]
            enc_embeds = self.mlp(enc_embeds)
            cl_loss = self.get_ssl_con_loss(enc_embeds, prf_embeds, self.tau)
        
        bpr_loss = self.calc_bpr_loss(E_u_norm,
                                      E_i_norm,
                                      user_list,
                                      pos_item_list,
                                      # neg_item_list
                                      label
                                      )
        ssl_loss = self.calc_ssl_loss(E_u_norm, E_i_norm, user_list, pos_item_list)
        if cl_loss is not None:
            total_loss = bpr_loss + ssl_loss + cl_loss * self.beta
        else:
            total_loss = bpr_loss + ssl_loss
        return total_loss

    def calc_bpr_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list,
                      # neg_item_list
                      label
                      ):
        r"""Calculate the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = E_u_norm[user_list]
        pi_e = E_i_norm[pos_item_list]
        # ni_e = E_i_norm[neg_item_list]
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        # neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        # loss1 = -(pos_scores - neg_scores).sigmoid().log().mean()
        loss1 = torch.nn.MSELoss()(pos_scores, label)

        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        # loss_reg = 0
        # reg_param_num = 0
        # for param in self.parameters():
        #     if param.requires_grad:
        #         loss_reg += param.norm(2).square()
        #         reg_param_num += param.numel()
        # if reg_param_num > 0:
        #     loss_reg = loss_reg / reg_param_num
        loss_reg *= self.lambda_2
        return loss1 + loss_reg

    def calc_ssl_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users in the original graph after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items in the original graph after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        # calculate G_u_norm&G_i_norm
        for layer in range(1, self.n_layers + 1):
            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

        # aggregate across layer
        G_u_norm = sum(self.G_u_list)
        G_i_norm = sum(self.G_i_list)

        neg_score = torch.log(torch.exp(G_u_norm[user_list] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[pos_item_list] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[user_list] * E_u_norm[user_list]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[pos_item_list] * E_i_norm[pos_item_list]).sum(1) / self.temp, -5.0, 5.0)).mean()
        ssl_loss = -pos_score + neg_score
        return self.lambda_1 * ssl_loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
