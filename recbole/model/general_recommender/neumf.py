# -*- coding: utf-8 -*-
# @Time   : 2020/6/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/22,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
import numpy as np
import time
from recbole.model.general_recommender.lightgcn import Mask

class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.mf_embedding_size = config["mf_embedding_size"]
        self.mlp_embedding_size = config["mlp_embedding_size"]
        self.mlp_hidden_size = eval(config["mlp_hidden_size"])
        self.dropout_prob = config["dropout_prob"]
        self.mf_train = config["mf_train"]
        self.mlp_train = config["mlp_train"]
        self.use_pretrain = config["use_pretrain"]
        self.mf_pretrain_path = config["mf_pretrain_path"]
        self.mlp_pretrain_path = config["mlp_pretrain_path"]

        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 4096
            self.mf_middle_dim = (self.openai_dim + self.mf_embedding_size) // 2
            self.mlp_middle_dim = (self.openai_dim + self.mlp_embedding_size) // 2
            # self.mf_middle_dim2 = (self.mf_middle_dim + self.mf_embedding_size) // 2
            # self.mlp_middle_dim2 = (self.mlp_middle_dim + self.mlp_embedding_size) // 2
            # self.mf_middle_dim = self.mf_embedding_size
            # self.mlp_middle_dim = self.mlp_embedding_size
            # print(dataset.useremb_feat['uid'])
            self.pretrained_user_emb = dataset.useremb_feat['user_emb']
            self.pretrained_item_emb = dataset.itememb_feat['item_emb']
            zeroline = torch.zeros((1, self.pretrained_user_emb.shape[1]))
            self.pretrained_user_emb = torch.cat((zeroline, self.pretrained_user_emb))
            self.pretrained_item_emb = torch.cat((zeroline, self.pretrained_item_emb))
            self.merge_embed = config['merge_embed']
            # self.sample_num = config['sample_num']
            self.beta = config['beta']
            self.tau = config['tau']
            self.user_ori_embed = torch.nn.Embedding.from_pretrained(self.pretrained_user_emb)
            self.item_ori_embed = torch.nn.Embedding.from_pretrained(self.pretrained_item_emb)
            freeze(self.user_ori_embed)
            freeze(self.item_ori_embed)
            if self.merge_embed != 'cl-gen':
                self.user_mf_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.mf_middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.mf_middle_dim, self.mf_middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mf_middle_dim, self.mf_embedding_size)
                )
                self.item_mf_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.mf_middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.mf_middle_dim, self.mf_middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mf_middle_dim, self.mf_embedding_size)
                )
                self.user_mlp_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.mlp_middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.mlp_middle_dim, self.mlp_middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mlp_middle_dim, self.mlp_embedding_size)
                )
                self.item_mlp_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.mlp_middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.mlp_middle_dim, self.mlp_middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mlp_middle_dim, self.mlp_embedding_size)
                )
            else:
                self.mf_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.mf_embedding_size, self.mf_middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mf_middle_dim, self.openai_dim)
                )
                self.mlp_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.mlp_embedding_size, self.mlp_middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.mlp_middle_dim, self.openai_dim)
                )
                self.mask_ratio = config['mask_ratio']
                self.mf_masker = Mask(self.mask_ratio, self.mf_embedding_size, config)
                self.mlp_masker = Mask(self.mask_ratio, self.mlp_embedding_size, config)
            if self.merge_embed != 'none':
                self.model_user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
                self.model_item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
                self.model_user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
                self.model_item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        else:
            # define layers and loss
            self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
            self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
            self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
            self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        # edit end
        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        )
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)
    
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
    # edit end

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path)
        mlp = torch.load(self.mlp_pretrain_path)
        self.user_mf_embedding.weight.data.copy_(mf.user_mf_embedding.weight)
        self.item_mf_embedding.weight.data.copy_(mf.item_mf_embedding.weight)
        self.user_mlp_embedding.weight.data.copy_(mlp.user_mlp_embedding.weight)
        self.item_mlp_embedding.weight.data.copy_(mlp.item_mlp_embedding.weight)

        for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

        predict_weight = torch.cat(
            [mf.predict_layer.weight, mlp.predict_layer.weight], dim=1
        )
        predict_bias = mf.predict_layer.bias + mlp.predict_layer.bias

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if self.use_llm_embed:
            if isinstance(module, nn.Linear):
                normal_(module.weight.data, 0, 0.01)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
            if self.merge_embed != 'none':
                normal_(self.model_user_mf_embedding.weight.data, mean=0.0, std=0.01)
                normal_(self.model_user_mlp_embedding.weight.data, mean=0.0, std=0.01)
                normal_(self.model_item_mf_embedding.weight.data, mean=0.0, std=0.01)
                normal_(self.model_item_mlp_embedding.weight.data, mean=0.0, std=0.01)
        else:
            if isinstance(module, nn.Embedding):
                normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        # edit
        cl_loss = None
        if not self.use_llm_embed:
            user_mf_e = self.user_mf_embedding(user)
            item_mf_e = self.item_mf_embedding(item)
            user_mlp_e = self.user_mlp_embedding(user)
            item_mlp_e = self.item_mlp_embedding(item)
        else:
            if self.merge_embed == 'none':
                user_mf_e = self.user_mf_linear(self.user_ori_embed.weight)[user]
                item_mf_e = self.item_mf_linear(self.item_ori_embed.weight)[item]
                user_mlp_e = self.user_mlp_linear(self.user_ori_embed.weight)[user]
                item_mlp_e = self.item_mlp_linear(self.item_ori_embed.weight)[item]
                # user_mf_e = self.user_mf_linear(self.user_ori_embed(user))
                # item_mf_e = self.item_mf_linear(self.item_ori_embed(item))
                # user_mlp_e = self.user_mlp_linear(self.user_ori_embed(user))
                # item_mlp_e = self.item_mlp_linear(self.item_ori_embed(item))
            elif self.merge_embed != 'cl-gen':
                self.llm_user_mf_embedding = self.user_mf_linear(self.user_ori_embed.weight)
                self.llm_item_mf_embedding = self.item_mf_linear(self.item_ori_embed.weight)
                self.llm_user_mlp_embedding = self.user_mlp_linear(self.user_ori_embed.weight)
                self.llm_item_mlp_embedding = self.item_mlp_linear(self.item_ori_embed.weight)
                # self.llm_user_mf_embedding = self.user_mf_linear(self.user_ori_embed(user))
                # self.llm_item_mf_embedding = self.item_mf_linear(self.item_ori_embed(item))
                # self.llm_user_mlp_embedding = self.user_mlp_linear(self.user_ori_embed(user))
                # self.llm_item_mlp_embedding = self.item_mlp_linear(self.item_ori_embed(item))
                if self.merge_embed == 'add':
                    user_mf_e = (self.model_user_mf_embedding.weight + self.llm_user_mf_embedding)[user]
                    item_mf_e = (self.model_item_mf_embedding.weight + self.llm_item_mf_embedding)[item]
                    user_mlp_e = (self.model_user_mlp_embedding.weight + self.llm_user_mlp_embedding)[user]
                    item_mlp_e = (self.model_item_mlp_embedding.weight + self.llm_item_mlp_embedding)[item]
                    # user_mf_e = self.model_user_mf_embedding(user) + self.llm_user_mf_embedding
                    # item_mf_e = self.model_item_mf_embedding(item) + self.llm_item_mf_embedding
                    # user_mlp_e = self.model_user_mlp_embedding(user) + self.llm_user_mlp_embedding
                    # item_mlp_e = self.model_item_mlp_embedding(item) + self.llm_item_mlp_embedding
                elif self.merge_embed[0:2] == 'cl':
                    # user_mf_e = (self.model_user_mf_embedding.weight + self.llm_user_mf_embedding)[user]
                    # item_mf_e = (self.model_item_mf_embedding.weight + self.llm_item_mf_embedding)[item]
                    # user_mlp_e = (self.model_user_mlp_embedding.weight + self.llm_user_mlp_embedding)[user]
                    # item_mlp_e = (self.model_item_mlp_embedding.weight + self.llm_item_mlp_embedding)[item]
                    user_mf_e = (self.model_user_mf_embedding.weight)[user]
                    item_mf_e = (self.model_item_mf_embedding.weight)[item]
                    user_mlp_e = (self.model_user_mlp_embedding.weight)[user]
                    item_mlp_e = (self.model_item_mlp_embedding.weight)[item]
                    
                    # user_mf_e = self.model_user_mf_embedding(user) + self.llm_user_mf_embedding
                    # item_mf_e = self.model_item_mf_embedding(item) + self.llm_item_mf_embedding
                    # user_mlp_e = self.model_user_mlp_embedding(user) + self.llm_user_mlp_embedding
                    # item_mlp_e = self.model_item_mlp_embedding(item) + self.llm_item_mlp_embedding
                    cur_model_user_mf_embed = self.model_user_mf_embedding.weight[self.batch_user]
                    cur_model_item_mf_embed = self.model_item_mf_embedding.weight[self.batch_item]
                    cur_model_user_mlp_embed = self.model_user_mlp_embedding.weight[self.batch_user]
                    cur_model_item_mlp_embed = self.model_item_mlp_embedding.weight[self.batch_item]
                    cur_llm_user_mf_embed = self.llm_user_mf_embedding[self.batch_user]
                    cur_llm_item_mf_embed = self.llm_item_mf_embedding[self.batch_item]
                    cur_llm_user_mlp_embed = self.llm_user_mlp_embedding[self.batch_user]
                    cur_llm_item_mlp_embed = self.llm_item_mlp_embedding[self.batch_item]
                    user_mf_cl_loss = self.get_cl_loss(cur_model_user_mf_embed, cur_llm_user_mf_embed)
                    item_mf_cl_loss = self.get_cl_loss(cur_model_item_mf_embed, cur_llm_item_mf_embed)
                    user_mlp_cl_loss = self.get_cl_loss(cur_model_user_mlp_embed, cur_llm_user_mlp_embed)
                    item_mlp_cl_loss = self.get_cl_loss(cur_model_item_mlp_embed, cur_llm_item_mlp_embed)
                    cl_loss = user_mf_cl_loss + item_mf_cl_loss + user_mlp_cl_loss + item_mlp_cl_loss
            else:
                pass

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )
        return output.squeeze(-1), cl_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        user_ids, item_ids = list(set(user.tolist())), list(set(item.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)

        output, cl_loss = self.forward(user, item)
        if cl_loss is None:
            return self.loss(output, label)
        else:
            return self.loss(output, label) + cl_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        output, _ = self.forward(user, item)
        predict = self.sigmoid(output)
        return predict

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
