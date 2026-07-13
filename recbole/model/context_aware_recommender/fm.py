# -*- coding: utf-8 -*-
# @Time   : 2020/7/8 10:09
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : fm.py

# UPDATE:
# @Time   : 2020/8/13,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
FM
################################################
Reference:
    Steffen Rendle et al. "Factorization Machines." in ICDM 2010.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine
from recbole.model.init import xavier_normal_initialization_linearonly
from recbole.model.general_recommender.lightgcn import Mask

import time
import numpy as np

class FM(ContextRecommender):
    """Factorization Machine considers the second-order interaction with features to predict the final score."""

    def __init__(self, config, dataset):

        super(FM, self).__init__(config, dataset)

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.latent_dim = config["embedding_size"]
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        
        if self.use_llm_embed:
            self.openai_dim = 1024
            self.middle_dim = (self.openai_dim + self.latent_dim) // 2
            # self.middle_dim2 = (self.middle_dim + self.latent_dim) // 2
            self.pretrained_user_emb = dataset.useremb_feat['user_emb']
            self.pretrained_item_emb = dataset.itememb_feat['item_emb']
            zeroline = torch.zeros((1, self.pretrained_user_emb.shape[1]))
            self.pretrained_user_emb = torch.cat((zeroline, self.pretrained_user_emb))
            self.pretrained_item_emb = torch.cat((zeroline, self.pretrained_item_emb))
            self.merge_embed = config['merge_embed']
            self.USER_ID = config["USER_ID_FIELD"]
            self.ITEM_ID = config["ITEM_ID_FIELD"]
            # self.sample_num = config['sample_num']
            self.beta = config['beta']
            self.tau = config['tau']
            self.n_users = dataset.num(self.USER_ID)
            self.n_items = dataset.num(self.ITEM_ID)
            
            # self.user_ori_embed = nn.Embedding.from_pretrained(self.pretrained_user_emb)
            # self.item_ori_embed = nn.Embedding.from_pretrained(self.pretrained_item_emb)
            # freeze(self.user_ori_embed)
            # freeze(self.item_ori_embed)
            # print(self.user_ori_embed.weight)
            # print(self.item_ori_embed.weight)
            if self.merge_embed != 'cl-gen':
                self.user_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.middle_dim, self.middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.latent_dim)
                )
                self.item_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    # torch.nn.Linear(self.middle_dim, self.middle_dim2),
                    # torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.latent_dim)
                )
            else:
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.openai_dim)
                )
                self.mask_ratio = config['mask_ratio']
                self.masker = Mask(self.mask_ratio, self.latent_dim, config)
            if self.merge_embed == 'none':
                self.apply(xavier_normal_initialization_linearonly)
            else:
                self.apply(self._init_weights)
            self.user_ori_embed = nn.Embedding.from_pretrained(self.pretrained_user_emb)
            self.item_ori_embed = nn.Embedding.from_pretrained(self.pretrained_item_emb)
            freeze(self.user_ori_embed)
            freeze(self.item_ori_embed)
            # print(self.user_ori_embed.weight)
            # print(self.item_ori_embed.weight)
            # time.sleep(1000)
        else:
            # parameters initialization, originally without indent
            self.apply(self._init_weights)
        # edit end
        
        # # parameters initialization, originally without indent
        #     self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
    
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

    def forward(self, interaction):
        # edit
        cl_loss = None
        if self.use_llm_embed and self.merge_embed != 'cl-gen':
            user_embeddings = self.user_linear(self.user_ori_embed.weight[interaction['user_id']])
            item_embeddings = self.item_linear(self.item_ori_embed.weight[interaction['item_id']])
            # user_embeddings = torch.reshape(user_embeddings, (user_embeddings.shape[0], 1, user_embeddings.shape[1]))
            # item_embeddings = torch.reshape(item_embeddings, (item_embeddings.shape[0], 1, item_embeddings.shape[1]))
            if self.merge_embed == 'none':
                fm_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
                fm_all_embeddings[:, 2, :] = user_embeddings
                fm_all_embeddings[:, 3, :] = item_embeddings
                # fm_all_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
            else:
                model_all_embeddings = self.concat_embed_input_fields(interaction)
                # print(model_all_embeddings.shape)
                # time.sleep(1000)
                # llm_all_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
                if self.merge_embed == 'add':
                    # fm_all_embeddings = model_all_embeddings + llm_all_embeddings # merge_func
                    model_all_embeddings[:, 2, :] += user_embeddings
                    model_all_embeddings[:, 3, :] += item_embeddings
                    fm_all_embeddings = model_all_embeddings
                elif self.merge_embed[0:2] == 'cl':
                    # fm_all_embeddings = model_all_embeddings + llm_all_embeddings
                    # model_all_embeddings[:, 2, :] += user_embeddings
                    # model_all_embeddings[:, 3, :] += item_embeddings
                    fm_all_embeddings = model_all_embeddings
                    try:
                        cur_model_user_embed = model_all_embeddings[:, 2, :][self.batch_user]
                        cur_model_item_embed = model_all_embeddings[:, 3, :][self.batch_item]
                        cur_llm_user_embed = user_embeddings[self.batch_user]
                        cur_llm_item_embed = item_embeddings[self.batch_item]
                        user_cl_loss = self.get_cl_loss(cur_model_user_embed, cur_llm_user_embed)
                        item_cl_loss = self.get_cl_loss(cur_model_item_embed, cur_llm_item_embed)
                        # print("cl_loss: {}, {}".format(user_cl_loss, item_cl_loss))
                        cl_loss = user_cl_loss + item_cl_loss
                    except:
                        cl_loss = None
        elif self.use_llm_embed and self.merge_embed == 'cl-gen':
            self.llm_user_embedding = self.user_ori_embed.weight[interaction['user_id']]
            self.llm_item_embedding = self.item_ori_embed.weight[interaction['item_id']]
            model_all_embeddings = self.concat_embed_input_fields(interaction)
            self.model_user_embedding = torch.nn.Parameter(model_all_embeddings[:, 2, :])
            self.model_item_embedding = torch.nn.Parameter(model_all_embeddings[:, 3, :])
            temp_embeds = torch.concat([self.model_user_embedding, self.model_item_embedding], axis=0)
            masked_embeds, self.seeds = self.masker(temp_embeds)
            self.model_user_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[:self.n_users]).weight
            self.model_item_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[self.n_users:]).weight
            model_all_embeddings[:, 2, :] = self.model_user_embedding[interaction['user_id']]
            model_all_embeddings[:, 3, :] = self.model_item_embedding[interaction['item_id']]
            fm_all_embeddings = model_all_embeddings
            
            model_all_embed = torch.concat([self.model_user_embedding, self.model_item_embedding], axis=0)
            llm_all_embed = torch.concat([self.llm_user_embedding, self.llm_item_embedding], axis=0)
            enc_embeds = model_all_embed[self.seeds]
            prf_embeds = llm_all_embed[self.seeds]
            enc_embeds = self.mlp(enc_embeds)
            cl_loss = self.get_ssl_con_loss(enc_embeds, prf_embeds, self.tau)
        else:
            fm_all_embeddings = self.concat_embed_input_fields(
                interaction
            )  # [batch_size, num_field, embed_dim]
            # print(fm_all_embeddings.shape)
            # time.sleep(1000)
        # edit end
        # print(fm_all_embeddings)
        # print(fm_all_embeddings.shape)
        # time.sleep(1000)
        # print(self.first_order_linear(interaction).shape)
        # print(self.fm(fm_all_embeddings).shape)
        # time.sleep(1000)
        
        # print(interaction['user_id'].shape)
        # print(self.first_order_linear(interaction).shape)
        y = self.first_order_linear(interaction) + self.fm(fm_all_embeddings)
        
        return y.squeeze(-1), cl_loss

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        # edit
        if self.use_llm_embed:
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
            
            user_ids, item_ids = list(set(user.tolist())), list(set(item.tolist()))
            # print(len(user_ids), len(item_ids))
            # time.sleep(1)
            user2, item2 = user.tolist(), item.tolist()
            self.batch_user, self.batch_item = [], []
            for i in range(len(user_ids)):
                self.batch_user.append(user2.index(user_ids[i]))
            for i in range(len(item_ids)):
                self.batch_item.append(item2.index(item_ids[i]))
            # self.batch_user = torch.LongTensor(list(set(user.tolist()))).cuda()
            # self.batch_item = torch.LongTensor(list(set(item.tolist()))).cuda()
            self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)
        # edit end

        output, cl_loss = self.forward(interaction)
        # print(self.loss(output, label), cl_loss)
        if cl_loss is None:
            return self.loss(output, label)
        else:
            return self.loss(output, label) + cl_loss * self.beta

    def predict(self, interaction):
        output, _ = self.forward(interaction)
        return self.sigmoid(output)