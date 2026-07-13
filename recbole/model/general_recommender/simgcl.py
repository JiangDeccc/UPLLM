# -*- coding: utf-8 -*-
r"""
SimGCL
################################################
Reference:
    Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, Quoc Viet Hung Nguyen. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." in SIGIR 2022.
"""


import torch
import torch.nn.functional as F

# from recbole.model.general_recommender import LightGCN
from recbole.recbole_gnn.recbole_gnn.model.general_recommender import LightGCN
from torch.nn.init import xavier_normal_, xavier_uniform_
from recbole.model.init import xavier_uniform_initialization, xavier_uniform_initialization_linearonly
from recbole.model.general_recommender.lightgcn import Mask

class SimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 1536
            self.middle_dim = (self.openai_dim + self.latent_dim) // 2
            # self.middle_dim2 = (self.middle_dim + self.latent_dim) // 2
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
            if self.merge_embed != 'none':
                self.model_user_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_users, embedding_dim=self.latent_dim
                )
                self.model_item_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_items, embedding_dim=self.latent_dim
                )
            self.apply(xavier_uniform_initialization_linearonly)
            if self.merge_embed != 'none':
                xavier_uniform_(self.model_user_embedding.weight.data)
                xavier_uniform_(self.model_item_embedding.weight.data)
        else:
            self.apply(xavier_uniform_initialization)
    
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
    
    # edit
    def get_ego_embeddings2(self):
        cl_loss = None
        if not self.use_llm_embed:
            user_embeddings = self.user_embedding.weight
            item_embeddings = self.item_embedding.weight
        else:
            if self.merge_embed == 'none':
                user_embeddings = self.user_embedding_
                item_embeddings = self.item_embedding_
            else:
                model_user_embeddings = self.model_user_embedding.weight
                model_item_embeddings = self.model_item_embedding.weight
                llm_user_embeddings = self.llm_user_embedding
                llm_item_embeddings = self.llm_item_embedding
                if self.merge_embed == 'add':
                    self.user_embedding_ = model_user_embeddings + llm_user_embeddings # merge_func
                    self.item_embedding_ = model_item_embeddings + llm_item_embeddings
                    user_embeddings = self.user_embedding_
                    item_embeddings = self.item_embedding_
                elif self.merge_embed[0:2] == 'cl' and self.merge_embed != 'cl-gen':
                    cur_model_user_embed = model_user_embeddings[self.batch_user]
                    cur_model_item_embed = model_item_embeddings[self.batch_item]
                    cur_llm_user_embed = llm_user_embeddings[self.batch_user]
                    cur_llm_item_embed = llm_item_embeddings[self.batch_item]
                    user_cl_loss = self.get_cl_loss(cur_model_user_embed, cur_llm_user_embed)
                    item_cl_loss = self.get_cl_loss(cur_model_item_embed, cur_llm_item_embed)
                    cl_loss = user_cl_loss + item_cl_loss
                    self.user_embedding = model_user_embeddings
                    self.item_embedding = model_item_embeddings
                    user_embeddings = self.user_embedding
                    item_embeddings = self.item_embedding
                else:
                    self.user_embedding = model_user_embeddings
                    self.item_embedding = model_item_embeddings
                    user_embeddings = self.user_embedding
                    item_embeddings = self.item_embedding
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, cl_loss

    def forward(self, perturbed=False):
        if self.use_llm_embed:
            if self.merge_embed == 'none':
                self.user_embedding_ = self.user_linear(self.user_ori_embed.weight)
                self.item_embedding_ = self.item_linear(self.item_ori_embed.weight)
            elif self.merge_embed != 'cl-gen':
                self.llm_user_embedding = self.user_linear(self.user_ori_embed.weight)
                self.llm_item_embedding = self.item_linear(self.item_ori_embed.weight)
            else:
                self.llm_user_embedding = self.user_ori_embed.weight
                self.llm_item_embedding = self.item_ori_embed.weight
                temp_embeds = torch.concat([self.model_user_embedding.weight, self.model_item_embedding.weight], axis=0)
                masked_embeds, self.seeds = self.masker(temp_embeds)
                self.model_user_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[:self.n_users])
                self.model_item_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[self.n_users:])
        
        # all_embs = self.get_ego_embeddings()
        cl_loss = None
        if not perturbed:
            all_embs, cl_loss = self.get_ego_embeddings2()
        else:
            all_embs = self.get_ego_embeddings()
        
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, cl_loss

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def calculate_loss(self, interaction):
        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])
        
        user_ids, item_ids = list(set(user.tolist())), list(set(pos_item.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)
        
        loss = super().calculate_loss(interaction)
        
        perturbed_user_embs_1, perturbed_item_embs_1, _ = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2, _ = self.forward(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])
        
        total_loss = loss + self.cl_rate * (user_cl_loss + item_cl_loss)

        # return loss + self.cl_rate * (user_cl_loss + item_cl_loss)
        return total_loss