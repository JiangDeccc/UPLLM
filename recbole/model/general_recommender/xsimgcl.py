# -*- coding: utf-8 -*-
r"""
XSimGCL
################################################
Reference:
    Junliang Yu, Xin Xia, Tong Chen, Lizhen Cui, Nguyen Quoc Viet Hung, Hongzhi Yin. "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation" in TKDE 2023.

Reference code:
    https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py
"""


import torch
import torch.nn.functional as F

# from recbole.recbole_gnn.recbole_gnn.model.general_recommender import LightGCN
from recbole.model.general_recommender.lightgcn import LightGCN
from torch.nn.init import xavier_normal_, xavier_uniform_
from recbole.model.init import xavier_uniform_initialization, xavier_uniform_initialization_linearonly
import time
from recbole.model.general_recommender.lightgcn import Mask
from tqdm import tqdm

class XSimGCL(LightGCN):
    def __init__(self, config, dataloader):
        # dataset = dataloader.dataset
        dataset = dataloader
        super(XSimGCL, self).__init__(config, dataset)
        
        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.layer_cl = config['layer_cl']
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        
        # self.user_list, self.positem_list = self.preload_unique(dataloader)
        
        if self.use_llm_embed:
            self.openai_dim = 4096
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
    
    def preload_unique(self, dataset):
        user_list, positem_list = [], []
        with tqdm(total=len(list(enumerate(dataset)))) as pbar:
            for batch_idx, interaction in enumerate(dataset):
                user = torch.unique(interaction[self.USER_ID]).to(self.device)
                pos_item = torch.unique(interaction[self.ITEM_ID]).to(self.device)
                user_list.append(user)
                positem_list.append(pos_item)
                pbar.update()
        return user_list, positem_list
    
    # edit
    def get_cl_loss(self, model_embed, llm_embed):
        cl_loss = 0.0
        llm_embed2 = llm_embed.T
        # dots = torch.mm(model_embed, llm_embed2) # multiply
        # time1 = time.time()
        dots = model_embed @ llm_embed.T
        # time2 = time.time()
        
        if self.merge_embed == 'cl-cos':
            # mod1 = torch.sqrt(model_embed.square().sum(-1, keepdim=True))
            # mod2 = torch.sqrt(llm_embed.square().sum(-1, keepdim=True))
            mod1 = model_embed.norm(dim=-1, keepdim=True)
            mod2 = llm_embed.norm(dim=-1, keepdim=True)
            # mod1 = mod1.unsqueeze(1)
            # mod2 = mod2.unsqueeze(0)
            # moddot = mod1.T @ mod2
            moddot = mod1 @ mod2.T
            # time3 = time.time()
            # moddot = torch.mm(mod1, mod2)
            dots /= moddot # cosine
        
        dots /= self.tau
        numerators = torch.exp(dots.diag())
        denominators = torch.exp(dots).sum(-1)
        cl_loss = torch.log(numerators / denominators).sum() / (-dots.shape[0])
        # time4 = time.time()
        # print("Calculate CL loss: two matmul: {} {}, cal loss: {}".format(time2 - time1, time3 - time2, time4 - time3))
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
                # time1 = time.time()
                self.llm_user_embedding = self.user_linear(self.user_ori_embed.weight)
                # time2 = time.time()
                self.llm_item_embedding = self.item_linear(self.item_ori_embed.weight)
                # time3 = time.time()
                # print("transfer_embed: {} {}".format(time2 - time1, time3 - time2))
            else:
                self.llm_user_embedding = self.user_ori_embed.weight
                self.llm_item_embedding = self.item_ori_embed.weight
                temp_embeds = torch.concat([self.model_user_embedding.weight, self.model_item_embedding.weight], axis=0)
                masked_embeds, self.seeds = self.masker(temp_embeds)
                self.model_user_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[:self.n_users])
                self.model_item_embedding = torch.nn.Embedding.from_pretrained(masked_embeds[self.n_users:])
        
        # all_embs = self.get_ego_embeddings()
        all_embs, cl_loss = self.get_ego_embeddings2()
        all_embs_cl = all_embs
        embeddings_list = []
        
        # time1 = time.time()
        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embs)
            if layer_idx == self.layer_cl - 1:
                all_embs_cl = all_embs
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.n_users, self.n_items])
        # time2 = time.time()
        # print("Main GCN: {}".format(time2 - time1))
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl, cl_loss
        return user_all_embeddings, item_all_embeddings, cl_loss

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()

    def calculate_loss(self, interaction, batch_idx=None):
        # clear the storage variable when training
        # start = time.time()
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID]
        label = interaction[self.LABEL]
        
        user_ids, item_ids = list(set(user.tolist())), list(set(pos_item.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)

        # time1 = time.time()
        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl, cl_loss = self.forward(perturbed=True)
        # time2 = time.time()
        # print("Forward: {}".format(time2 - time1))
        
        if self.use_llm_embed and self.merge_embed == 'cl-gen':
            model_all_embed = torch.concat([user_all_embeddings, item_all_embeddings], axis=0)
            llm_all_embed = torch.concat([self.llm_user_embedding, self.llm_item_embedding], axis=0)
            enc_embeds = model_all_embed[self.seeds]
            prf_embeds = llm_all_embed[self.seeds]
            enc_embeds = self.mlp(enc_embeds)
            cl_loss = self.get_ssl_con_loss(enc_embeds, prf_embeds, self.tau)
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        # neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        # time3 = time.time()
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # time4 = time.time()
        # print("Pos scores: {}".format(time4 - time3))
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        # mf_loss = self.mf_loss(pos_scores, neg_scores)
        mf_loss = self.mf_loss(pos_scores, label)
        # time5 = time.time()
        # print("mf_loss: {}".format(time5 - time4))

        if not self.use_llm_embed:
            # calculate regularization Loss
            u_ego_embeddings = self.user_embedding(user)
            pos_ego_embeddings = self.item_embedding(pos_item)
            # neg_ego_embeddings = self.item_embedding(neg_item)
        else:
            if self.merge_embed != 'none' and self.merge_embed != 'add':
                u_ego_embeddings = self.user_embedding[user]
                pos_ego_embeddings = self.item_embedding[pos_item]
            else:
                u_ego_embeddings = self.user_embedding_[user]
                pos_ego_embeddings = self.item_embedding_[pos_item]
        
        # time6 = time.time()
        # print("Indexing: {}".format(time6 - time5))
        
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            # neg_ego_embeddings,
            require_pow=self.require_pow
        )
        
        # time7 = time.time()
        # print("reg_loss: {}".format(time7 - time6))
        
        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])
        # user = self.user_list[batch_idx]
        # pos_item = self.positem_list[batch_idx]
        
        # time8 = time.time()
        # print("Unique: {}".format(time8 - time7))
        
        # calculate CL Loss
        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[user], user_all_embeddings_cl[user])
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[pos_item], item_all_embeddings_cl[pos_item])
        # end = time.time()
        # print("final_cl_loss: {}".format(end - time8))
        
        # print("Total time: {}".format(end - start))
        # time.sleep(1)

        # if cl_loss is not None:
        #     return mf_loss, self.reg_weight * reg_loss, self.cl_rate * (user_cl_loss + item_cl_loss), cl_loss
        # else:
        #     return mf_loss, self.reg_weight * reg_loss, self.cl_rate * (user_cl_loss + item_cl_loss)
        
        if cl_loss is not None:
            return mf_loss + self.reg_weight * reg_loss + self.cl_rate * (user_cl_loss + item_cl_loss) + cl_loss * self.beta
        else:
            return mf_loss + self.reg_weight * reg_loss + self.cl_rate * (user_cl_loss + item_cl_loss)
