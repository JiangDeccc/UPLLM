# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import time
from torch.nn.init import xavier_normal_, xavier_uniform_
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization, xavier_uniform_initialization_linearonly
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class Mask(torch.nn.Module):
    def __init__(self, mask_ratio, embedding_size, config):
        super(Mask, self).__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = torch.nn.Parameter(torch.zeros(1, embedding_size))
        self.config = config
    
    def forward(self, embeds):
        seeds = np.random.choice(embeds.shape[0], size=max(int(embeds.shape[0] * self.mask_ratio), 1), replace=False)
        seeds = torch.LongTensor(seeds).to(self.config['device'])
        mask = torch.ones(embeds.shape[0]).to(self.config['device'])
        mask[seeds] = 0
        mask = mask.view(-1, 1)
        masked_embeds = embeds * mask + self.mask_token * (1. - mask)
        return masked_embeds, seeds 


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        
        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 4096
            # self.middle_dim = self.latent_dim
            self.middle_dim = (self.openai_dim + self.latent_dim) // 2
            # self.middle_dim2 = (self.middle_dim + self.latent_dim) // 2
            self.pretrained_user_emb = dataset.useremb_feat['user_emb']
            self.pretrained_item_emb = dataset.itememb_feat['item_emb']
            zeroline = torch.zeros((1, self.pretrained_user_emb.shape[1]))
            self.pretrained_user_emb = torch.cat((zeroline, self.pretrained_user_emb))
            self.pretrained_item_emb = torch.cat((zeroline, self.pretrained_item_emb))
            # print(type(self.pretrained_user_emb))
            # self.pretrained_user_emb = dataset.user_feat['user_emb']
            # self.pretrained_item_emb = dataset.item_feat['item_emb']
            self.merge_embed = config['merge_embed']
            # self.sample_num = config['sample_num']
            self.beta = config['beta']
            self.tau = config['tau']
            # print(pretrained_user_emb)
            # print(pretrained_user_emb.shape, type(pretrained_user_emb))
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
        else:
            # define layers and loss
            self.user_embedding = torch.nn.Embedding(
                num_embeddings=self.n_users, embedding_dim=self.latent_dim
            )
            self.item_embedding = torch.nn.Embedding(
                num_embeddings=self.n_items, embedding_dim=self.latent_dim
            )
        
        # print("2")
        # print(self.user_ori_embed.weight)
        # edit end
        
        # self.mf_loss = BPRLoss()
        self.mf_loss = torch.nn.MSELoss()
        self.LABEL = config["LABEL_FIELD"]
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # 为 XSimGCL 提供 PyG 风格接口
        indices = self.norm_adj_matrix.coalesce().indices()
        self.edge_index = indices
        self.edge_weight = self.norm_adj_matrix.coalesce().values()

        class _SparseConv(torch.nn.Module):
            def forward(self, x, edge_index, edge_weight):
                n = x.shape[0]
                adj = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n), device=x.device)
                return torch.sparse.mm(adj, x)

        self.gcn_conv = _SparseConv()

        # parameters initialization
        # print("before user_embed")
        # print(self.user_ori_embed.weight)
        # print("before item_embed")
        # print(self.item_ori_embed.weight)
        if self.use_llm_embed:
            self.apply(xavier_uniform_initialization_linearonly)
            if self.merge_embed != 'none':
                xavier_uniform_(self.model_user_embedding.weight.data)
                xavier_uniform_(self.model_item_embedding.weight.data)
        else:
            self.apply(xavier_uniform_initialization)
        
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        # print("after user_embed")
        # print(self.user_ori_embed.weight)
        # print("after item_embed")
        # print(self.item_ori_embed.weight)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        n = self.n_users + self.n_items
        rows = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        cols = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(rows), dtype=np.float32)
        
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        A = A.todok()  # 转成 dok 只为复用后续逻辑，也可直接用 csr

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    # def get_norm_adj_mat(self):
    #     r"""Get the normalized interaction matrix of users and items.

    #     Construct the square matrix from the training data and normalize it
    #     using the laplace matrix.

    #     .. math::
    #         A_{hat} = D^{-0.5} \times A \times D^{-0.5}

    #     Returns:
    #         Sparse tensor of the normalized interaction matrix.
    #     """
    #     # build adj matrix
    #     A = sp.dok_matrix(
    #         (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
    #     )
    #     inter_M = self.interaction_matrix
    #     inter_M_t = self.interaction_matrix.transpose()
    #     data_dict = dict(
    #         zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
    #     )
    #     data_dict.update(
    #         dict(
    #             zip(
    #                 zip(inter_M_t.row + self.n_users, inter_M_t.col),
    #                 [1] * inter_M_t.nnz,
    #             )
    #         )
    #     )
    #     A._update(data_dict)
    #     # norm adj matrix
    #     sumArr = (A > 0).sum(axis=1)
    #     # add epsilon to avoid divide by zero Warning
    #     diag = np.array(sumArr.flatten())[0] + 1e-7
    #     diag = np.power(diag, -0.5)
    #     D = sp.diags(diag)
    #     L = D * A * D
    #     # covert norm_adj matrix to tensor
    #     L = sp.coo_matrix(L)
    #     row = L.row
    #     col = L.col
    #     i = torch.LongTensor(np.array([row, col]))
    #     data = torch.FloatTensor(L.data)
    #     SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    #     return SparseL
    
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
        
        # for i in range(dots.shape[0]):
        #     numerator = numerators[i]
        #     sample_ids = np.arange(dots.shape[0])
        #     sample_ids = np.delete(sample_ids, i)
        #     sample_ids = np.random.choice(a=sample_ids, size=self.sample_num, replace=False)
        #     denominator = numerator + torch.sum(torch.exp(dots[i][sample_ids]))
        #     # denominator = torch.sum(torch.exp(dots[i][sample_ids]))
        #     cl_loss += torch.log(numerator / denominator)
        # cl_loss /= (-dots.shape[0])
        return cl_loss
    
    # edit
    def get_ssl_con_loss(self, x, y, temp=1.0):
        x = F.normalize(x)
        y = F.normalize(y)
        mole = torch.exp(torch.sum(x * y, dim=1) / temp)
        deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)
        return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()
    
    # def get_cl_loss(self):
    #     # start = time.time()
    #     model_user_embeddings = self.model_user_embedding.weight[self.batch_user]
    #     model_item_embeddings = self.model_item_embedding.weight[self.batch_item]
    #     llm_user_embeddings = self.llm_user_embedding[self.batch_user]
    #     llm_item_embeddings = self.llm_item_embedding[self.batch_item]
    #     # print(model_user_embeddings.device, model_item_embeddings.device, llm_user_embeddings.device, llm_item_embeddings.device)
    #     # print(model_user_embeddings.shape, model_item_embeddings.shape, llm_user_embeddings.shape, llm_item_embeddings.shape)
    #     # time.sleep(1000)
    #     user_loss, item_loss = 0.0, 0.0
    #     user_dots = model_user_embeddings * llm_user_embeddings
    #     item_dots = model_item_embeddings * llm_item_embeddings
    #     user_numerators = torch.exp(user_dots.sum(axis=1))
    #     item_numerators = torch.exp(item_dots.sum(axis=1))
    #     llm_user_embeddings = llm_user_embeddings.T
    #     llm_item_embeddings = llm_item_embeddings.T
    #     user_dots = torch.mm(model_user_embeddings, llm_user_embeddings)
    #     item_dots = torch.mm(model_item_embeddings, llm_item_embeddings)
    #     for i in range(model_user_embeddings.shape[0]):
    #         numerator = user_numerators[i]
    #         sample_ids = np.arange(model_user_embeddings.shape[0])
    #         sample_ids = np.delete(sample_ids, i)
    #         sample_ids = np.random.choice(a=sample_ids, size=self.sample_num, replace=False)
    #         denominator = torch.sum(torch.exp(user_dots[i][sample_ids]))
    #         user_loss += numerator / denominator
        
    #     for i in range(model_item_embeddings.shape[0]):
    #         numerator = item_numerators[i]
    #         sample_ids = np.arange(model_item_embeddings.shape[0])
    #         sample_ids = np.delete(sample_ids, i)
    #         sample_ids = np.random.choice(a=sample_ids, size=self.sample_num, replace=False)
    #         denominator = torch.sum(torch.exp(item_dots[i][sample_ids]))
    #         item_loss += numerator / denominator
    #         # print(i, numerator.item(), denominator.item(), item_loss.item())
        
    #     # print("----")
    #     # print(user_loss, item_loss)
        
    #     cl_loss = user_loss + item_loss
    #     # end = time.time()
    #     # print("cost: {}".format(end - start))
    #     # time.sleep(1000)
    #     return cl_loss
    # edit end

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        # edit
        cl_loss = None
        if not self.use_llm_embed:
            user_embeddings = self.user_embedding.weight
            item_embeddings = self.item_embedding.weight
        else:
            if self.merge_embed == 'none':
                user_embeddings = self.user_embedding
                item_embeddings = self.item_embedding
            else:
                model_user_embeddings = self.model_user_embedding.weight
                model_item_embeddings = self.model_item_embedding.weight
                llm_user_embeddings = self.llm_user_embedding
                llm_item_embeddings = self.llm_item_embedding
                if self.merge_embed == 'add':
                    self.user_embedding = model_user_embeddings + llm_user_embeddings # merge_func
                    self.item_embedding = model_item_embeddings + llm_item_embeddings
                    user_embeddings = self.user_embedding
                    item_embeddings = self.item_embedding
                elif self.merge_embed[0:2] == 'cl' and self.merge_embed != 'cl-gen':
                    cur_model_user_embed = model_user_embeddings[self.batch_user]
                    # print(model_item_embeddings.shape)
                    # print(self.batch_item)
                    cur_model_item_embed = model_item_embeddings[self.batch_item]
                    cur_llm_user_embed = llm_user_embeddings[self.batch_user]
                    cur_llm_item_embed = llm_item_embeddings[self.batch_item]
                    user_cl_loss = self.get_cl_loss(cur_model_user_embed, cur_llm_user_embed)
                    item_cl_loss = self.get_cl_loss(cur_model_item_embed, cur_llm_item_embed)
                    cl_loss = user_cl_loss + item_cl_loss
                    # self.user_embedding = model_user_embeddings + llm_user_embeddings
                    # self.item_embedding = model_item_embeddings + llm_item_embeddings
                    self.user_embedding = model_user_embeddings
                    self.item_embedding = model_item_embeddings
                    user_embeddings = self.user_embedding
                    item_embeddings = self.item_embedding
                else:
                    self.user_embedding = model_user_embeddings
                    self.item_embedding = model_item_embeddings
                    user_embeddings = self.user_embedding
                    item_embeddings = self.item_embedding
                # print(model_user_embeddings.shape)
                # print(llm_user_embeddings.shape)
                # print(model_item_embeddings.shape)
                # print(llm_item_embeddings.shape)
                # time.sleep(2)
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, cl_loss
    # edit end

    def forward(self):
        # edit
        # print("user_embed")
        # print(self.user_ori_embed.weight)
        # print("item_embed")
        # print(self.item_ori_embed.weight)
        # time.sleep(1)
        if self.use_llm_embed:
            if self.merge_embed == 'none':
                self.user_embedding = self.user_linear(self.user_ori_embed.weight)
                self.item_embedding = self.item_linear(self.item_ori_embed.weight)
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
            # print(self.user_embedding.shape, self.item_embedding.shape)
            # time.sleep(1000)
        # edit end
        all_embeddings, cl_loss = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings, cl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID] # edit, using mse loss, not bpr loss.
        label = interaction[self.LABEL]
        
        # edit
        # self.batch_user = torch.LongTensor(list(set(user.tolist()))).cuda()
        # self.batch_item = torch.LongTensor(list(set(pos_item.tolist()))).cuda()
        user_ids, item_ids = list(set(user.tolist())), list(set(pos_item.tolist()))
        # print(len(user_ids), len(item_ids))
        # time.sleep(1)
        # print(type(user))
        # user2, item2 = user.tolist(), pos_item.tolist()
        # self.batch_user, self.batch_item = [], []
        # for i in range(len(user_ids)):
        #     self.batch_user.append(user2.index(user_ids[i]))
        # for i in range(len(item_ids)):
        #     self.batch_item.append(item2.index(item_ids[i]))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)
        # print(user_ids, len(user_ids))
        # print(self.batch_user, self.batch_user.shape)
        # print(self.batch_user.shape)
        # print(self.batch_item.shape)
        # time.sleep(1000)
        # edit end

        user_all_embeddings, item_all_embeddings, cl_loss = self.forward()
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
        
        # print(user.shape, pos_item.shape, label.shape)
        # print(u_embeddings.shape, pos_embeddings.shape)
        # time.sleep(2)

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # print(pos_scores.shape)
        # pos_scores = torch.nn.functional.sigmoid(pos_scores)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        # mf_loss = self.mf_loss(pos_scores, neg_scores)
        mf_loss = self.mf_loss(pos_scores, label)
        # print(mf_loss)
        # print(user)
        # print(pos_item)
        # print(label)
        # time.sleep(1000)

        # calculate BPR Loss
        # edit
        if not self.use_llm_embed:
            u_ego_embeddings = self.user_embedding(user)
            pos_ego_embeddings = self.item_embedding(pos_item)
            # neg_ego_embeddings = self.item_embedding(neg_item)
        else:
            u_ego_embeddings = self.user_embedding[user]
            pos_ego_embeddings = self.item_embedding[pos_item]

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            # neg_ego_embeddings,
            require_pow=self.require_pow,
        )
        # edit end

        loss = mf_loss + self.reg_weight * reg_loss
        # edit
        if cl_loss is not None:
            loss += cl_loss * self.beta
        # edit end

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward() # edit

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward() # edit
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
