# -*- coding: utf-8 -*-
# @Time   : 2020/10/2
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

"""
SpectralCF
################################################

Reference:
    Lei Zheng et al. "Spectral collaborative filtering." in RecSys 2018.

Reference code:
    https://github.com/lzheng21/SpectralCF
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.init import xavier_normal_, xavier_uniform_
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization, xavier_uniform_initialization_linearonly
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.general_recommender.lightgcn import Mask

class SpectralCF(GeneralRecommender):
    r"""SpectralCF is a spectral convolution model that directly learns latent factors of users and items 
    from the spectral domain for recommendation.

    The spectral convolution operation with C input channels and F filters is shown as the following:

    .. math::
        \left[\begin{array} {c} X_{new}^{u} \\
        X_{new}^{i} \end{array}\right]=\sigma\left(\left(U U^{\top}+U \Lambda U^{\top}\right)
        \left[\begin{array}{c} X^{u} \\
        X^{i} \end{array}\right] \Theta^{\prime}\right)

    where :math:`X_{new}^{u} \in R^{n_{users} \times F}` and :math:`X_{new}^{i} \in R^{n_{items} \times F}` 
    denote convolution results learned with F filters from the spectral domain for users and items, respectively; 
    :math:`\sigma` denotes the logistic sigmoid function.

    Note:

        Our implementation is a improved version which is different from the original paper.
        For a better stability, we replace :math:`U U^T` with identity matrix :math:`I` and
        replace :math:`U \Lambda U^T` with laplace matrix :math:`L`.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SpectralCF, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.emb_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        # "A_hat = I + L" is equivalent to "A_hat = U U^T + U \Lambda U^T"
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        I = self.get_eye_mat(self.n_items + self.n_users)
        L = self.get_laplacian_matrix()
        A_hat = I + L
        self.A_hat = A_hat.to(self.device)

        # edit
        def freeze(layer):
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        self.use_llm_embed = config['use_llm_embed']
        self.device = config['device']
        if self.use_llm_embed:
            self.openai_dim = 1536
            # self.middle_dim = self.emb_dim
            self.middle_dim = (self.openai_dim + self.emb_dim) // 2
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
                self.user_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.emb_dim)
                )
                self.item_linear = torch.nn.Sequential(
                    torch.nn.Linear(self.openai_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.emb_dim)
                )
            else:
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.emb_dim, self.middle_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(self.middle_dim, self.openai_dim)
                )
                self.mask_ratio = config['mask_ratio']
                self.masker = Mask(self.mask_ratio, self.emb_dim, config)
            if self.merge_embed != 'none':
                self.model_user_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_users, embedding_dim=self.emb_dim
                )
                self.model_item_embedding = torch.nn.Embedding(
                    num_embeddings=self.n_items, embedding_dim=self.emb_dim
                )
        else:
            # define layers and loss
            self.user_embedding = torch.nn.Embedding(
                num_embeddings=self.n_users, embedding_dim=self.emb_dim
            )
            self.item_embedding = torch.nn.Embedding(
                num_embeddings=self.n_items, embedding_dim=self.emb_dim
            )
        # edit end
        self.filters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.normal(
                        mean=0.01, std=0.02, size=(self.emb_dim, self.emb_dim)
                    ).to(self.device),
                    requires_grad=True,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.sigmoid = torch.nn.Sigmoid()
        # self.mf_loss = BPRLoss()
        self.mf_loss = torch.nn.MSELoss() # edit
        self.LABEL = config["LABEL_FIELD"] # edit
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        # edit
        if self.use_llm_embed:
            self.apply(xavier_uniform_initialization_linearonly)
            if self.merge_embed != 'none':
                xavier_uniform_(self.model_user_embedding.weight.data)
                xavier_uniform_(self.model_item_embedding.weight.data)
        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_laplacian_matrix(self):
        r"""Get the laplacian matrix of users and items.

        .. math::
            L = I - D^{-1} \times A

        Returns:
            Sparse tensor of the laplacian matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        D = sp.diags(diag)
        A_tilde = D * A

        # covert norm_adj matrix to tensor
        A_tilde = sp.coo_matrix(A_tilde)
        row = A_tilde.row
        col = A_tilde.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_tilde.data)
        A_tilde = torch.sparse.FloatTensor(i, data, torch.Size(A_tilde.shape))

        # generate laplace matrix
        L = self.get_eye_mat(self.n_items + self.n_users) - A_tilde
        return L

    def get_eye_mat(self, num):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Args:
            num: number of column of the square matrix

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)
    
    # edit
    def get_cl_loss(self, model_embed, llm_embed):
        cl_loss = 0.0
        llm_embed2 = llm_embed.T
        # dots = torch.mm(model_embed, llm_embed2) # multiply
        dots = model_embed @ llm_embed.T
        
        if self.merge_embed == 'cl-cos':
            mod1 = torch.sqrt(model_embed.square().sum(-1, keepdim=True))
            mod2 = torch.sqrt(llm_embed.square().sum(-1, keepdim=True))
            # mod1 = mod1.unsqueeze(1)
            # mod2 = mod2.unsqueeze(0)
            moddot = mod1.T @ mod2
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

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
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
        # user_embeddings = self.user_embedding.weight
        # item_embeddings = self.item_embedding.weight
        # edit
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, cl_loss

    def forward(self):
        # edit
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
        # edit end
        all_embeddings, cl_loss = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            all_embeddings = self.sigmoid(torch.mm(all_embeddings, self.filters[k]))
            embeddings_list.append(all_embeddings)

        new_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            new_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings, cl_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID]
        label = interaction[self.LABEL]

        # edit
        user_ids, item_ids = list(set(user.tolist())), list(set(pos_item.tolist()))
        self.batch_user, self.batch_item = user_ids, item_ids
        self.batch_user, self.batch_item = torch.LongTensor(self.batch_user).to(self.device), torch.LongTensor(self.batch_item).to(self.device)
        # edit end

        user_all_embeddings, item_all_embeddings, cl_loss = self.forward()
        if self.use_llm_embed and self.merge_embed == 'cl-gen':
            model_all_embed = torch.concat([user_all_embeddings, item_all_embeddings], axis=0)
            print(model_all_embed.shape)
            llm_all_embed = torch.concat([self.llm_user_embedding, self.llm_item_embedding], axis=0)
            enc_embeds = model_all_embed[self.seeds]
            prf_embeds = llm_all_embed[self.seeds]
            enc_embeds = self.mlp(enc_embeds)
            cl_loss = self.get_ssl_con_loss(enc_embeds, prf_embeds, self.tau)
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        # neg_embeddings = item_all_embeddings[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        # mf_loss = self.mf_loss(pos_scores, neg_scores)
        mf_loss = self.mf_loss(pos_scores, label)


        reg_loss = self.reg_loss(u_embeddings,
                                 pos_embeddings,
                                 # neg_embeddings
                                 )
        loss = mf_loss + self.reg_weight * reg_loss
        # edit
        if cl_loss is not None:
            loss += cl_loss * self.beta
        # edit end

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
