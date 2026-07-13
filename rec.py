import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logging import getLogger, FileHandler
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, data_preparation_train
from recbole.model.context_aware_recommender import FM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, set_color, get_model
# from recbole.quick_start import run_recbole


from tqdm import tqdm
import os
import shutil
import copy
import pickle
import time
import numpy as np
import itertools
import random
import sys
import traceback
import wandb
from best_params import *

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['WANDB_API_KEY'] = 'Your key'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['WANDB_MODE'] = 'offline'

def get_groupid(a):
    # if a <= 20:
    #     return 0
    # elif 20 < a <= 30:
    #     return 1
    # elif 30 < a <= 40:
    #     return 2
    # elif 40 < a <= 50:
    #     return 3
    # elif 50 < a <= 60:
    #     return 4
    if a <= 10:
        return 0
    elif 10 < a <= 20:
        return 1
    elif 20 < a <= 35:
        return 2
    elif 35 < a <= 65:
        return 3
    elif 65 < a <= 80:
        return 4

def get_group(path):
    if "ml100k" in path:
        with open("{}/u_train.data".format(path), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
    elif "amazon" in path or "ml-25m" in path:
        with open("{}/u_train.data".format(path), "r") as f:
            lines = f.read().splitlines()
    user_dict = {}
    # user_groups = [[], [], [], [], []]
    for i in range(len(lines)):
        temp = lines[i].split("\t")
        if temp[0] not in user_dict.keys():
            user_dict[temp[0]] = 0
        user_dict[temp[0]] += 1
    return user_dict

# Recbole requirements: dataset_name.xxx.inter, dataset_name.user, dataset_name.item

def prepare_dataset(dataset):
    if dataset == "ml100k":
        path = "./ml100k/user_200_3"
    elif "amazon" in dataset:
        subset = dataset.split("-")[1]
        # path = "./amazon/user_200_3/{}".format(subset)
        path = "./amazon/user_5k/{}".format(subset)
    elif dataset == "ml-25m":
        path = "./ml-25m/user_5k"
        
    # if not os.path.exists("{}/recbole".format(path)):
    #     os.mkdir("{}/recbole".format(path))
    # if not os.path.exists("{}/recbole/user_200".format(path)):
    #     os.mkdir("{}/recbole/user_200".format(path))
    # if not os.path.exists("{}/recbole/user_200/user_200_total".format(path)):
    #     os.mkdir("{}/recbole/user_200/user_200_total".format(path))
    # for i in range(5):
    #     if not os.path.exists("{}/recbole/user_200/user_200_{}".format(path, i)):
    #         os.mkdir("{}/recbole/user_200/user_200_{}".format(path, i))
    
    if not os.path.exists("{}/recbole".format(path)):
        os.mkdir("{}/recbole".format(path))
    if not os.path.exists("{}/recbole/user_5k".format(path)):
        os.mkdir("{}/recbole/user_5k".format(path))
    if not os.path.exists("{}/recbole/user_5k/user_5k_total".format(path)):
        os.mkdir("{}/recbole/user_5k/user_5k_total".format(path))
    for i in range(5):
        if not os.path.exists("{}/recbole/user_5k/user_5k_{}".format(path, i)):
            os.mkdir("{}/recbole/user_5k/user_5k_{}".format(path, i))
    user_dict = get_group(path) # num of interactions of users in the training set.
    
    if dataset == "ml100k":
        data_columns = "user_id:token|item_id:token|rating:float|timestamp:float"
        user_columns = "user_id:token|age:token|gender:token|occupation:token|zip_code:token"
        item_columns = "item_id:token|movie_title:token_seq|genre:token_seq|release_year:token"
    elif "amazon" in dataset or dataset == "ml-25m":
        data_columns = "user_id:token|item_id:token|rating:float|timestamp:float"
        user_columns = "user_id:token|profile:token_seq"
        item_columns = "item_id:token|item_title:token_seq|category:token_seq|description:token_seq"
        
    
    user_grouplines = [[user_columns], [user_columns], [user_columns], [user_columns], [user_columns]]
    # item_grouplines = [[item_columns], [item_columns], [item_columns], [item_columns], [item_columns]]
    
    # print(user_dict)
    
    with open("{}/u.user".format(path), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        # print(lines[i])
        temp = lines[i].split("|")
        # print(temp[0], user_dict[temp[0]])
        group_id = get_groupid(user_dict[temp[0]])
        # print(group_id)
        # print(user_grouplines[group_id])
        user_grouplines[group_id].append(lines[i])
    
    # with open("{}/recbole/user_200/user_200_total/user_200_total.user".format(path), "w", encoding="latin-1") as f:
    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.user".format(path), "w", encoding="latin-1") as f:
        f.write("{}\n".format(user_columns))
        for i in range(len(lines)):
            f.write("{}\n".format(lines[i]))
    # for i in range(5):
    #     shutil.copy("{}/recbole/user_200/user_200_total/user_200_total.user".format(path),
    #                 "{}/recbole/user_200/user_200_{}/user_200_{}.user".format(path, i, i))
    
    for i in range(5):
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.user".format(path, i, i), "w", encoding="latin-1") as f:
            # f.write("{}\n".format(user_columns))
            for j in range(len(user_grouplines[i])):
                f.write("{}\n".format(user_grouplines[i][j]))
    
    # shutil.copy("{}/u.item".format(path), "{}/recbole/u.item".format(path))
    if dataset == "ml100k":
        with open("{}/u.item".format(path), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            temp = lines[i].split("|")
            temp[-1] = temp[-1].split("-")[-1]
            newline = ""
            for j in range(len(temp)):
                newline += temp[j]
                if j != len(temp) - 1:
                    newline += "|"
            lines[i] = newline
    elif "amazon" in dataset or dataset == "ml-25m":
        with open("{}/u.item".format(path), "r",) as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            temp = lines[i].split("|")
            temp2 = temp[2].split(". ")
            new_category, newline = "", ""
            for j in range(len(temp2) - 1):
                new_category += temp2[j]
                if j != len(temp2) - 2:
                    new_category += ", "
            for j in range(len(temp)):
                if j != 2:
                    newline += temp[j]
                else:
                    newline += new_category
                if j != len(temp) - 1:
                    newline += "|"
            lines[i] = newline

    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.item".format(path), "w") as f:
        f.write("{}\n".format(item_columns))
        for i in range(len(lines)):
            f.write("{}\n".format(lines[i]))
    # for i in range(5):
    #     shutil.copy("{}/recbole/user_200/user_200_total/user_200_total.item".format(path),
    #                 "{}/recbole/user_200/user_200_{}/user_200_{}.item".format(path, i, i))
    
    # for i in range(5):
    #     with open("{}/recbole/user_200/user_200_{}/user_200_{}.item".format(path, i, i), "w", encoding="latin-1") as f:
    #         f.write("{}\n".format(item_columns))
    #         for j in range(len(lines)):
    #             f.write("{}\n".format(lines[j]))
    
    train_grouplines = [[data_columns], [data_columns], [data_columns], [data_columns], [data_columns]]
    with open("{}/u_train.data".format(path), "r", encoding="latin-1") as f:
        train_lines = f.read().splitlines()
    with open("{}/u_valid.data".format(path), "r", encoding="latin-1") as f:
        valid_lines = f.read().splitlines()
    with open("{}/u_test.data".format(path), "r", encoding="latin-1") as f:
        test_lines = f.read().splitlines()
    for i in range(len(train_lines)):
        temp = train_lines[i].split("\t")
        group_id = get_groupid(user_dict[temp[0]])
        newline = ""
        for j in range(len(temp)):
            newline += temp[j]
            if j != len(temp) - 1:
                newline += "|"
        train_lines[i] = newline
        train_grouplines[group_id].append(train_lines[i])
    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.train.inter".format(path), "w", encoding="latin-1") as f:
        f.write("{}\n".format(data_columns))
        for i in range(len(train_lines)):
            f.write("{}\n".format(train_lines[i]))
    for i in range(5):
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.train.inter".format(path, i, i), "w", encoding="latin-1") as f:
            for j in range(len(train_grouplines[i])):
                f.write("{}\n".format(train_grouplines[i][j]))
    
    valid_grouplines = [[data_columns], [data_columns], [data_columns], [data_columns], [data_columns]]
    for i in range(len(valid_lines)):
        temp = valid_lines[i].split("\t")
        group_id = get_groupid(user_dict[temp[0]])
        newline = ""
        for j in range(len(temp)):
            newline += temp[j]
            if j != len(temp) - 1:
                newline += "|"
        valid_lines[i] = newline
        valid_grouplines[group_id].append(valid_lines[i])
    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.valid.inter".format(path), "w", encoding="latin-1") as f:
        f.write("{}\n".format(data_columns))
        for i in range(len(valid_lines)):
            f.write("{}\n".format(valid_lines[i]))
    for i in range(5):
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.valid.inter".format(path, i, i), "w", encoding="latin-1") as f:
            for j in range(len(valid_grouplines[i])):
                f.write("{}\n".format(valid_grouplines[i][j]))
    
    test_grouplines = [[data_columns], [data_columns], [data_columns], [data_columns], [data_columns]]
    for i in range(len(test_lines)):
        temp = test_lines[i].split("\t")
        group_id = get_groupid(user_dict[temp[0]])
        newline = ""
        for j in range(len(temp)):
            newline += temp[j]
            if j != len(temp) - 1:
                newline += "|"
        test_lines[i] = newline
        test_grouplines[group_id].append(test_lines[i])
    
    # temporarily valid == test
    
    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.test.inter".format(path), "w", encoding="latin-1") as f:
        f.write("{}\n".format(data_columns))
        for i in range(len(test_lines)):
            f.write("{}\n".format(test_lines[i]))
    # shutil.copy("{}/recbole/user_200/user_200_total/user_200_total.test.inter".format(path),
    #             "{}/recbole/user_200/user_200_total/user_200_total.valid.inter".format(path))
    
    for i in range(5):
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.test.inter".format(path, i, i), "w", encoding="latin-1") as f:
            for j in range(len(test_grouplines[i])):
                f.write("{}\n".format(test_grouplines[i][j]))
        # with open("{}/recbole/user_200/user_200_{}/user_200_{}.valid.inter".format(path, i, i), "w", encoding="latin-1") as f:
        #     for j in range(len(valid_grouplines[i])):
        #         f.write("{}\n".format(valid_grouplines[i][j]))
        # shutil.copy("{}/recbole/user_200/user_200_{}/user_200_{}.test.inter".format(path, i, i),
        #             "{}/recbole/user_200/user_200_{}/user_200_{}.valid.inter".format(path, i, i))
    
    items_dict = {}
    for i in range(len(lines)):
        if dataset == "ml100k" or dataset == "ml-25m":
            temp = int(lines[i].split("|")[0]) # Need to be modified in Amazon
        elif "amazon" in dataset:
            temp = lines[i].split("|")[0]
        items_dict[temp] = lines[i]
    
    for i in range(5):
        temp_dict = {}
        for j in range(1, len(train_grouplines[i])):
            if dataset == "ml100k" or dataset == "ml-25m":
                temp = int(train_grouplines[i][j].split("|")[1])
            elif "amazon" in dataset:
                temp = train_grouplines[i][j].split("|")[1]
            if temp not in temp_dict.keys():
                temp_dict[temp] = items_dict[temp]
        for j in range(1, len(valid_grouplines[i])):
            if dataset == "ml100k" or dataset == "ml-25m":
                temp = int(valid_grouplines[i][j].split("|")[1])
            elif "amazon" in dataset:
                temp = valid_grouplines[i][j].split("|")[1]
            if temp not in temp_dict.keys():
                temp_dict[temp] = items_dict[temp]
        for j in range(1, len(test_grouplines[i])):
            if dataset == "ml100k" or dataset == "ml-25m":
                temp = int(test_grouplines[i][j].split("|")[1])
            elif "amazon" in dataset:
                temp = test_grouplines[i][j].split("|")[1]
            if temp not in temp_dict.keys():
                temp_dict[temp] = items_dict[temp]

        temp_list = sorted(temp_dict.items(), key=lambda x:x[0])
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.item".format(path, i, i), "w") as f:
            f.write("{}\n".format(item_columns))
            for j in range(len(temp_list)):
                f.write("{}\n".format(temp_list[j][1]))

def gen_ablation_dataset(dataset):
    condense = False
    if condense:
        if dataset == "ml100k":
            path = "./ml100k/user_200_3"
            llm_path = "5-1"
        elif "amazon" in dataset:
            subset = dataset.split("-")[1]
            path = "./amazon/user_200_3/{}".format(subset)
            if subset == "CDs_and_Vinyl":
                llm_path = "5-2"
            elif subset == "Office_Products":
                llm_path = "5-3"
    else:
        if dataset == "ml100k":
            path = "./ml100k/user_200_3"
            llm_path = "99-1"
        elif "amazon" in dataset:
            subset = dataset.split("-")[1]
            path = "./amazon/user_200_3/{}".format(subset)
            if subset == "CDs_and_Vinyl":
                llm_path = "99-2-new"
            
    
    with open("{}/recbole/user_200/user_200_4/user_200_4.user".format(path), "r") as f:
        lines = f.read().splitlines()
    userid_list = []
    for i in range(1, len(lines)):
        cur_userid = lines[i].split("|", 1)[0]
        userid_list.append(cur_userid)
    
    for length in range(5, 65, 5):
        sub_folder = "user_200_total_{}_nc".format(length)
        if not os.path.exists("{}/recbole/user_200/{}".format(path, sub_folder)):
            os.mkdir("{}/recbole/user_200/{}".format(path, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.item".format(path), "{}/recbole/user_200/{}/{}.item".format(path, sub_folder, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.itememb".format(path), "{}/recbole/user_200/{}/{}.itememb".format(path, sub_folder, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.test.inter".format(path), "{}/recbole/user_200/{}/{}.test.inter".format(path, sub_folder, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.train.inter".format(path), "{}/recbole/user_200/{}/{}.train.inter".format(path, sub_folder, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.user".format(path), "{}/recbole/user_200/{}/{}.user".format(path, sub_folder, sub_folder))
        shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.valid.inter".format(path), "{}/recbole/user_200/{}/{}.valid.inter".format(path, sub_folder, sub_folder))
        with open("{}/recbole/user_200/{}/{}.useremb".format(path, sub_folder, sub_folder), "w") as g:
            g.write("uid:token|user_emb:float_seq\n")
            for i in range(len(userid_list)):
                cur_userid = userid_list[i]
                if length <= 50:
                    with open("./LLM_results/{}/checkpoints/user_{}_{}_totalembed.pkl".format(llm_path, cur_userid, length), "rb") as f:
                        cur_totalembed = pickle.load(f)
                elif length == 55:
                    if os.path.exists("./LLM_results/{}/checkpoints/user_{}_{}_totalembed.pkl".format(llm_path, cur_userid, length)):
                        with open("./LLM_results/{}/checkpoints/user_{}_{}_totalembed.pkl".format(llm_path, cur_userid, length), "rb") as f:
                            cur_totalembed = pickle.load(f)
                    else:
                        with open("./LLM_results/{}/checkpoints/user_{}_final_totalembed.pkl".format(llm_path, cur_userid, length), "rb") as f:
                            cur_totalembed = pickle.load(f)
                elif length == 60:
                    with open("./LLM_results/{}/checkpoints/user_{}_final_totalembed.pkl".format(llm_path, cur_userid, length), "rb") as f:
                        cur_totalembed = pickle.load(f)
                str_totalembed = str(cur_totalembed)[1:-1]
                g.write("{}|{}, \n".format(cur_userid, str_totalembed))
    
    # length = 60
    # if not os.path.exists("{}/recbole/user_200/user_200_total_{}".format(path, length)):
    #     os.mkdir("{}/recbole/user_200/user_200_total_{}".format(path, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.item".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.item".format(path, length, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.itememb".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.itememb".format(path, length, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.test.inter".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.test.inter".format(path, length, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.train.inter".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.train.inter".format(path, length, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.user".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.user".format(path, length, length))
    # shutil.copy("{}/recbole/user_200/user_200_4/user_200_4.valid.inter".format(path), "{}/recbole/user_200/user_200_total_{}/user_200_total_{}.valid.inter".format(path, length, length))
    # with open("{}/recbole/user_200/user_200_total_{}/user_200_total_{}.useremb".format(path, length, length), "w") as g:
    #     g.write("uid:token|user_emb:float_seq\n")
    #     for i in range(len(userid_list)):
    #         cur_userid = userid_list[i]
    #         with open("./LLM_results/{}/checkpoints/user_{}_final_totalembed.pkl".format(llm_path, cur_userid, length), "rb") as f:
    #             cur_totalembed = pickle.load(f)
    #         str_totalembed = str(cur_totalembed)[1:-1]
    #         g.write("{}|{}, \n".format(cur_userid, str_totalembed))
    
    sys.exit(1)

def gen_explore_dataset(dataset):
    if dataset == "ml100k":
        path = "./ml100k/user_200_3"
        llm_path = "5-1"
    elif "amazon" in dataset:
        subset = dataset.split("-")[1]
        path = "./amazon/user_200_3/{}".format(subset)
        if subset == "CDs_and_Vinyl":
            llm_path = "5-2"
    
    history_len = 10
    
    userembed_dict = {}
    for file in os.listdir("./LLM_results/{}/checkpoints".format(llm_path)):
        if file.endswith("{}_totalembed.pkl".format(history_len)):
            with open("./LLM_results/{}/checkpoints/{}".format(llm_path, file), "rb") as f:
                cur_embed = pickle.load(f)
            cur_userid = file.split("_")[1]
            if dataset == "ml100k":
                cur_userid = int(cur_userid)
            userembed_dict[cur_userid] = cur_embed
        if file.endswith("final_totalembed.pkl".format(history_len)):
            cur_userid = file.split("_")[1]
            if dataset == "ml100k":
                cur_userid = int(cur_userid)
            if not os.path.exists("./LLM_results/{}/checkpoints/user_{}_{}_totalembed.pkl".format(llm_path, cur_userid, history_len)):
                with open("./LLM_results/{}/checkpoints/{}".format(llm_path, file), "rb") as f:
                    cur_embed = pickle.load(f)
                userembed_dict[cur_userid] = cur_embed
            
    userembed_dict = {k:v for k, v in sorted(userembed_dict.items(), key=lambda x:x[0])}
    
    explore_path = "{}/recbole/user_200/user_200_total_{}_exp".format(path, history_len)
    if not os.path.exists(explore_path):
        os.mkdir(explore_path)
    with open("{}/user_200_total_{}_exp.useremb".format(explore_path, history_len), "w") as f:
        f.write("uid:token|user_emb:float_seq\n")
        for key in userembed_dict.keys():
            f.write("{}|".format(key))
            for i in range(len(userembed_dict[key])):
                f.write("{}, ".format(userembed_dict[key][i]))
            f.write("\n")
    
    for file in os.listdir("{}/recbole/user_200/user_200_total".format(path)):
        if not file.endswith("useremb"):
            filename, suffix = file.split(".", maxsplit=1)
            shutil.copy("{}/recbole/user_200/user_200_total/{}".format(path, file),
                        "{}/{}_{}_exp.{}".format(explore_path, filename, history_len, suffix))
    sys.exit(1)
    
def prepare_embedding(dataset):
    
    # # below code used only in ablation study
    # llm_result_folder = "99-3"
    # profile_length, dataset_num = llm_result_folder.split("-")
    # if dataset_num == "1":
    #     dataset = "ml100k"
    # elif dataset_num == "2":
    #     dataset = "amazon-CDs_and_Vinyl"
    # elif dataset_num == "3":
    #     dataset = "amazon-Office_Products"
    # # above code user only in ablation study
    
    if dataset == "ml100k":
        path = "./ml100k/user_200_3"
        name = dataset
    elif "amazon" in dataset:
        subset = dataset.split("-")[1]
        path = "./amazon/user_5k/{}".format(subset)
        name = "amazon"
    elif dataset == "ml-25m":
        path = "./ml-25m/user_5k"
        name = dataset
    
    user_dict = get_group(path)
    user_list = list(user_dict.keys())
    
    if dataset == "ml100k":
        with open("{}/u.item".format(path), "r", encoding="latin-1") as f:
            item_list = f.read().splitlines()
    elif "amazon" in dataset or dataset == "ml-25m":
        with open("{}/u.item".format(path), "r") as f:
            item_list = f.read().splitlines()
    
    for i in range(len(item_list)):
        item_list[i] = item_list[i].split("|")[0]
        
    # with open("./Embeddings/{}/user_totalembed.pkl".format(dataset), "rb") as f:
    #     user_totalembed = pickle.load(f)
    
    llm_result_folder = "{}_5_32_newprompt".format(name)
    
    user_embed_dict, user_embed_dict2 = {}, {}
    result_path = "./{}_results/{}/checkpoints".format(name, llm_result_folder)
    count = 0
    suffix = "final_totalembed.pkl"
    with tqdm(total=len(os.listdir(result_path))) as pbar:
        for file in os.listdir(result_path):
            if file.endswith(suffix):
                cur_userid = file.split("_")[1]
                if dataset == "ml-25m":
                    cur_userid = str(int(float(cur_userid)))
                with open("{}/{}".format(result_path, file), "rb") as f:
                    user_embed_dict[cur_userid] = pickle.load(f)
            if file.endswith("final_totalembed.pkl"):
                cur_userid = file.split("_")[1]
                if dataset == "ml-25m":
                    cur_userid = str(int(float(cur_userid)))
                prefix = file.split("final_totalembed.pkl")[0]
                if not os.path.exists("{}/{}{}".format(result_path, prefix, suffix)):
                    with open("{}/{}final_totalembed.pkl".format(result_path, prefix), "rb") as f:
                        user_embed_dict[cur_userid] = pickle.load(f)
                    count += 1
            pbar.update()
            # if file.endswith("final_totalembed2.pkl"):
            #     cur_userid = file.split("_")[1]
            #     with open("{}/{}".format(result_path, file), "rb") as f:
            #         user_embed_dict2[cur_userid] = pickle.load(f)
                    
                # with open("{}/{}".format(result_path, file), "rb") as f:
                #     if cur_userid == "60":
                #         user_embed_dict[cur_userid] = pickle.load(f)
                #     else:
                #         user_embed_dict[cur_userid] = user_embed_dict["60"]
    
    if dataset == "ml100k" or dataset == "ml-25m":
        with open("./Embeddings/{}/item_totalembed.pkl".format(dataset), "rb") as f:
            item_totalembed = pickle.load(f)
        # with open("./Embeddings/{}/item_totalembed2.pkl".format(dataset), "rb") as f:
        #     item_totalembed2 = pickle.load(f)
    elif "amazon" in dataset:
        with open("./Embeddings/amazon/{}/item_totalembed.pkl".format(subset), "rb") as f:
            item_totalembed = pickle.load(f)
        # with open("./Embeddings/amazon/{}/item_totalembed2.pkl".format(subset), "rb") as f:
        #     item_totalembed2 = pickle.load(f)
    
    item_embed_dict, item_embed_dict2 = {}, {}
    with tqdm(total=len(item_totalembed)) as pbar:
        for i in range(len(item_totalembed)):
            item_embed_dict[item_list[i]] = item_totalembed[i]
            pbar.update()
        # item_embed_dict2[item_list[i]] = item_totalembed2[i]
        # item_embed_dict[item_list[i]] = item_totalembed[0]
    
    # # below code are used in group 5 only
    # # remember to change the name of the folder before running the code.
    # print(user_dict)
    # time.sleep(1000)
    # user_list2 = list(user_dict.items())
    # temp_list = []
    # with tqdm(total=len(user_list2)) as pbar:
    #     for i in range(len(user_list2)):
    #         if 65 < user_list2[i][1] <= 80:
    #             temp_list.append(user_list2[i][0])
    #         pbar.update()
    # user_list = copy.deepcopy(temp_list)
    
    # print("{}/{}".format(count, len(user_list)))
    
    # with open("{}/recbole/user_5k/user_5k_4/user_5k_4.item".format(path), "r") as f:
    #     temp_lines = f.read().splitlines()
    # item_embed_dict2, item_list2 = {}, []
    # with tqdm(total=len(temp_lines) - 1) as pbar:
    #     for i in range(1, len(temp_lines)):
    #         cur_itemid, _, _, _ = temp_lines[i].split("|")
    #         item_embed_dict2[cur_itemid] = item_embed_dict[cur_itemid]
    #         item_list2.append(cur_itemid)
    #         pbar.update()
    # item_embed_dict = copy.deepcopy(item_embed_dict2)
    # item_list = copy.deepcopy(item_list2)
    # # src_file, dst_file
    
    # num = suffix.split("_")[0]
    # # llm_result_folder = llm_result_folder.rsplit("_", maxsplit=1)[0]
    # llm_result_folder += "_{}".format(num)
    # # llm_result_folder += "_"
    
    # if not os.path.exists("{}/recbole/user_5k/user_5k_total_{}".format(path, llm_result_folder)):
    #     os.mkdir("{}/recbole/user_5k/user_5k_total_{}".format(path, llm_result_folder))
    
    # shutil.copy("{}/recbole/user_5k/user_5k_4/user_5k_4.item".format(path), "{}/recbole/user_5k/user_5k_total_{}/user_5k_total_{}.item".format(path, llm_result_folder, llm_result_folder))
    # shutil.copy("{}/recbole/user_5k/user_5k_4/user_5k_4.test.inter".format(path), "{}/recbole/user_5k/user_5k_total_{}/user_5k_total_{}.test.inter".format(path, llm_result_folder, llm_result_folder))
    # shutil.copy("{}/recbole/user_5k/user_5k_4/user_5k_4.train.inter".format(path), "{}/recbole/user_5k/user_5k_total_{}/user_5k_total_{}.train.inter".format(path, llm_result_folder, llm_result_folder))
    # shutil.copy("{}/recbole/user_5k/user_5k_4/user_5k_4.user".format(path), "{}/recbole/user_5k/user_5k_total_{}/user_5k_total_{}.user".format(path, llm_result_folder, llm_result_folder))
    # shutil.copy("{}/recbole/user_5k/user_5k_4/user_5k_4.valid.inter".format(path), "{}/recbole/user_5k/user_5k_total_{}/user_5k_total_{}.valid.inter".format(path, llm_result_folder, llm_result_folder))

    # user_columns = "uid:token|user_emb:float_seq"
    # item_columns = "iid:token|item_emb:float_seq"
    # # path2 = "./ml100k/user_200/recbole/user_200"
    # path2 = "{}/recbole/user_5k".format(path)
    # with open("{}/user_5k_total_{}/user_5k_total_{}.useremb".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
    #     f.write("{}\n".format(user_columns))
    #     with tqdm(total=len(user_list)) as pbar:
    #         for i in range(len(user_list)):
    #             temp_emb = ""
    #             cur_emb = user_embed_dict[user_list[i]]
    #             for j in range(len(cur_emb)):
    #                 temp_emb += str(cur_emb[j])
    #                 temp_emb += ", "
    #             cur_line = "{}|{}\n".format(user_list[i], temp_emb)
    #             f.write(cur_line)
    #             pbar.update()
    
    # with open("{}/user_5k_total_{}/user_5k_total_{}.itememb".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
    #     f.write("{}\n".format(item_columns))
    #     with tqdm(total=len(item_list)) as pbar:
    #         for i in range(len(item_list)):
    #             temp_emb = ""
    #             cur_emb = item_embed_dict[item_list[i]]
    #             for j in range(len(cur_emb)):
    #                 temp_emb += str(cur_emb[j])
    #                 temp_emb += ", "
    #             cur_line = "{}|{}\n".format(item_list[i], temp_emb)
    #             f.write(cur_line)
    #             pbar.update()
    # sys.exit(1)
    # # above code are used in group 5 only
    
    user_columns = "uid:token|user_emb:float_seq"
    item_columns = "iid:token|item_emb:float_seq"
    # path2 = "./ml100k/user_200/recbole/user_200"
    path2 = "{}/recbole/user_5k".format(path)
    
    groups = [[], [], [], [], []]
    groups2 = [[], [], [], [], []]
    
    if not os.path.exists("{}/user_5k_total_{}".format(path2, llm_result_folder)):
        os.mkdir("{}/user_5k_total_{}".format(path2, llm_result_folder))
        print("Create folder: {}/user_5k_total_{}".format(path2, llm_result_folder))
    else:
        print("Using folder: {}/user_5k_total_{}".format(path2, llm_result_folder))
    with open("{}/user_5k_total_{}/user_5k_total_{}.useremb".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
        f.write("{}\n".format(user_columns))
        with tqdm(total=len(user_dict)) as pbar:
            for i in range(len(user_list)):
                temp_emb = ""
                cur_emb = user_embed_dict[user_list[i]]
                for j in range(len(cur_emb)):
                    temp_emb += str(cur_emb[j])
                    temp_emb += ", "
                cur_line = "{}|{}\n".format(user_list[i], temp_emb)
                f.write(cur_line)
                groups[get_groupid(user_dict[user_list[i]])].append(cur_line)
                pbar.update()
    # with open("{}/user_200_total_{}/user_200_total_{}.useremb2".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
    #     f.write("{}\n".format(user_columns))
    #     for i in range(len(user_list)):
    #         temp_emb = ""
    #         cur_emb = user_embed_dict2[user_list[i]]
    #         for j in range(len(cur_emb)):
    #             temp_emb += str(cur_emb[j])
    #             temp_emb += ", "
    #         cur_line = "{}|{}\n".format(user_list[i], temp_emb)
    #         f.write(cur_line)
    #         groups2[get_groupid(user_dict[user_list[i]])].append(cur_line)
    
    # Temporarily comment out the generation of group file.
    # for i in range(5):
    #     with open("{}/user_200_{}/user_200_{}.useremb".format(path2, i, i), "w", encoding='latin-1') as f:
    #         f.write("{}\n".format(user_columns))
    #         for j in range(len(groups[i])):
    #             f.write(groups[i][j])
    #     with open("{}/user_200_{}/user_200_{}.useremb2".format(path2, i, i), "w", encoding='latin-1') as f:
    #         f.write("{}\n".format(user_columns))
    #         for j in range(len(groups2[i])):
    #             f.write(groups2[i][j])
    
    # groups = [[], [], [], [], []]
    item_groups = [{}, {}, {}, {}, {}]
    item_groups2 = [{}, {}, {}, {}, {}]
    for i in range(5):
        with open("{}/user_5k_{}/user_5k_{}.item".format(path2, i, i), "r", encoding='latin-1') as f:
            lines = f.read().splitlines()
        with tqdm(total=len(lines) - 1) as pbar:
            for j in range(1, len(lines)):
                temp = lines[j].split("|")[0]
                item_groups[i][temp] = None
                item_groups2[i][temp] = None
                pbar.update()
        # print(len(item_groups[i]))
    
    with open("{}/user_5k_total_{}/user_5k_total_{}.itememb".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
        f.write("{}\n".format(item_columns))
        with tqdm(total=len(item_list)) as pbar:
            for i in range(len(item_list)):
                temp_emb = ""
                cur_emb = item_embed_dict[item_list[i]]
                for j in range(len(cur_emb)):
                    temp_emb += str(cur_emb[j])
                    temp_emb += ", "
                cur_line = "{}|{}\n".format(item_list[i], temp_emb)
                f.write(cur_line)
                for j in range(5):
                    if item_list[i] in item_groups[j].keys():
                        item_groups[j][item_list[i]] = cur_line
                pbar.update()
    # with open("{}/user_200_total_{}/user_200_total_{}.itememb2".format(path2, llm_result_folder, llm_result_folder), "w", encoding='latin-1') as f:
    #     f.write("{}\n".format(item_columns))
    #     for i in range(len(item_list)):
    #         temp_emb = ""
    #         cur_emb = item_embed_dict2[item_list[i]]
    #         for j in range(len(cur_emb)):
    #             temp_emb += str(cur_emb[j])
    #             temp_emb += ", "
    #         cur_line = "{}|{}\n".format(item_list[i], temp_emb)
    #         f.write(cur_line)
    #         for j in range(5):
    #             if item_list[i] in item_groups2[j].keys():
    #                 item_groups2[j][item_list[i]] = cur_line
    
    for postfix in ['.item', '.test.inter', '.train.inter', '.user', '.valid.inter']:
        shutil.copy("{}/user_5k_total/user_5k_total{}".format(path2, postfix),
                    "{}/user_5k_total_{}/user_5k_total_{}{}".format(path2, llm_result_folder, llm_result_folder, postfix))
    
    sys.exit(1)
    # for i in range(5):
    #     cur_item_group = sorted(item_groups[i].items(), key=lambda x:x[0])
    #     cur_item_group2 = sorted(item_groups2[i].items(), key=lambda x:x[0])
    #     with open("{}/user_200_{}/user_200_{}.itememb".format(path2, i, i), "w", encoding='latin-1') as f:
    #         f.write("{}\n".format(item_columns))
    #         for j in range(len(cur_item_group)):
    #             f.write("{}".format(cur_item_group[j][1]))

    #     with open("{}/user_200_{}/user_200_{}.itememb2".format(path2, i, i), "w", encoding='latin-1') as f:
    #         f.write("{}\n".format(item_columns))
    #         for j in range(len(cur_item_group2)):
    #             f.write("{}".format(cur_item_group2[j][1]))
    # print(len(user_totalembed), len(user_totalembed[0]))

def ndcg_naive(test_tag_sorted): # Using when there is only one positive item, and use 0/1 to represent the score.
    for i in range(len(test_tag_sorted)):
        if test_tag_sorted[i][1] == 1:
            return float(1 / np.log2(i + 2))

def mrr(test_tag_sorted):
    for i in range(len(test_tag_sorted)):
        if test_tag_sorted[i][1] == 1:
            return 1 / (i + 1)

def get_ndcg_mrr(zipped_list):
    # print(zipped_list)
    count = 0
    dcg, idcg, mrr = 0.0, 0.0, 0.0
    for i in range(len(zipped_list)):
        # if zipped_list[i][1] >= 4:
        if zipped_list[i][1] == 1:
            count += 1
            dcg += float(1 / np.log2(i + 2))
            mrr += float(1 / (i + 1))
    for i in range(count):
        idcg += float(1 / np.log2(i + 2))
    if count == 0:
        return 0.0, 0.0
    mrr /= count
    return dcg / idcg, mrr

def get_bpref(zipped_list):
    bpref_score = 0.0
    pos_num, neg_num = 0, 0
    negnum_list = np.zeros(len(zipped_list))
    for i in range(len(zipped_list)):
        # if i >= 1 and 0 < zipped_list[i - 1][0] <= 3:
        if i >= 1 and zipped_list[i - 1][1] == -1:
            negnum_list[i] = negnum_list[i - 1] + 1
        else:
            negnum_list[i] = negnum_list[i - 1]
        # if zipped_list[i][0] >= 4:
        if zipped_list[i][1] == 1:
            pos_num += 1
        elif zipped_list[i][1] == -1:
            neg_num += 1
    denominator = min(pos_num, neg_num)
    for i in range(len(zipped_list)): # modified, previous metric is wrong
        if negnum_list[i] > pos_num:
            negnum_list[i] = pos_num
    if denominator == 0:
        return 0.0
    for i in range(len(zipped_list)):
        # if zipped_list[i][0] >= 4:
        if zipped_list[i][1] == 1:
            bpref_score += (1 - negnum_list[i] / denominator)
    bpref_score /= pos_num
    return bpref_score

def diff(test_tag_sorted):
    for i in range(len(test_tag_sorted)):
        if test_tag_sorted[i][1] == 1:
            pos_rank = i
        elif test_tag_sorted[i][1] == -1:
            neg_rank = i
            neg_mrr = 1 / (i + 1)
    return neg_mrr, pos_rank, neg_rank

def group_eval(userinter_dict, file, eval_length, type):
    with open(file, "r", encoding='latin-1') as f:
        userid_list = f.read().splitlines()
    # print(len(userid_list))
    random.seed(2024)
    inter_sortdict, return_dict = {}, {}
    for i in range(1, len(userid_list)):
        userid = userid_list[i].split("|")[0]
        interaction = userinter_dict[userid]
        # print(interaction)
        # time.sleep(1000)
        # random.seed(int(time.time() * 1e7 % 1e7))
        random.shuffle(interaction)
        # with open("./check_score.txt", "a") as f:
        #     f.write("User {} before ranking:\n".format(userid))
        #     for j in range(len(interaction)):
        #         f.write("{}\t{}\t{}\n".format(interaction[j][0], interaction[j][1], interaction[j][2]))
        # print(interaction)
        # time.sleep(1000)
        interaction = sorted(interaction, key=lambda x:x[2], reverse=True)
            # f.write("User {} after ranking:\n".format(userid))
            # for j in range(len(interaction)):
            #     f.write("{}\t{}\t{}\n".format(interaction[j][0], interaction[j][1], interaction[j][2]))
        inter_sortdict[userid] = interaction
    # print(inter_sortdict)
    # time.sleep(1000)
    
    pointwise_dict = {}
    
    for i in range(len(eval_length)):
        total_mrr, total_ndcg = 0.0, 0.0
        # totaldiff_mrr, totaldiff_rank = 0.0, 0.0
        total_bpref = 0.0
        ndcg_list, mrr_list, bpref_list = [], [], []
        
        for key in inter_sortdict.keys():
            interaction = inter_sortdict[key][:eval_length[i]]
            # cur_ndcg, cur_mrr = ndcg_naive(interaction), mrr(interaction)
            # neg_mrr, pos_rank, neg_rank = diff(interaction)
            cur_ndcg, cur_mrr = get_ndcg_mrr(interaction)
            cur_bpref = get_bpref(interaction)
            total_ndcg += cur_ndcg
            total_mrr += cur_mrr
            total_bpref += cur_bpref
            
            ndcg_list.append(cur_ndcg)
            mrr_list.append(cur_mrr)
            bpref_list.append(cur_bpref)
            # total_mrr += cur_mrr
            # total_ndcg += cur_ndcg
            # totaldiff_mrr += (cur_mrr - neg_mrr)
            # totaldiff_rank += (pos_rank - neg_rank)
        total_ndcg /= (len(userid_list) - 1)
        total_mrr /= (len(userid_list) - 1)
        # totaldiff_mrr /= (len(userid_list) - 1)
        # totaldiff_rank /= (len(userid_list) - 1)
        total_bpref /= (len(userid_list) - 1)
        return_dict[eval_length[i]] = {'ndcg@{}'.format(eval_length[i]):total_ndcg, 'mrr@{}'.format(eval_length[i]):total_mrr, 'bpref@{}'.format(eval_length[i]):total_bpref}
        pointwise_dict[eval_length[i]] = {'ndcg@{}'.format(eval_length[i]):ndcg_list, 'mrr@{}'.format(eval_length[i]):mrr_list, 'bpref@{}'.format(eval_length[i]):bpref_list}
    # return {'ndcg@20': total_ndcg, 'mrr@20': total_mrr, 'diff_mrr@20': totaldiff_mrr, 'diff_rank@20': totaldiff_rank}
    # return {'ndcg@20': total_ndcg, 'mrr@20': total_mrr, 'bpref@20': total_bpref}
    return return_dict, pointwise_dict

def manual_eval(dataset, field2id_token, saved_model_file, logger, filename, eval_length, dataset_choice):
    folder = saved_model_file.split(".")[0]
    batch_num = 0
    total_userid = []
    total_itemid = []
    total_scores = []
    while True:
        if not os.path.exists("{}/{}/batched_data_{}.pkl".format(folder, filename, batch_num)):
            break
        # print(folder, filename, batch_num)
        # time.sleep(10)
        with open("{}/{}/batched_data_{}.pkl".format(folder, filename, batch_num), "rb") as f:
            interaction, scores, positive_u, positive_i = pickle.load(f)
            cur_userid = interaction['user_id'].tolist()
            cur_itemid = interaction['item_id'].tolist()
            cur_scores = scores.tolist()
            # with open("./check_score.txt", "w") as g:
            #     for i in range(len(cur_userid)):
            #         g.write("{}\t{}\t{}\n".format(cur_userid[i], cur_itemid[i], cur_scores[i]))
            
            total_userid += cur_userid
            total_itemid += cur_itemid
            total_scores += cur_scores
        batch_num += 1
    
    for i in range(len(total_userid)):
        total_userid[i] = field2id_token['user_id'][total_userid[i]]
    for i in range(len(total_itemid)):
        total_itemid[i] = field2id_token['item_id'][total_itemid[i]]
    
    userinter_dict = {}
    userinter_dict2 = {}
    if dataset == "ml100k":
        path = "./ml100k/user_200_3/recbole/user_200"
    elif "amazon" in dataset_type:
        subset = dataset_type.split("-")[1]
        path = "./amazon/user_5k/{}/recbole/user_5k".format(subset)
    elif dataset == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k"
    
    with open("{}/{}/{}.{}.inter".format(path, dataset_choices, dataset_choices, filename), "r", encoding='latin-1') as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        user_id, item_id, rating, timestamp = lines[i].split("|")
        if user_id not in userinter_dict.keys():
            userinter_dict[user_id] = {}
        if float(rating) >= 4:
            is_pos = 1
        elif 0 < float(rating) < 4:
            is_pos = -1
        else:
            is_pos = 0
        userinter_dict[user_id][item_id] = [item_id, is_pos]
    for i in range(len(total_userid)):
        userinter_dict[total_userid[i]][total_itemid[i]].append(total_scores[i])
    for key in userinter_dict.keys():
        cur_inter = list(userinter_dict[key].items())
        userinter_dict2[key] = []
        for i in range(len(cur_inter)):
            userinter_dict2[key].append(cur_inter[i][1])
    # with open("{}/userinter_dict.pkl".format(folder), "wb") as f:
    #     pickle.dump(userinter_dict2, f)
    
    result_list, pointwise_list = [], []
    # total_test_result = group_eval(userinter_dict2, "{}/user_200_total/user_200_total.userprof".format(path))
    total_test_result, total_pointwise_result = group_eval(userinter_dict2, "{}/{}/{}.user".format(path, dataset_choices, dataset_choices), eval_length, "full")
    logger.info(set_color("total {} result".format(filename), "yellow") + f": {total_test_result}")
    result_list.append(total_test_result)
    pointwise_list.append(total_pointwise_result)
    
    # only evaluate one group, please comment out the code below
    for i in range(5):
        # group_test_result = group_eval(userinter_dict2, "{}/user_200_{}/user_200_{}.userprof".format(path, i, i))
        group_test_result, group_pointwise_result = group_eval(userinter_dict2, "{}/user_5k_{}/user_5k_{}.user".format(path, i, i), eval_length, "group_{}".format(i))
        logger.info(set_color("{} result of group {}".format(filename, i + 1), "yellow") + f": {group_test_result}")
        result_list.append(group_test_result)
        pointwise_list.append(group_pointwise_result)
    return result_list, pointwise_list

def train_eval(dataset_type, model_name, config_dict, eval_length):
    # if dataset_type == "ml100k":
    #     path = "./ml100k/user_200_3/recbole/user_200"
    # elif "amazon" in dataset_type:
    #     subset = dataset_type.split("-")[1]
    #     path = "./amazon/user_5k/{}/recbole/user_5k".format(subset)
    # elif dataset_type == "ml-25m":
    #     path = "./ml-25m/user_5k/recbole/user_5k"
    
    # if use_llm_embed:
    #     embed_config = {
    #         'load_col': {
    #             'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
    #             'user': ['user_id', 'user_emb'],
    #             'item': ['item_id', 'item_emb']
    #         },
    #         # 'alias_of_user_id': ['uid'],
    #         # 'alias_of_item_id': ['iid'],
    #         # 'preload_weight': {
    #         #     'uid': 'useremb',
    #         #     'iid': 'itememb'
    #         # },
    #         'use_llm_embed': True
    #     }
    #     config_dict.update(embed_config)
    
    # dataset_choice = dataset_choices
    
    # dataset_choice = 'user_200_total_10_exp'
    # dataset_choice = 'user_200_total_base'
    # dataset_choice = 'user_200_total_55'
    # dataset_choice = 'user_200_total'
    
    # config_dict_total = copy.deepcopy(config_dict)
    
    # total_dataconfig = {
    #     'data_path': path,
    #     'dataset': dataset_choice
    # }
    # config_dict_total.update(total_dataconfig)
    
    # total_config = Config(model=model_name, config_dict=config_dict_total)
    # print(total_config["device"])
    # time.sleep(1000)
    init_seed(total_config['seed'], total_config['reproducibility'])
    try:
        # print(new_logger)
        # time.sleep(5)
        if new_logger is None:
            init_logger(total_config)
            logger = getLogger()
        else:
            logger = new_logger
    except NameError:
        init_logger(total_config)
        logger = getLogger()
    logger.info(total_config)
    
    # total_dataset = create_dataset(total_config)
    # logger.info(total_dataset)
    
    # total_train_data, total_valid_data, total_test_data = data_preparation(total_config, total_dataset)
    # total_train_data, total_valid_data, total_test_data = quickload(dataset_type)
    
    # print("1")
    # print(total_train_data.dataset.user_feat['user_emb'])
    
    # if model_name == "XSimGCL":
    #     model = get_model(model_name)(total_config, total_train_data).to(total_config['device'])
    # else:
    #     model = get_model(model_name)(total_config, total_train_data.dataset).to(total_config['device'])
    model = get_model(model_name)(total_config, total_train_data.dataset).to(total_config['device'])
    logger.info(model)
    
    trainer = Trainer(total_config, model)
    # time.sleep(1000)
    try:
        best_valid_score, best_valid_result = trainer.fit_preprocess(total_train_data, total_valid_data, interaction_list, evaluation_list, batched_data_list)
    except Exception as e:
        logger.info(e)
        traceback.print_exc()
        # field2id_token = total_dataset.field2id_token
        # saved_model_file = trainer.saved_model_file
        # total_valid_result = trainer.evaluate(total_valid_data, filename="valid")
        # valid_result_list = manual_eval(dataset_type, field2id_token, saved_model_file, logger, "valid", eval_length, dataset_choice)
        
        # total_test_result = trainer.evaluate(total_test_data, filename="test")
        # test_result_list = manual_eval(dataset_type, field2id_token, saved_model_file, logger, "test", eval_length, dataset_choice)
        # logger.info(set_color("total test result", "yellow") + f": {total_test_result}")
        
        saved_model_file = trainer.saved_model_file
        # folder = saved_model_file.split(".")[0]
        # try:
        #     os.remove(saved_model_file)
        # except:
        #     pass
        # os.system("rm -rf {}".format(folder))
        best_valid_result = "unknown"
        
        # if model_name == "FM":
        #     return None, None, None, None, logger
        # return valid_result_list, test_result_list, saved_model_file, logger
        # return None, None, None, logger

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    
    field2id_token = total_dataset.field2id_token
    saved_model_file = trainer.saved_model_file
    # print(type(total_test_data))
    # time.sleep(1000)
    # total_valid_result = trainer.evaluate_preprocess(total_valid_data, filename="valid")
    # print(total_valid_result)
    # time.sleep(1000)
    # total_valid_result = trainer.evaluate(total_valid_data, filename="valid")
    # try:
    total_valid_result = trainer.evaluate(total_valid_data, filename="valid")
    # except Exception as e:
    #     print(e)
    #     return None, None, None, None, logger
    valid_result_list, _ = manual_eval(dataset_type, field2id_token, saved_model_file, logger, "valid", eval_length, dataset_choice)
    
    total_test_result = trainer.evaluate(total_test_data, filename="test")
    test_result_list, pointwise_test_list = manual_eval(dataset_type, field2id_token, saved_model_file, logger, "test", eval_length, dataset_choice)
    logger.info(set_color("total test result", "yellow") + f": {total_test_result}")
    
    logger.info("Evaluation finished, start to clean up the saved model file.")
    # logger.info("Tuning Param", tuning_param)
    if tuning_param == "Complement":
        if use_llm_embed == False:
            tuple_result = trainer.evaluate(total_train_data_full, filename="train")
        else:
            tuple_result = None
        return valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger, tuple_result
    else:
        return valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger
    
    
    # config_list = []
    # test_data_list = []
    # for i in range(5):
    #     group_dataconfig = {
    #         'data_path': path,
    #         'dataset': 'user_200_{}'.format(i)
    #     }
    #     group_config_dict = copy.deepcopy(config_dict)
    #     group_config_dict.update(group_dataconfig)
        
    #     group_config = Config(model=model_name, config_dict=group_config_dict)
    #     # logger.info(group_config)
    #     group_dataset = create_dataset(group_config)
    #     group_train_data, group_valid_data, group_test_data = data_preparation(group_config, group_dataset)
        
    #     # print(group_test_data.dataset.shape)
    #     # time.sleep(1000)
        
    #     # group_model = get_model(model_name)(group_config, group_train_data.dataset).to(group_config['device'])
    #     # logger.info(group_model)
    #     # group_trainer = Trainer(group_config, group_model)
    #     # group_test_result = group_trainer.evaluate(group_test_data, model_file=saved_model_file)
        
    #     group_test_result = trainer.evaluate(group_test_data, model_file=saved_model_file)
    #     logger.info(set_color("test result of group {}".format(i + 1), "yellow") + f": {group_test_result}")

def param_tuning(dataset_type, model_name, config_dict, use_llm_embed, tuning_param):
    best_params_dict = Best_params()
    eval_length = [1, 5, 10, 20]
    if tuning_param in ['True', 'Wandb', 'Complement', 'Simple', 'True_1diff']:
        global new_logger
        new_logger = None
        if not use_llm_embed:
            if model_name == 'LightGCN':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                n_layers = [1, 2, 3, 4]
                # reg_weight = [1e-05, 1e-04, 1e-03, 1e-02]
                reg_weight = [1e-05]
                perm = list(itertools.product(learning_rate, n_layers, reg_weight))
                attr = ["learning_rate", "n_layers", "reg_weight"]
            elif model_name == 'FM':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                perm = list(itertools.product(learning_rate))
                attr = ["learning_rate"]
            elif model_name == 'WideDeep':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                dropout_prob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                mlp_hidden_size = ['[32,16,8]', '[64,32,16]', '[128,64,32]', '[256,128,64]']
                perm = list(itertools.product(learning_rate, dropout_prob, mlp_hidden_size))
                attr = ["learning_rate", "drouput_prob", "mlp_hidden_size"]
            elif model_name == 'ItemKNN':
                k = [10, 50, 100, 200, 250, 300, 400, 500, 1000, 1500, 2000, 2500]
                shrink = [0.0, 1.0]
                perm = list(itertools.product(k, shrink))
                attr = ["k", "shrink"]
            elif model_name == 'SimpleX':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                # gamma = [0.3, 0.5, 0.7]
                gamma = [0.3]
                history_len = [0, 10, 20, 30, 40, 50]
                perm = list(itertools.product(learning_rate, gamma, history_len))
                attr = ["learning_rate", "gamma", "history_len"]
            elif model_name == 'NeuMF':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                # dropout_prob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                # dropout_prob = [0.0, 0.1, 0.2, 0.3]
                dropout_prob = [0.0]
                mlp_hidden_size = ['[64,32,16]', '[32,16,8]', '[128,64]']
                perm = list(itertools.product(learning_rate, dropout_prob, mlp_hidden_size))
                attr = ["learning_rate", "dropout_prob", "mlp_hidden_size"]
            elif model_name == 'NCL':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                ssl_temp = [0.05, 0.07, 0.1]
                ssl_reg = [1e-6, 1e-7]
                proto_reg = [1e-6, 1e-7, 1e-8]
                n_layers = [3]
                reg_weight = [1e-04]
                perm = list(itertools.product(learning_rate, ssl_temp, ssl_reg, proto_reg, n_layers, reg_weight))
                attr = ["learning_rate", "ssl_temp", "ssl_reg", "proto_reg", "n_layers", "reg_weight"]
            elif model_name == 'SpectralCF':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                reg_weight = [0.01, 0.002, 0.001, 0.0005]
                n_layers = [1, 2, 3, 4]
                perm = list(itertools.product(learning_rate, reg_weight, n_layers))
                attr = ["learning_rate", "reg_weight", "n_layers"]
            elif model_name == 'SGL':
                learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                ssl_tau = [0.1, 0.2, 0.5, 1.0]
                drop_ratio = [0, 0.1, 0.2, 0.4, 0.5]
                ssl_weight = [0.005, 0.05, 0.1, 0.5, 1.0]
                n_layers = [3]
                reg_weight = [1e-05]
                perm = list(itertools.product(learning_rate, ssl_tau, drop_ratio, ssl_weight, n_layers, reg_weight))
                attr = ["learning_rate", "ssl_tau", "drop_ratio", "ssl_weight", "n_layers", "reg_weight"]
            elif model_name == 'SimGCL':
                # learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                # n_layers = [1, 2, 3, 4]
                reg_weight = [1e-04]
                lamda = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
                eps = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
                temperature = [0.2]
                perm = list(itertools.product(reg_weight, lamda, eps, temperature))
                attr = ["reg_weight", "lamda", "eps", "temperature"]
            elif model_name == 'XSimGCL':
                # learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                # n_layers = [1, 2, 3, 4]
                reg_weight = [1e-04]
                # lamda = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
                lamda = [0.01, 0.05, 0.1]
                eps = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
                temperature = [0.2]
                # layer_cl = [1, 2, 3, 4]
                layer_cl = [1, 2]
                perm = list(itertools.product(reg_weight, lamda, eps, temperature, layer_cl))
                attr = ["reg_weight", "lambda", "eps", "temperature", "layer_cl"]
            elif model_name == 'LightGCL':
                dropout = [0, 0.25]
                temp = [0.3, 0.5, 1, 3, 10]
                lambda1 = [1e-5, 1e-6, 1e-7]
                lambda2 = [1e-4, 1e-5]
                q = [5]
                perm = list(itertools.product(dropout, temp, lambda1, lambda2, q))
                attr = ["dropout", "temp", "lambda1", "lambda2", "q"]
        else:
            merge_embed = config_dict['merge_embed']
            learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            attr = ["learning_rate"]
            if merge_embed == 'none' or merge_embed == 'add':
                perm = list(itertools.product(learning_rate))
            elif merge_embed[0:2] == 'cl':
                if tuning_param in ['True', 'True_1diff']:
                    if merge_embed != 'cl-gen':
                        reduced_tuning_lists = best_params_dict.reduced_tuning_lists[dataset_type][model_name]
                        learning_rate = reduced_tuning_lists['learning_rate']
                        beta = reduced_tuning_lists['beta']
                        tau = reduced_tuning_lists['tau']
                        # beta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
                        # beta = [1]
                        
                        # sample_num = [9, 19]
                        
                        # tau = [1, 0.5, 0.1, 0.05, 0.01]
                        # tau = [0.01]
                        attr.extend(["beta", "tau"])
                        perm = list(itertools.product(learning_rate, beta, tau))
                    else:
                        beta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
                        tau = [1, 0.5, 0.1, 0.05, 0.01]
                        mask_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
                        attr.extend(["beta", "tau", "mask_ratio"])
                        perm = list(itertools.product(learning_rate, beta, tau, mask_ratio))
                elif tuning_param == 'Wandb':
                    setup_wandb()
        if tuning_param in ['True', 'Complement', 'Simple', 'True_1diff']:
            seed_list = [2020, 2021, 2022, 2023, 2024]
            # seed_list = [2025, 2026, 2027, 2028, 2029]
            # seed_list = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]
            # seed_list = [2020]
            total_ndcg, total_mrr, total_bpref = {}, {}, {}
            total_point_ndcg, total_point_mrr, total_point_bpref = {}, {}, {}
            if tuning_param == 'Complement' and (not use_llm_embed):
                total_train_tuple = []
            
            for i in range(len(eval_length)):
                total_ndcg[eval_length[i]] = np.zeros(6)
                total_mrr[eval_length[i]] = np.zeros(6)
                total_bpref[eval_length[i]] = np.zeros(6)
                
                total_point_ndcg[eval_length[i]] = [[], [], [], [], [], []]
                total_point_mrr[eval_length[i]] = [[], [], [], [], [], []]
                total_point_bpref[eval_length[i]] = [[], [], [], [], [], []]
                
            for seed in seed_list:
                temp = {'seed': seed}
                config_dict.update(temp)
                
                global total_train_data, total_valid_data, total_test_data, total_config, total_dataset, interaction_list, evaluation_list, batched_data_list, total_train_data_full, total_dataset_full
                total_train_data, total_valid_data, total_test_data, total_config, total_dataset, interaction_list, evaluation_list, batched_data_list, total_train_data_full, total_dataset_full = quickload(dataset_type, config_dict)
                
                if model_name in ["SimGCL", "XSimGCL", "LightGCL"]:
                    static_params = best_params_dict.basic_params[dataset_type]['LightGCN'][str(seed)]
                    total_config.final_config_dict.update(static_params)
                    # config_dict.update(static_params)
                if use_llm_embed:
                    static_params = best_params_dict.basic_params[dataset_type][model_name][str(seed)]
                    total_config.final_config_dict.update(static_params)
                    # config_dict.update(static_params)
                best_ndcg, best_params, best_results, best_pointwise = 0.0, tuple(), list(), list()
                if tuning_param == "True":
                    # print(perm)
                    for i in range(len(perm)):
                        cur_params = {}
                        for j in range(len(perm[i])):
                            cur_params[attr[j]] = perm[i][j]
                        # config_dict.update(cur_params)
                        total_config.final_config_dict.update(cur_params)
                        valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, eval_length)
                        # global new_logger
                        new_logger = logger
                        logger.info("Above is the result of parameter: {}".format(cur_params))
                        if valid_result_list is None and test_result_list is None and saved_model_file is None:
                            continue
                        if valid_result_list[0][20]['ndcg@20'] > best_ndcg:
                            best_ndcg = valid_result_list[0][20]['ndcg@20']
                        # if valid_result_list[0]['bpref@20'] > best_ndcg:
                        #     best_ndcg = valid_result_list[0]['bpref@20']
                            best_params = cur_params
                            best_results = test_result_list
                            best_pointwise = pointwise_test_list
                        folder = saved_model_file.split(".")[0]
                        os.remove(saved_model_file)
                        os.system("rm -rf {}".format(folder))
                elif tuning_param == "True_1diff":
                    def tuning_func(perms2):
                        best_ndcg_temp, best_params_temp, best_results_temp, best_pointwise_temp = 0.0, tuple(), list(), list()
                        for i in range(len(perms2)):
                            cur_params = {}
                            for j in range(len(perms2[i])):
                                cur_params[attr[j]] = perms2[i][j]
                            # config_dict.update(cur_params)
                            total_config.final_config_dict.update(cur_params)
                            valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, eval_length)
                            global new_logger
                            new_logger = logger
                            logger.info("Above is the result of parameter: {}".format(cur_params))
                            if valid_result_list is None and test_result_list is None and saved_model_file is None:
                                continue
                            if valid_result_list[0][20]['ndcg@20'] >= best_ndcg_temp:
                                best_ndcg_temp = valid_result_list[0][20]['ndcg@20']
                            # if valid_result_list[0]['bpref@20'] > best_ndcg:
                            #     best_ndcg = valid_result_list[0]['bpref@20']
                                best_params_temp = cur_params
                                best_results_temp = test_result_list
                                best_pointwise_temp = pointwise_test_list
                            folder = saved_model_file.split(".")[0]
                            os.remove(saved_model_file)
                            os.system("rm -rf {}".format(folder))
                        return best_ndcg_temp, best_params_temp, best_results_temp, best_pointwise_temp
                    
                    prev_best_dict = best_params_dict.cos_noadd_params_old[dataset_type][model_name][str(seed)]
                    learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                    beta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
                    tau = [1, 0.5, 0.1, 0.05, 0.01]
                    # learning_rate = [0.0001]
                    # beta = [0.005, 0.001]
                    # tau = [1]
                    perms2 = []
                    for i in range(len(learning_rate)):
                        perms2.append((learning_rate[i], prev_best_dict['beta'], prev_best_dict['tau']))
                    best_ndcg, best_params, best_results, best_pointwise = tuning_func(perms2)
                    perms2 = []
                    for i in range(len(beta)):
                        perms2.append((best_params['learning_rate'], beta[i], prev_best_dict['tau']))
                    best_ndcg, best_params, best_results, best_pointwise = tuning_func(perms2)
                    perms2 = []
                    for i in range(len(tau)):
                        perms2.append((best_params['learning_rate'], best_params['beta'], tau[i]))
                    best_ndcg, best_params, best_results, best_pointwise = tuning_func(perms2)
                    
                        
                elif tuning_param == "Simple":
                    if (not use_llm_embed) or (merge_embed != "cl-dot" and merge_embed != "cl-cos"):
                        raise TypeError
                    learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]
                    beta = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
                    tau = [1, 0.5, 0.1, 0.05, 0.01]
                    attr = ["learning_rate", "beta", "tau"]
                    perm = list(itertools.product(learning_rate, beta, tau))
                    if merge_embed == "cl-dot":
                        static_params2 = best_params_dict.dot_noadd_params[dataset_type][model_name][str(seed)]
                    elif merge_embed == "cl-cos":
                        static_params2 = best_params_dict.cos_noadd_params[dataset_type][model_name][str(seed)]
                    for i in range(len(perm) - 1, -1, -1):
                        count = 0
                        for j in range(len(perm[i])):
                            if perm[i][j] == static_params2[attr[j]]:
                                count += 1
                        if count < len(perm[i]) - 1:
                            perm.pop(i)
                    for i in range(len(perm)):
                        cur_params = {}
                        for j in range(len(perm[i])):
                            cur_params[attr[j]] = perm[i][j]
                        # config_dict.update(cur_params)
                        total_config.final_config_dict.update(cur_params)
                        valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, eval_length)
                        # global new_logger
                        new_logger = logger
                        logger.info("Above is the result of parameter: {}".format(cur_params))
                        if valid_result_list is None and test_result_list is None and saved_model_file is None:
                            continue
                        if valid_result_list[0][20]['ndcg@20'] > best_ndcg:
                            best_ndcg = valid_result_list[0][20]['ndcg@20']
                        # if valid_result_list[0]['bpref@20'] > best_ndcg:
                        #     best_ndcg = valid_result_list[0]['bpref@20']
                            best_params = cur_params
                            best_results = test_result_list
                            best_pointwise = pointwise_test_list
                        folder = saved_model_file.split(".")[0]
                        os.remove(saved_model_file)
                        os.system("rm -rf {}".format(folder))
                elif tuning_param == "Complement":
                    if use_llm_embed and merge_embed == "cl-dot":
                        static_params2 = best_params_dict.dot_noadd_params[dataset_type][model_name][str(seed)]
                    elif use_llm_embed and merge_embed == "cl-cos":
                        # cur_size = int(dataset_choices.split("_")[-1])
                        # cur_size = int(dataset_choices.split("_")[-2])
                        # paramchoice = {
                        #     1: best_params_dict.cos_noadd_params_1,
                        #     3: best_params_dict.cos_noadd_params_3,
                        #     5: best_params_dict.cos_noadd_params_5,
                        #     10: best_params_dict.cos_noadd_params_10,
                        # }
                        static_params2 = best_params_dict.cos_noadd_params_5[dataset_type][model_name][str(seed)]
                        # print(cur_size, type(cur_size))
                        # print(paramchoice[cur_size])
                        # print(paramchoice[cur_size].keys())
                        # time.sleep(1000)
                        # static_params2 = paramchoice[cur_size][dataset_type][model_name][str(seed)]
                    elif not use_llm_embed:
                        static_params2 = best_params_dict.basic_params[dataset_type][model_name][str(seed)]
                    total_config.final_config_dict.update(static_params2)
                    # config_dict.update(static_params2)
                    valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger, tuple_result = train_eval(dataset_type, model_name, config_dict, eval_length)
                    if tuple_result != None:
                        total_train_tuple.append(tuple_result)
                    new_logger = logger
                    best_params = static_params2
                    best_results = test_result_list
                    best_pointwise = pointwise_test_list
                    folder = saved_model_file.split(".")[0]
                    try:
                        os.remove(saved_model_file)
                    except:
                        pass
                    try:
                        os.system("rm -rf {}".format(folder))
                    except:
                        pass
                
                if tuning_param == "True_1diff":
                    logger = new_logger
                logger.info("Best results of seed {}:".format(seed))
                for i in range(len(best_results)):
                    if i == 0:
                        logger.info(set_color("total test result".format(i), "yellow") + f": {best_results[0]}")
                    else:
                        logger.info(set_color("test result of group {}".format(i), "yellow") + f": {best_results[i]}")
                    for key in best_results[i].keys():
                        total_ndcg[key][i] += best_results[i][key]['ndcg@{}'.format(key)]
                        total_mrr[key][i] += best_results[i][key]['mrr@{}'.format(key)]
                        total_bpref[key][i] += best_results[i][key]['bpref@{}'.format(key)]
                        # print(len(best_pointwise[i][key]['ndcg@{}'.format(key)]), len(best_pointwise[i][key]['mrr@{}'.format(key)]), len(best_pointwise[i][key]['bpref@{}'.format(key)]))
                        total_point_ndcg[key][i].append(best_pointwise[i][key]['ndcg@{}'.format(key)])
                        total_point_mrr[key][i].append(best_pointwise[i][key]['mrr@{}'.format(key)])
                        total_point_bpref[key][i].append(best_pointwise[i][key]['bpref@{}'.format(key)])
                logger.info("With parameters: {}".format(best_params))
                
                
            for i in range(len(eval_length)):
                total_ndcg[eval_length[i]] /= len(seed_list)
                total_mrr[eval_length[i]] /= len(seed_list)
                total_bpref[eval_length[i]] /= len(seed_list)
                # print(len(total_point_ndcg[eval_length[i]]))
                # for j in range(len(total_point_ndcg[eval_length[i]])):
                #     print(len(total_point_ndcg[eval_length[i]][j]))
                #     for k in range(len(total_point_ndcg[eval_length[i]][j])):
                #         print(len(total_point_ndcg[eval_length[i]][j][k]))
                
                for j in range(len(total_point_ndcg[eval_length[i]])):
                    try:
                        total_point_ndcg[eval_length[i]][j] = np.array(total_point_ndcg[eval_length[i]][j]).mean(axis=0)
                        total_point_mrr[eval_length[i]][j] = np.array(total_point_mrr[eval_length[i]][j]).mean(axis=0)
                        total_point_bpref[eval_length[i]][j] = np.array(total_point_bpref[eval_length[i]][j]).mean(axis=0)
                        # print(total_point_ndcg[eval_length[i]][j].shape)
                        # print(total_point_mrr[eval_length[i]][j].shape)
                        # print(total_point_bpref[eval_length[i]][j].shape)
                    except:
                        pass
                    
                # total_point_ndcg[eval_length[i]] = np.array(total_point_ndcg[eval_length[i]]).mean(axis=1)
                # total_point_mrr[eval_length[i]] = np.array(total_point_mrr[eval_length[i]]).mean(axis=1)
                # total_point_bpref[eval_length[i]] = np.array(total_point_bpref[eval_length[i]]).mean(axis=1)
                # print(total_point_ndcg[eval_length[i]].shape)
                # print(total_point_mrr[eval_length[i]].shape)
                # print(total_point_bpref[eval_length[i]].shape)
                
            for i in range(6):
                if i == 0:
                    logger.info(set_color("total average test results", "yellow"))
                else:
                    logger.info(set_color("average test result of group {}".format(i), "yellow"))
                for j in range(len(eval_length)):
                    logger.info("{{'ndcg@{}': {}, 'mrr@{}': {}, 'bpref@{}': {}}}".format(eval_length[j], total_ndcg[eval_length[j]][i], eval_length[j], total_mrr[eval_length[j]][i], eval_length[j], total_bpref[eval_length[j]][i]))
            
            for i in range(len(logger.handlers)):
                if isinstance(logger.handlers[i], FileHandler):
                    logger_path = logger.handlers[i].baseFilename
                    break
            folder_path = logger_path.split(".")[0]
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            with open("{}/ndcg.pkl".format(folder_path), "wb") as f:
                pickle.dump(total_point_ndcg, f)
            with open("{}/mrr.pkl".format(folder_path), "wb") as f:
                pickle.dump(total_point_mrr, f)
            with open("{}/bpref.pkl".format(folder_path), "wb") as f:
                pickle.dump(total_point_bpref, f)
            
            if tuning_param == 'Complement' and (not use_llm_embed):
                field2id_token_full = total_dataset_full.field2id_token
                total_train_scoredict = {}
                with tqdm(total=len(total_train_tuple) * len(total_train_tuple[0][0])) as pbar:
                    for tup in total_train_tuple:
                        for cur_userid, cur_itemid, cur_rating, cur_score in list(zip(tup[0], tup[1], tup[2], tup[3])):
                            cur_userid = field2id_token_full['user_id'][cur_userid]
                            cur_itemid = field2id_token_full['item_id'][cur_itemid]
                            if cur_userid not in total_train_scoredict.keys():
                                total_train_scoredict[cur_userid] = {}
                            if cur_itemid not in total_train_scoredict[cur_userid].keys():
                                total_train_scoredict[cur_userid][cur_itemid] = [cur_rating, cur_score]
                            else:
                                total_train_scoredict[cur_userid][cur_itemid][1] += cur_score
                            pbar.update()
                total_train_scoredict2 = {}
                with tqdm(total=len(total_train_scoredict.keys())) as pbar:
                    for cur_userid in total_train_scoredict.keys():
                        total_train_scoredict2[cur_userid] = []
                        for cur_itemid in total_train_scoredict[cur_userid].keys():
                            cur_rating = total_train_scoredict[cur_userid][cur_itemid][0]
                            ave_score = total_train_scoredict[cur_userid][cur_itemid][1] / len(total_train_tuple)
                            total_train_scoredict2[cur_userid].append([cur_itemid, cur_rating, ave_score])
                        total_train_scoredict2[cur_userid] = sorted(total_train_scoredict2[cur_userid], key=lambda x:x[2], reverse=True)
                        # total_train_scoredict2[cur_userid] = total_train_scoredict2[cur_userid][:10]
                        pbar.update()
                with open("{}/train_full.pkl".format(folder_path), "wb") as f:
                    pickle.dump(total_train_scoredict2, f)
                process_train_full(folder_path)
            # for i in range(6):
            #     if i == 0:
            #         logger.info(set_color("total average test results", "yellow") + ": ndcg: {}, mrr: {}, bpref: {}".format(total_ndcg[i], total_mrr[i], total_bpref[i]))
            #     else:
            #         logger.info(set_color("average test result of group {}".format(i), "yellow") + ": ndcg: {}, mrr: {}, bpref: {}".format(total_ndcg[i], total_mrr[i], total_bpref[i]))
        # elif tuning_param == 'Seed':
        #     if dataset_type == 'ml100k':
        #         if model_name == 'LightGCN':
        #             static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 19, 'tau': 0.05}
        #         elif model_name == 'WideDeep':
        #             static_params = {'learning_rate': 0.01, 'beta': 0.1, 'sample_num': 9, 'tau': 0.5}
        #         elif model_name == 'FM':
        #             static_params = {'learning_rate': 0.01, 'beta': 0.001, 'sample_num': 19, 'tau': 1}
        #         elif model_name == 'SimpleX':
        #             static_params = {'learning_rate': 0.005, 'beta': 0.001, 'sample_num': 9, 'tau': 0.5}
        #         elif model_name == 'NeuMF':
        #             static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 19, 'tau': 0.5}
        #         elif model_name == 'SpectralCF':
        #             static_params = {'learning_rate': 0.005, 'beta': 0.05, 'sample_num': 19, 'tau': 0.1}
        #         # if model_name == 'LightGCN':
        #         #     static_params = {'learning_rate': 0.01, 'beta': 0.1, 'sample_num': 19, 'tau': 0.1}
        #         # elif model_name == 'WideDeep':
        #         #     static_params = {'learning_rate': 0.0001, 'beta': 0.05, 'sample_num': 9, 'tau': 0.05}
        #         # # elif model_name == 'FM':
        #         # #     static_params = 
        #         # elif model_name == 'SimpleX':
        #         #     static_params = {'learning_rate': 0.001, 'beta': 1, 'sample_num': 9, 'tau': 0.1}
        #         # elif model_name == 'NeuMF':
        #         #     static_params = {'learning_rate': 0.0005, 'beta': 1, 'sample_num': 9, 'tau': 0.05}
        #         # elif model_name == 'SpectralCF':
        #         #     static_params = {'learning_rate': 0.005, 'beta': 0.005, 'sample_num': 9, 'tau': 1}
        #     elif dataset_type == 'amazon-CDs_and_Vinyl':
        #         if model_name == 'LightGCN':
        #             static_params = {'learning_rate': 0.0005, 'beta': 0.05, 'sample_num': 9, 'tau': 0.5}
        #         elif model_name == 'WideDeep':
        #             static_params = {'learning_rate': 0.01, 'beta': 0.5, 'sample_num': 19, 'tau': 0.5}
        #         elif model_name == 'FM':
        #             static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 9, 'tau': 1}
        #         elif model_name == 'SimpleX':
        #             static_params = {'learning_rate': 0.0005, 'beta': 0.001, 'sample_num': 9, 'tau': 0.5}
        #         elif model_name == 'NeuMF':
        #             static_params = {'learning_rate': 0.0005, 'beta': 1, 'sample_num': 9, 'tau': 0.1}
        #         elif model_name == 'SpectralCF':
        #             static_params = {'learning_rate': 0.005, 'beta': 1, 'sample_num': 9, 'tau': 0.5}
        #         # if model_name == 'LightGCN':
        #         #     static_params = {'learning_rate': 0.005, 'beta': 0.1, 'sample_num': 19, 'tau': 0.01}
        #         # elif model_name == 'WideDeep':
        #         #     static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 19, 'tau': 1}
        #         # # elif model_name == 'FM':
        #         # #     static_params = 
        #         # elif model_name == 'SimpleX':
        #         #     static_params = {'learning_rate': 0.0005, 'beta': 0.001, 'sample_num': 9, 'tau': 1}
        #         # elif model_name == 'NeuMF':
        #         #     static_params = {'learning_rate': 0.005, 'beta': 1, 'sample_num': 19, 'tau': 0.5}
        #         # elif model_name == 'SpectralCF':
        #         #     static_params = {'learning_rate': 0.001, 'beta': 1, 'sample_num': 9, 'tau': 0.01}
        #     elif dataset_type == 'amazon-Office_Products':
        #         if model_name == 'LightGCN':
        #             static_params = {'learning_rate': 0.0005, 'beta': 0.005, 'sample_num': 9, 'tau': 1}
        #         elif model_name == 'WideDeep':
        #             static_params = {'learning_rate': 0.005, 'beta': 0.01, 'sample_num': 19, 'tau': 0.1}
        #         elif model_name == 'FM':
        #             static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 19, 'tau': 0.05}
        #         elif model_name == 'SimpleX':
        #             static_params = {'learning_rate': 0.0001, 'beta': 0.05, 'sample_num': 9, 'tau': 1}
        #         elif model_name == 'NeuMF':
        #             static_params = {'learning_rate': 0.001, 'beta': 1, 'sample_num': 9, 'tau': 0.5}
        #         elif model_name == 'SpectralCF':
        #             static_params = {}
        #         # if model_name == 'LightGCN':
        #         #     static_params = {'learning_rate': 0.005, 'beta': 1, 'sample_num': 19, 'tau': 0.01}
        #         # elif model_name == 'WideDeep':
        #         #     static_params = {'learning_rate': 0.0005, 'beta': 0.5, 'sample_num': 19, 'tau': 0.5}
        #         # # elif model_name == 'FM':
        #         # #     static_params = 
        #         # elif model_name == 'SimpleX':
        #         #     static_params = {'learning_rate': 0.001, 'beta': 0.5, 'sample_num': 19, 'tau': 0.1}
        #         # elif model_name == 'NeuMF':
        #         #     static_params = {'learning_rate': 0.01, 'beta': 1, 'sample_num': 9, 'tau': 0.05}
        #         # elif model_name == 'SpectralCF':
        #         #     static_params = {'learning_rate': 0.001, 'beta': 1, 'sample_num': 19, 'tau': 0.01}
        #     config_dict.update(static_params)
        #     seed_list = [2020, 2021, 2022, 2023, 2024]
        #     total_ndcg, total_mrr, total_bpref = np.zeros(6), np.zeros(6), np.zeros(6)
        #     for seed in seed_list:
        #         temp = {'seed': seed}
        #         config_dict.update(temp)
        #         valid_result_list, test_result_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, use_llm_embed)
        #         new_logger = logger
        #         logger.info("Above is the result of seed: {}".format(seed))
        #         for i in range(6):
        #             total_ndcg[i] += test_result_list[i]['ndcg@20']
        #             total_mrr[i] += test_result_list[i]['mrr@20']
        #             total_bpref[i] += test_result_list[i]['bpref@20']
        #         folder = saved_model_file.split(".")[0]
        #         os.remove(saved_model_file)
        #         os.system("rm -rf {}".format(folder))
        #     total_ndcg /= 5
        #     total_mrr /= 5
        #     total_bpref /= 5
        #     for i in range(6):
        #         if i == 0:
        #             logger.info(set_color("total average test results", "yellow") + ": ndcg: {}, mrr: {}, bpref: {}".format(total_ndcg[i], total_mrr[i], total_bpref[i]))
        #         else:
        #             logger.info(set_color("average test result of group {}".format(i), "yellow") + ": ndcg: {}, mrr: {}, bpref: {}".format(total_ndcg[i], total_mrr[i], total_bpref[i]))
    else:
        valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, eval_length)

def setup_wandb(): # nonstandard
    # api_key = 85978ff2b285559fe8b12d59a1df5abfc53bd130
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'valid_ndcg'},
        'parameters': {
            'learning_rate': {'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]},
            'beta': {'values': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]},
            'sample_num': {'values': [9, 19]},
            'tau': {'values': [1, 0.5, 0.1, 0.05, 0.01]}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
    wandb.agent(sweep_id, function=run_wandb, count=50)
    sys.exit(1)

def run_wandb(): # nonstandard
    run = wandb.init()
    learning_rate = wandb.config.learning_rate
    beta = wandb.config.beta
    sample_num = wandb.config.sample_num
    tau = wandb.config.tau
    temp = {
        'learning_rate': learning_rate,
        'beta': beta,
        'sample_num': sample_num,
        'tau': tau
    }
    config_dict.update(temp)
    valid_result_list, test_result_list, pointwise_test_list, saved_model_file, logger = train_eval(dataset_type, model_name, config_dict, use_llm_embed)
    global new_logger
    new_logger = logger
    valid_ndcg = valid_result_list[0]['ndcg@20']
    wandb.log(
        {
            # 'valid_result_list': valid_result_list,
            # 'test_result_list': test_result_list
            'valid_ndcg': valid_result_list[0]['ndcg@20'],
            'valid_bpref': valid_result_list[0]['bpref@20'],
            'test_ndcg': test_result_list[0]['ndcg@20'],
            'test_bpref': test_result_list[0]['bpref@20']
        }
    )
    logger.info("Above is the result of parameter: {}".format(temp))
    folder = saved_model_file.split(".")[0]
    os.remove(saved_model_file)
    os.system("rm -rf {}".format(folder))

def testing(config_dict):
    if dataset_type == "ml100k":
        path = "./ml100k/user_200_3/recbole/user_200"
    elif "amazon" in dataset_type:
        subset = dataset_type.split("-")[1]
        path = "./amazon/user_5k/{}/recbole/user_5k".format(subset)
    elif dataset_type == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k"
    total_dataconfig = {
        'data_path': path,
        'dataset': 'user_200_total'
    }
    config_dict.update(total_dataconfig)
    total_config = Config(model=model_name, config_dict=config_dict)
    init_logger(total_config)
    logger = getLogger()
    logger.info(logger.name)
    time.sleep(1)

def quickload(dataset_type, config_dict):
    if dataset_type == "ml100k":
        path = "./ml100k/user_200_3/recbole/user_200"
    elif "amazon" in dataset_type:
        subset = dataset_type.split("-")[1]
        path = "./amazon/user_5k/{}/recbole/user_5k".format(subset)
    elif dataset_type == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k"
    
    # dataset_choice = dataset_choices
    config_dict_total = copy.deepcopy(config_dict)
    
    total_dataconfig = {
        'data_path': path,
        'dataset': dataset_choices,
        'benchmark_filename': ['train', 'valid', 'test'],
    }
    config_dict_total.update(total_dataconfig)
    total_config = Config(model=model_name, config_dict=config_dict_total)
    print("Creating dataset...")
    total_dataset = create_dataset(total_config)
    print("Complete!")
    print("Data preparation...")
    total_train_data, total_valid_data, total_test_data = data_preparation(total_config, total_dataset)
    
    if use_llm_embed == False and tuning_param == 'Complement':
        total_config.final_config_dict.update({'benchmark_filename': ['train_full', 'valid', 'test']})
        print("Creating full dataset...")
        total_dataset_full = create_dataset(total_config)
        print("Complete!")
        print("Full data preparation...")
        total_train_data_full = data_preparation_train(total_config, total_dataset_full)
        total_config.final_config_dict.update({'benchmark_filename': ['train', 'valid', 'test']})
    else:
        total_dataset_full, total_train_data_full = None, None
    
    global interaction_list
    interaction_list = []
    for batch_idx, interaction in enumerate(total_train_data):
        interaction = interaction.to(config_dict['device'])
        interaction_list.append(interaction)
    
    global evaluation_list, batched_data_list
    evaluation_list, batched_data_list = [], []
    for batch_idx, batched_data in enumerate(total_valid_data):
        interaction = batched_data[0]
        interaction = interaction.to(config_dict['device'])
        evaluation_list.append(interaction)
        batched_data_list.append(batched_data)
    
    # print(total_train_data.inter_feat)
    # print(total_valid_data.inter_feat)
    # print(total_test_data.inter_feat)
    print("Complete!")
    # time.sleep(1000)
    return total_train_data, total_valid_data, total_test_data, total_config, total_dataset, interaction_list, evaluation_list, batched_data_list, total_train_data_full, total_dataset_full

def rename():
    for i in range(6, 66, 6):
        path = "./ml-25m/user_5k/recbole/user_5k/user_5k_total_ml-25m_3_64_{}".format(i)
        for file in os.listdir(path):
            prefix = file.split(".", maxsplit=1)[0]
            suffix = file.split(prefix)[-1]
            prefix = prefix.rsplit("_", maxsplit=1)[0]
            prefix = prefix + "_{}".format(i)
            os.rename("{}/{}".format(path, file), "{}{}".format(prefix, suffix))
    sys.exit(1)

def direct_dot(dataset_type):
    if dataset_type == "amazon-CDs_and_Vinyl":
        path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_total_amazon_5_48_full"
        with open("./Embeddings/amazon/CDs_and_Vinyl/user_totalembed.pkl", "rb") as f:
            oriembed_list = pickle.load(f)
        with open("./amazon/user_5k/CDs_and_Vinyl/u.user", "r") as f:
            lines = f.read().splitlines()
        userid_list = []
        for i in range(len(lines)):
            user_id, _ = lines[i].split("|")
            userid_list.append(user_id)
        zipped = list(zip(userid_list, oriembed_list))
        user_oriembed_dict = {x[0]:x[1] for x in zipped}
    else:
        path = "./ml-25m/user_5k/recbole/user_5k/user_5k_total_ml-25m_5_64_full"
        with open("./Embeddings/ml-25m/user_totalembed.pkl", "rb") as f:
            oriembed_list = pickle.load(f)
        with open("./ml-25m/user_5k/u.user", "r") as f:
            lines = f.read().splitlines()
        userid_list = []
        for i in range(len(lines)):
            user_id, _ = lines[i].split("|")
            userid_list.append(user_id)
        zipped = list(zip(userid_list, oriembed_list))
        user_oriembed_dict = {x[0]:x[1] for x in zipped}
    userembed_dict, itemembed_dict = {}, {}
    inter_dict = {}
    
    
    for file in os.listdir(path):
        if file.endswith("useremb"):
            # with open("{}/{}".format(path, file), "r") as f:
            #     lines = f.read().splitlines()
            # with tqdm(total=len(lines) - 1) as pbar:
            #     for i in range(1, len(lines)):
            #         cur_id, cur_embed = lines[i].split("|")
            #         cur_embed = "[" + cur_embed + "]"
            #         cur_embed = eval(cur_embed)
            #         userembed_dict[cur_id] = cur_embed
            #         pbar.update()
            userembed_dict = copy.deepcopy(user_oriembed_dict)
        elif file.endswith("itememb"):
            with open("{}/{}".format(path, file), "r") as f:
                lines = f.read().splitlines()
            with tqdm(total=len(lines) - 1) as pbar:
                for i in range(1, len(lines)):
                    cur_id, cur_embed = lines[i].split("|")
                    cur_embed = "[" + cur_embed + "]"
                    cur_embed = eval(cur_embed)
                    itemembed_dict[cur_id] = cur_embed
                    pbar.update()
        elif file.endswith("test.inter"):
            with open("{}/{}".format(path, file), "r") as f:
                lines = f.read().splitlines()
            with tqdm(total=len(lines) - 1) as pbar:
                for i in range(1, len(lines)):
                    user_id, item_id, rating, _ = lines[i].split("|")
                    rating = float(rating)
                    if rating >= 4.0:
                        rating = 1.0
                    elif 0 < rating <= 3.5:
                        rating = -1.0
                    user_embed, item_embed = userembed_dict[user_id], itemembed_dict[item_id]
                    dot = np.dot(np.array(user_embed), np.array(item_embed))
                    if user_id not in inter_dict.keys():
                        inter_dict[user_id] = []
                    inter_dict[user_id].append([item_id, rating, dot])

    random.seed(2024)
    eval_length = [1, 5, 10, 20]
    result_dict = {}
    
        # for i in range(len(eval_length)):
        # total_mrr, total_ndcg = 0.0, 0.0
        # # totaldiff_mrr, totaldiff_rank = 0.0, 0.0
        # total_bpref = 0.0
        # ndcg_list, mrr_list, bpref_list = [], [], []
        
        # for key in inter_sortdict.keys():
        #     interaction = inter_sortdict[key][:eval_length[i]]
        #     # cur_ndcg, cur_mrr = ndcg_naive(interaction), mrr(interaction)
        #     # neg_mrr, pos_rank, neg_rank = diff(interaction)
        #     cur_ndcg, cur_mrr = get_ndcg_mrr(interaction)
        #     cur_bpref = get_bpref(interaction)
    # {'ndcg@{}'.format(eval_length[i]):total_ndcg, 'mrr@{}'.format(eval_length[i]):total_mrr, 'bpref@{}'.format(eval_length[i]):total_bpref}
    
    for key in inter_dict.keys():
        random.shuffle(inter_dict[key])
        inter_dict[key] = sorted(inter_dict[key], key=lambda x:x[2], reverse=True)
        for length in eval_length:
            for metric in ['ndcg', 'mrr', 'bpref']:
                if "{}@{}".format(metric, length) not in result_dict.keys():
                    result_dict["{}@{}".format(metric, length)] = 0.0
            interaction = inter_dict[key][:length]
            cur_ndcg, cur_mrr = get_ndcg_mrr(interaction)
            cur_bpref = get_bpref(interaction)
            result_dict["ndcg@{}".format(length)] += cur_ndcg
            result_dict["mrr@{}".format(length)] += cur_mrr
            result_dict["bpref@{}".format(length)] += cur_bpref
    for metric in ['ndcg', 'mrr', 'bpref']:
        for length in eval_length:
            result_dict["{}@{}".format(metric, length)] /= len(inter_dict)
    print(result_dict)
    sys.exit(1)

def norm_embed():
    file = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_total_amazon_5_48_full/user_5k_total_amazon_5_48_full.itememb"
    with open(file, "r") as f:
        lines = f.read().splitlines()
    with open("{}_new".format(file), "w") as f:
        f.write("{}\n".format(lines[0]))
        # with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            cur_id, cur_embed = lines[i].split("|")
            cur_embed = "[" + cur_embed + "]"
            cur_embed = np.array(eval(cur_embed))
            length = np.linalg.norm(cur_embed)
            cur_embed = (cur_embed / length).tolist()
            cur_embed = str(cur_embed)[1:-1]
            f.write("{}|{}\n".format(cur_id, cur_embed))
            if i % 500 == 0:
                print("{}/{}".format(i, len(lines) - 1))
                # pbar.update()
    sys.exit(1)

def regenerate_dataset(dataset_type):
    if dataset_type == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k"
        folder_list = ["user_5k_total"]
    elif dataset_type == "amazon-CDs_and_Vinyl":
        path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k"
        folder_list = ["user_5k_total"]
    for folder in folder_list:
        with open("{}/{}/{}.train.inter".format(path, folder, folder), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        user_inter_dict, user_item_dict = dict(), dict()
        item_set = set()
        with tqdm(total=len(lines) - 1) as pbar:
            for i in range(1, len(lines)):
                user_id, item_id, rating, _ = lines[i].split("|")
                rating = float(rating)
                if user_id not in user_inter_dict.keys():
                    user_inter_dict[user_id] = []
                if user_id not in user_item_dict.keys():
                    user_item_dict[user_id] = set()
                user_inter_dict[user_id].append([item_id, rating])
                user_item_dict[user_id].add(item_id)
                item_set.add(item_id)
                pbar.update()
        with tqdm(total=len(user_inter_dict.keys())) as pbar:
            for key in user_inter_dict.keys():
                temp1, temp2 = [], []
                for item in user_inter_dict[key]:
                    if item[1] >= 4:
                        temp1.append(item)
                    else:
                        temp2.append(item)
                new_inter_list = temp1 + temp2
                not_interacted = list(item_set - user_item_dict[key])
                for i in range(len(not_interacted)):
                    new_inter_list.append([not_interacted[i], 0.0])
                user_inter_dict[key] = new_inter_list
                pbar.update()
        with open("{}/{}/{}.train_full.inter".format(path, folder, folder), "w", encoding="latin-1") as f:
            f.write("{}\n".format(lines[0]))
            with tqdm(total=len(user_inter_dict.keys())) as pbar:
                for key in user_inter_dict.keys():
                    for item in user_inter_dict[key]:
                        f.write("{}|{}|{}|0\n".format(key, item[0], item[1]))
                    pbar.update()
    sys.exit(1)

def process_train_full(folder_path):
    print("Saving top item infomation...")
    path = "./ml-25m/user_5k/recbole/user_5k/{}".format(dataset_choices)
    with open("{}/train_full.pkl".format(folder_path), "rb") as f:
        total_train_scoredict = pickle.load(f)
    
    portion_dict = {'Seen': 10, 'Unseen': 10}
    total_train_scoredict2 = {}
    with tqdm(total=len(total_train_scoredict.keys())) as pbar:
        for cur_userid in total_train_scoredict.keys():
            seen_list, unseen_list = [], []
            for i in range(len(total_train_scoredict[cur_userid])):
                _, cur_rating, _ = total_train_scoredict[cur_userid][i]
                if cur_rating >= 4.0 and len(seen_list) < portion_dict['Seen']:
                    seen_list.append(total_train_scoredict[cur_userid][i])
                if cur_rating == 0.0 and len(unseen_list) < portion_dict['Unseen']:
                    unseen_list.append(total_train_scoredict[cur_userid][i])
                if len(seen_list) == portion_dict['Seen'] and len(unseen_list) == portion_dict['Unseen']:
                    break
            total_train_scoredict2 = seen_list + unseen_list
            pbar.update()
            
    with open("{}/{}.item".format(path, dataset_choices), "r") as f:
        lines = f.read().splitlines()
    item_infodict = {}
    for i in range(1, len(lines)):
        cur_itemid, cur_title, cur_category, cur_description = lines[i].split("|")
        # if dataset_type == "ml-25m":
        #     cur_itemid = int(cur_itemid)
        item_infodict[cur_itemid] = "Title: {}. Category: {}. Description: {}".format(cur_title, cur_category, cur_description)
    with open("{}/{}.fulliteminfo_{}".format(path, dataset_choices, model_name), "w") as f:
        f.write("user_id:token|item_id:token|item_info:token_seq|rating:float|score:float\n")
        with tqdm(total=len(total_train_scoredict2.keys())) as pbar:
            for cur_userid in total_train_scoredict2.keys():
                for i in range(len(total_train_scoredict2[cur_userid])):
                    cur_itemid, cur_rating, cur_score = total_train_scoredict2[cur_userid][i]
                    # print(cur_itemid, cur_rating, cur_score)
                    f.write("{}|{}|{}|{}|{}\n".format(cur_userid, cur_itemid, cur_rating, cur_score, item_infodict[cur_itemid]))
                pbar.update()

# def test():
#     with open("./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_total/user_5k_total.train_full.inter", "r") as f:
#         lines = f.read().splitlines()
#     print(len(lines))
#     sys.exit(1)

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="amazon-CDs_and_Vinyl", help="dataset type")
    parser.add_argument("--dataset_choice", type=str, default="user_200_total_16", help="dataset choice")
    parser.add_argument("--model", type=str, default="SimpleX", help="model name")
    parser.add_argument("--gpu_id", type=str, default='7', help="gpu id")
    parser.add_argument("--tuning_param", type=str, default='Complement', help="tuning parameter")
    args = parser.parse_args()
    return args


# nohup python test2.py &> office.log
if __name__ == "__main__":
    # test()
    # norm_embed()
    # rename()
    # dataset_type = 'ml100k'
    # dataset_type = "amazon-CDs_and_Vinyl"
    # dataset_type = "amazon-Office_Products"
    # dataset_type = "ml-25m"
    # dataset_type = sys.argv[1]
    # direct_dot(dataset_type)
    
    # model_name = 'LightGCN'
    # model_name = 'FM'
    # model_name = 'ItemKNN'
    # model_name = 'WideDeep'
    # model_name = sys.argv[2]
    # gen_ablation_dataset("amazon-CDs_and_Vinyl")
    # gen_explore_dataset("ml100k")
    # gen_explore_dataset("amazon-CDs_and_Vinyl")
    
    # prepare_dataset(dataset_type)
    # prepare_embedding(dataset_type)
    # sys.exit(1)
    # regenerate_dataset(dataset_type)
    
    # for dataset_type in ['ml100k', 'amazon-CDs_and_Vinyl', 'amazon-Office_Products']:
    args = parse_args()
    datasets = [args.dataset_type]
    dataset_choice = [args.dataset_choice]
    gpu_id = args.gpu_id
    tuning_param = args.tuning_param

    
    
    for dataset_type in datasets:
    # dataset_type = 'amazon-CDs_and_Vinyl'
    # dataset_type = 'ml100k'
    # for dataset_choices in ['user_200_total_16', 'user_200_total_32', 'user_200_total_48', 'user_200_total_99']:
    # for dataset_choices in ['user_200_total_32']:
    # dataset_choice = ['user_5k_total_amazon_5_48_group5_{}'.format(x) for x in range(5, 80, 5)]
    # dataset_choice.append('user_5k_total_amazon_5_48_group5_final')
        # dataset_choice = ['user_5k_total_amazon_5_64']
        # if dataset_type == 'amazon-CDs_and_Vinyl':
        #     dataset_choice = ['user_5k_total_amazon_5_32_full_replicate']
        # else:
        #     # dataset_choice = ['user_5k_total_ml-25m_5_32_newprompts']
        #     dataset_choice = ['user_5k_total_JINA_new']
    # dataset_choice = ['user_5k_total']
    
    # dataset_choice = ['user_5k_total_amazon_1_32_full']
    # dataset_choice = ['user_5k_total_ml-25m_1_32_full', 'user_5k_total_ml-25m_3_32_full', 'user_5k_total_ml-25m_5_32_full', 'user_5k_total_ml-25m_10_32_full']
    
    # dataset_choice = ['user_5k_total_ml-25m_5_64_full']
    # dataset_choice = ['user_5k_total_ml-25m_10']
    # dataset_choice = ['user_5k_total_amazon_1', 'user_5k_total_amazon_3', 'user_5k_total_amazon_5']
    
    # dataset_choice = ['user_5k_total_amazon_3_16', 'user_5k_total_amazon_3_48', 'user_5k_total_amazon_3_64', 'user_5k_total_amazon_5_16', 'user_5k_total_amazon_5_40', 'user_5k_total_amazon_5_48', 'user_5k_total_amazon_5_64']
    # dataset_choice = ['user_5k_total_amazon_5_16', 'user_5k_total_amazon_5_40', 'user_5k_total_amazon_5_48', 'user_5k_total_amazon_5_64']
    # dataset_choice = ['user_5k_total_amazon_3_16', 'user_5k_total_amazon_3_48', 'user_5k_total_amazon_3_64']
    
    # dataset_choice = ['user_5k_total_ml-25m_3_16', 'user_5k_total_ml-25m_3_48', 'user_5k_total_ml-25m_3_64']
    # dataset_choice = ['user_5k_total_ml-25m_5_16', 'user_5k_total_ml-25m_5_48', 'user_5k_total_ml-25m_5_64']
    # dataset_choice = ['user_5k_total_ml-25m_3_16', 'user_5k_total_ml-25m_3_48', 'user_5k_total_ml-25m_3_64', 'user_5k_total_ml-25m_5_16', 'user_5k_total_ml-25m_5_40', 'user_5k_total_ml-25m_5_48', 'user_5k_total_ml-25m_5_64']
    
        for dataset_choices in dataset_choice:
            # folder_path = "./log/FM/FM-user_5k_total-Oct-27-2024_18-27-43-383080"
            # process_train_full(folder_path)
            # sys.exit(1)
            # for model_name in ['LightGCN', 'FM', 'SimGCL']:
            # for model_name in ['SimpleX', 'WideDeep', 'XSimGCL']:
            # for model_name in ['LightGCN', 'SimpleX', 'NeuMF', 'FM']:
            # for model_name in ['SGL', 'XSimGCL', 'LightGCL']:
            
            # for model_name in ['XSimGCL', 'LightGCN', 'SimpleX']:
            # for model_name in ['FM', 'NeuMF']:
            
            # for model_name in ['LightGCN', 'XSimGCL']:
            # for model_name in ['FM', 'LightGCL']:
            # for model_name in ['LightGCN', 'SimpleX', 'FM', 'LightGCL', 'XSimGCL']:
            for model_name in [args.model]:
            # for model_name in ['LightGCN', 'XSimGCL', 'SimpleX']:
            # for model_name in ['LightGCL']:
        
        # Please remember to modify the source code of the model after changing this parameter.
                use_llm_embed = True
                if use_llm_embed:
                    merge_embed = 'cl-cos' # 'none' / 'add' / 'cl-dot' / 'cl-cos' / 'cl-gen' / TODO
                    # merge_embed = sys.argv[3]
                    # sample_num = 19
                    beta = 0.1
                    tau = 0.1
                    mask_ratio = 0.1
                # if sys.argv[3] == "False":
                #     use_llm_embed = False
                # else:
                #     use_llm_embed = True
                
                # if sys.argv[4] == "False":
                #     tuning_param = False
                # else:
                #     tuning_param = True
                
                # tuning_param = 'True'
                # tuning_param = 'Complement' # 'False' / 'True' / 'Wandb' / 'Complement' / 'Simple'
                
                # print(dataset_type, model_name, use_llm_embed, tuning_param)
                # sys.exit(1)
                
                # used default values are also listed here.
                config_dict = {
                    'dataset_type': dataset_type,
                    'USER_ID_FIELD': 'user_id',
                    'ITEM_ID_FIELD': 'item_id',
                    'RATING_FIELD': 'rating',
                    'TIME_FIELD': 'timestamp',
                    'field_separator': '|',
                    'seq_separator': ', ',
                    'seed': 2028,
                    'reproducibility': True,
                    'device': 'cuda',
                    'encoding': 'latin-1',
                    'benchmark_filename': ['train', 'valid', 'test'],
                    'shuffle': False,
                    'checkpoint_dir': 'saved/',
                    'LABEL_FIELD': 'rating',
                    'threshold': {'rating': 4},
                    'epochs': 1000,
                    'train_batch_size': 2048,
                    'learning_rate': 0.001,
                    # 'train_neg_sample_args': {
                    #     'distribution': 'uniform',
                    #     'sample_num': 1,
                    #     'dynamic': False
                    # },
                    'train_neg_sample_args': None,
                    'eval_step': 1,
                    # 'eval_args': {
                    #     'group_by': 'user',
                    #     'order': 'RO',
                    #     'mode': 'uni19'
                    # },
                    'eval_args': {
                        'group_by': 'user',
                        'mode': 'labeled'
                    },
                    #'metrics': ['MRR', 'NDCG'],
                    'metrics': ['AUC'],
                    'topk': 10,
                    # 'valid_metric': 'NDCG@20',
                    'valid_metric': 'AUC',
                    'eval_batch_size': '4096',
                    'embedding_size': 64,
                    'n_layers': 1,
                    'reg_weight': 1e-05,
                    'use_llm_embed': use_llm_embed,
                    'dropout_prob': 0.0,
                    'mlp_hidden_size': '[64,32,16]',
                    'show_progress': True,
                    'stopping_step': 10,
                    'gpu_id': gpu_id,
                    'history_len': 50,
                    'margin': 0.0,
                    'gamma': 0.3,
                    'negative_weight': 50,
                    'lambda': 1,
                    'eps': 1,
                    'temperature': 1,
                    'layer_cl': 1,
                    'dropout': 0.1,
                    'temp': 1,
                    'lambda1': 1,
                    'lambda2': 1,
                    'q': 1
                }
                if use_llm_embed:
                    temp = {
                        'merge_embed': merge_embed,
                        # 'sample_num': sample_num,
                        'beta': beta,
                        'tau': tau,
                        'mask_ratio': mask_ratio,
                        'additional_feat_suffix': ['useremb', 'itememb'],
                        'alias_of_user_id': ['uid'],
                        'alias_of_item_id': ['iid'],
                        'preload_weight': {
                            'uid': 'user_emb',
                            'iid': 'item_emb'
                        },
                    }
                    config_dict.update(temp)
                if dataset_type == "ml100k":
                    if use_llm_embed:
                        temp = {
                            'load_col': {
                                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                'user': ['user_id', 'age', 'gender', 'occupation'],
                                'item': ['item_id', 'movie_title', 'genre', 'release_year'],
                                # 'user': ['user_id'],
                                # 'item': ['item_id'],
                                'useremb': ['uid', 'user_emb'],
                                'itememb': ['iid', 'item_emb']
                            }
                        }
                    else:
                        temp = {
                            'load_col': {
                                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                'user': ['user_id', 'age', 'gender', 'occupation'],
                                'item': ['item_id', 'movie_title', 'genre', 'release_year'],
                                # 'user': ['user_id'],
                                # 'item': ['item_id'],
                                # 'useremb': ['uid', 'user_emb'],
                                # 'itememb': ['iid', 'item_emb']
                            }
                        }
                    config_dict.update(temp)
                elif "amazon" in dataset_type or dataset_type == "ml-25m":
                    if use_llm_embed:
                        temp = {
                            'load_col': {
                                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                'user': ['user_id'],
                                'item': ['item_id', 'item_title', 'category'],
                                # 'item': ['item_id'],
                                'useremb': ['uid', 'user_emb'],
                                'itememb': ['iid', 'item_emb']
                            }
                        }
                    else:
                        temp = {
                            'load_col': {
                                'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                'user': ['user_id'],
                                'item': ['item_id', 'item_title', 'category'],
                                # 'item': ['item_id'],
                                # 'useremb': ['uid', 'user_emb'],
                                # 'itememb': ['iid', 'item_emb']
                            }
                        }
                    config_dict.update(temp)
                # train_eval(dataset_type, model_name, config_dict, use_llm_embed)
                total_train_data, total_valid_data, total_test_data, total_config, total_dataset, total_dataset_full = None, None, None, None, None, None
                interaction_list, batched_data_list, evaluation_list = None, None, None
                total_train_data_full = None
                
                param_tuning(dataset_type, model_name, config_dict, use_llm_embed, tuning_param)
                # testing(config_dict)


"""TODO:
1. 把llm baseline中的gene loss整合到原来的模型当中
2. 看一下添加1e-8之后，原来的loss is nan的参数是否能有值——之前的loss函数是否有问题
3. loss函数中，是否需要添加激活函数（leaky relu），及其区别，在一个参数上先看一下"""

# ml25m: {'ndcg@1': 0.4374, 'mrr@1': 0.4374, 'bpref@1': 0.0, 'ndcg@5': 0.7278527875541785, 'mrr@5': 0.4585511111111072, 'bpref@5': 0.4014999999999999, 'ndcg@10': 0.7270528268388466, 'mrr@10': 0.3926482936507912, 'bpref@10': 0.47639444444444434, 'ndcg@20': 0.7182903104023717, 'mrr@20': 0.36226189869727604, 'bpref@20': 0.5123944444444433}
# amazon: {'ndcg@1': 0.2506, 'mrr@1': 0.2506, 'bpref@1': 0.0, 'ndcg@5': 0.5043554606759427, 'mrr@5': 0.36786111111111014, 'bpref@5': 0.2391166666666667, 'ndcg@10': 0.560857199887598, 'mrr@10': 0.3223050661375664, 'bpref@10': 0.3858111111111112, 'ndcg@20': 0.5631639481374903, 'mrr@20': 0.2355044432522536, 'bpref@20': 0.49973333333333364}


# ml25m: {'ndcg@1': 0.3616, 'mrr@1': 0.3616, 'bpref@1': 0.0, 'ndcg@5': 0.6456401100989061, 'mrr@5': 0.45302277777777555, 'bpref@5': 0.3173666666666668, 'ndcg@10': 0.6571005557613206, 'mrr@10': 0.377104431216931, 'bpref@10': 0.4073444444444439, 'ndcg@20': 0.6421626677780118, 'mrr@20': 0.30120974475358137, 'bpref@20': 0.4820888888888874}
# amazon: 