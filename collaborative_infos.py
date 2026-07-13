import pickle
import shutil
import os
# from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import time
import numpy as np
import sys
import ollama
import threading
from prompt import *
import traceback
import itertools
import re
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def load_totalembed(dataset):
    if dataset == "ml-25m":
        embedpath = "/*********/*****/**********/***/Embeddings/ml-25m"
        datapath = "/*********/*****/**********/***/ml-25m/user_5k/recbole/user_5k/"
        folder = "user_5k_total_ml-25m_5_32_full"
    elif dataset == "amazon-CDs_and_Vinyl":
        embedpath = "/*********/*****/**********/***/Embeddings/amazon/CDs_and_Vinyl"
        datapath = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/"
        folder = "user_5k_total_amazon_5_32_full"
    with open("{}/user_totalembed.pkl".format(embedpath), "rb") as f:
        totalembed = pickle.load(f)
    embed_str = ""
    for i in range(len(totalembed[0])):
        embed_str += str(totalembed[0][i])
        embed_str += ", "
    
    if not os.path.exists("{}/user_5k_total_baseprofile".format(datapath)):
        os.mkdir("{}/user_5k_total_baseprofile".format(datapath))
    for postfix in ["user", "item", "train.inter", "valid.inter", "test.inter", "itememb"]:
        shutil.copy(
            "{}/{}/{}.{}".format(datapath, folder, folder, postfix),
            "{}/user_5k_total_baseprofile/user_5k_total_baseprofile.{}".format(datapath, postfix)
        )
    
    with open("{}/{}/{}.user".format(datapath, folder, folder), "r") as f:
        lines = f.read().splitlines()
    userid_list = []
    for i in range(1, len(lines)):
        cur_userid, _ = lines[i].split("|")
        userid_list.append(cur_userid)
    with open("{}/user_5k_total_baseprofile/user_5k_total_baseprofile.useremb".format(datapath), "w") as f:
        f.write("uid:token|user_emb:float_seq\n")
        for i in range(len(userid_list)):
            f.write("{}|{}\n".format(userid_list[i], embed_str))
    # for i in range(10):
    #     print(totalembed[i][:10])

def generate_BGE_user_embed(dataset):
    if dataset == "ml-25m":
        # path = "/*********/*****/**********/***/ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full/checkpoints"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "/*********/*****/**********/***/amazon_results/amazon_5_48_full/checkpoints"
        path = "/*********/*****/**********/***/amazon_results/amazon_5_32_full/checkpoints"
    user_id_list, sentence_list = [], []
    
    with tqdm(total=5000, desc="Loading user profiles...") as pbar:
        for files in os.listdir(path):
            if files.endswith("final_profile.pkl"):
                user_id = files.split("_")[1]
                if dataset == "ml-25m":
                    user_id = str(int(float(user_id)))
                user_id_list.append(user_id)
                with open("{}/{}".format(path, files), "rb") as f:
                    profile_list = pickle.load(f)
                sentence = ""
                for i in range(len(profile_list)):
                    sentence += profile_list[i]
                    if i < len(profile_list) - 1:
                        sentence += ", "
                    else:
                        sentence += "."
                sentence_list.append(sentence)
                pbar.update()
    
    model = BGEM3FlagModel('/*********/*****/**********/BGE', use_fp16=True)
    embedding_list = model.encode(sentence_list, batch_size=12, max_length=8192)['dense_vecs']
    print(embedding_list.shape)
    with tqdm(total=len(embedding_list), desc="Saving embeddings...") as pbar:
        with open("./BGE_embeds_{}".format(dataset), "w") as f:
            f.write("uid:token|user_emb:float_seq\n")
            for i in range(len(user_id_list)):
                f.write("{}|".format(user_id_list[i]))
                for j in range(len(embedding_list[i])):
                    f.write(str(embedding_list[i][j]))
                    f.write(", ")
                f.write("\n")
                pbar.update()

def generate_BGE_item_embed(dataset):
    if dataset == "ml-25m":
        # path = "/*********/*****/**********/***/ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full/checkpoints"
        item_path = "/*********/*****/**********/***/ml-25m/user_5k/u.item"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "/*********/*****/**********/***/amazon_results/amazon_5_48_full/checkpoints"
        path = "/*********/*****/**********/***/amazon_results/amazon_5_32_full/checkpoints"
        item_path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/u.item"
    item_id_list, sentence_list = [], []
    with open(item_path, "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines), desc="Processing item profiles...") as pbar:
        for i in range(len(lines)):
            item_id, item_title, item_category, item_description = lines[i].split("|")
            item_id_list.append(item_id)
            sentence = item_title + ". " + item_category + item_description
            sentence_list.append(sentence)
            pbar.update()
    
    model = BGEM3FlagModel('/*********/*****/**********/BGE', use_fp16=True)
    embedding_list = model.encode(sentence_list, batch_size=12, max_length=8192)['dense_vecs']
    print(embedding_list.shape)
    with tqdm(total=len(embedding_list), desc="Saving embeddings...") as pbar:
        with open("./BGE_embeds_{}_new".format(dataset), "w") as f:
            f.write("iid:token|item_emb:float_seq\n")
            for i in range(len(item_id_list)):
                f.write("{}|".format(item_id_list[i]))
                for j in range(len(embedding_list[i])):
                    f.write(str(embedding_list[i][j]))
                    f.write(", ")
                f.write("\n")
                pbar.update()

def generate_BGE_recbole(dataset):
    if dataset == "ml-25m":
        path = "/*********/*****/**********/***/ml-25m/user_5k/recbole/user_5k/user_5k_total_baseprofile"
        new_folder = "user_5k_total_JINA_ori"
        new_path = "/*********/*****/**********/***/ml-25m/user_5k/recbole/user_5k/{}".format(new_folder)
    elif dataset == "amazon-CDs_and_Vinyl":
        path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_total_baseprofile"
        new_folder = "user_5k_total_JINA_new"
        new_path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/{}".format(new_folder)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for postfix in ["user", "item", "train.inter", "valid.inter", "test.inter"]:
        shutil.copy(
            "{}/user_5k_total_baseprofile.{}".format(path, postfix),
            "{}/{}.{}".format(new_path, new_folder, postfix)
        )

def test():
    path = "/*********/*****/**********/***/all_profile_ml-25m.pkl"
    with open(path, "rb") as f:
        cur_dict = pickle.load(f)
    print(len(cur_dict.keys()))
    print(type(cur_dict['1']))
    print(cur_dict['1'])
    sys.exit(1)

def saving_user_totalprofiles(dataset):
    if dataset == "ml-25m":
        # path = "/*********/*****/**********/***/ml-25m_results/ml-25m_3_64_full/checkpoints"
        path = "/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full/checkpoints"
    elif dataset == "amazon-CDs_and_Vinyl":
        # path = "/*********/*****/**********/***/amazon_results/amazon_5_48_full/checkpoints"
        path = "/*********/*****/**********/***/amazon_results/amazon_5_32_full/checkpoints"
    
    user_id_list, sentence_list = [], []
    with tqdm(total=5000, desc="Loading user profiles...") as pbar:
        for files in os.listdir(path):
            if files.endswith("final_profile.pkl"):
                user_id = files.split("_")[1]
                if dataset == "ml-25m":
                    user_id = str(int(float(user_id)))
                user_id_list.append(user_id)
                with open("{}/{}".format(path, files), "rb") as f:
                    profile_list = pickle.load(f)
                sentence = ""
                for i in range(len(profile_list)):
                    sentence += profile_list[i]
                    if i < len(profile_list) - 1:
                        sentence += ", "
                    else:
                        sentence += "."
                sentence_list.append(sentence)
                pbar.update()
    
    with open("./{}_userprofiles_new".format(dataset), "w") as f:
        with tqdm(total=len(user_id_list), desc="Saving user profiles...") as pbar:
            for i in range(len(user_id_list)):
                f.write("{}|{}\n".format(user_id_list[i], sentence_list[i]))
                pbar.update()

def resample_standard(dataset):
    if dataset == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k"
        path2 = "./ml-25m_new2/user_5k/recbole/user_5k"
        default = "I enjoy watching movies very much."
    elif dataset == "amazon":
        path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k"
        path2 = "./amazon_new2/user_5k/CDs_and_Vinyl/recbole/user_5k"
        default = "I enjoy listening to music very much."
    positive_dict, train_dict, intered_dict = {}, {}, {}
    with open("{}/user_5k_total/user_5k_total.train.inter".format(path), "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            cur_userid, cur_itemid, cur_rating, cur_timestamp = lines[i].split("|")
            cur_timestamp = int(cur_timestamp)
            if cur_userid not in positive_dict.keys():
                positive_dict[cur_userid] = {}
            if cur_userid not in train_dict.keys():
                train_dict[cur_userid] = []
            if cur_userid not in intered_dict.keys():
                intered_dict[cur_userid] = set()
            if float(cur_rating) >= 4.0:
                if cur_itemid not in positive_dict[cur_userid].keys():
                    positive_dict[cur_userid][cur_itemid] = [cur_itemid, cur_rating, cur_timestamp]
                    intered_dict[cur_userid].add(cur_itemid)
            else:
                train_dict[cur_userid].append([cur_itemid, cur_rating, cur_timestamp])
                intered_dict[cur_userid].add(cur_itemid)
            pbar.update()
    
    banned_positive, banned_notseen = {}, {}
    with open("{}/user_5k_total/user_5k_total.valid.inter".format(path), "r") as f:
        lines = f.read().splitlines()
    with open("{}/user_5k_total/user_5k_total.test.inter".format(path), "r") as f:
        lines2 = f.read().splitlines()
    lines += lines2[1:]
    with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            cur_userid, cur_itemid, cur_rating, cur_timestamp = lines[i].split("|")
            cur_timestamp = int(cur_timestamp)
            if cur_userid not in banned_positive.keys():
                banned_positive[cur_userid] = set()
            if cur_userid not in banned_notseen.keys():
                banned_notseen[cur_userid] = set()
            if float(cur_rating) >= 4.0:
                positive_dict[cur_userid][cur_itemid] = [cur_itemid, cur_rating, cur_timestamp]
                banned_positive[cur_userid].add(cur_itemid)
                intered_dict[cur_userid].add(cur_itemid)
            elif float(cur_rating) > 0:
                train_dict[cur_userid].append([cur_itemid, cur_rating, cur_timestamp])
                intered_dict[cur_userid].add(cur_itemid)
            else:
                banned_notseen[cur_userid].add(cur_userid)
            pbar.update()
    
    item_set = set()
    with open("{}/user_5k_total/user_5k_total.item".format(path), "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            cur_itemid, _ = lines[i].split("|", maxsplit=1)
            item_set.add(cur_itemid)
            pbar.update()
    
    random.seed(2024)
    np.random.seed(2024)
    valid_dict, test_dict = {}, {}
    user_id_list = list(train_dict.keys())
    user_id_list.sort()
    # print(len(user_id_list))
    # time.sleep(1000)
    with tqdm(total=len(user_id_list)) as pbar:
        count_test, count_valid = 0, 0
        for cur_userid in user_id_list:
            # print(cur_userid)
            total_num = len(positive_dict[cur_userid])
            sample_num = int(0.1 * total_num)
            if sample_num == 0:
                sample_num = 1
            sample_ids = np.array(list(set(positive_dict[cur_userid].keys()) - banned_positive[cur_userid]))
            sample_unseen_ids = np.array(list(item_set - banned_notseen[cur_userid] - intered_dict[cur_userid]))
            
            adjust_test = False
            try:
                test_dict[cur_userid] = []
                test_ids = np.random.choice(sample_ids, size=sample_num, replace=False)
                for cur_itemid in test_ids:
                    test_dict[cur_userid].append(positive_dict[cur_userid][cur_itemid])
                    positive_dict[cur_userid].pop(cur_itemid)
            except:
                # assert len(sample_ids) == 0
                adjust_test = True
                count_test += 1
            
            sample_ids = np.array(list(set(sample_ids) - set(test_ids)))
            adjust_valid = False
            try:
                valid_dict[cur_userid] = []
                valid_ids = np.random.choice(sample_ids, size=sample_num, replace=False)
                for cur_itemid in valid_ids:
                    valid_dict[cur_userid].append(positive_dict[cur_userid][cur_itemid])
                    positive_dict[cur_userid].pop(cur_itemid)
            except:
                # assert len(sample_ids) == 0
                adjust_valid = True
                count_valid += 1
            
            for cur_itemid in positive_dict[cur_userid].keys():
                train_dict[cur_userid].append(positive_dict[cur_userid][cur_itemid])
            train_dict[cur_userid] = sorted(train_dict[cur_userid], key=lambda x:x[-1], reverse=True)

            if adjust_test == False:
                test_unseen_ids = np.random.choice(sample_unseen_ids, size=sample_num * 999, replace=False)
            else:
                test_unseen_ids = np.random.choice(sample_unseen_ids, size=sample_num * 1000, replace=False)
            for cur_itemid in test_unseen_ids:
                test_dict[cur_userid].append([cur_itemid, 0, 0])
            
            sample_unseen_ids = np.array(list(set(sample_unseen_ids) - set(test_unseen_ids)))
            if adjust_valid == False:
                valid_unseen_ids = np.random.choice(sample_unseen_ids, size=sample_num * 999, replace=False)
            else:
                valid_unseen_ids = np.random.choice(sample_unseen_ids, size=sample_num * 1000, replace=False)
            for cur_itemid in valid_unseen_ids:
                valid_dict[cur_userid].append([cur_itemid, 0, 0])
            pbar.update()
        print("test: {}, valid: {}".format(count_test, count_valid))
            
    for i in range(6):
        if i == 5:
            folder = "total"
            group_userid_list = user_id_list
        else:
            folder = i
            with open("{}/user_5k_{}/user_5k_{}.user".format(path, folder, folder), "r") as f:
                lines = f.read().splitlines()
            group_userid_list = []
            for j in range(1, len(lines)):
                cur_userid, _ = lines[j].split("|")
                group_userid_list.append(cur_userid)
        
        with open("{}/user_5k_{}/user_5k_{}.train.inter".format(path2, folder, folder), "w") as f:
            with open("{}/user_5k_{}/user_5k_{}.valid.inter".format(path2, folder, folder), "w") as g:
                with open("{}/user_5k_{}/user_5k_{}.test.inter".format(path2, folder, folder), "w") as h:
                    with open("{}/user_5k_{}/user_5k_{}.user".format(path2, folder, folder), "w") as ff:
                        f.write("user_id:token|item_id:token|rating:float|timestamp:float\n")
                        g.write("user_id:token|item_id:token|rating:float|timestamp:float\n")
                        h.write("user_id:token|item_id:token|rating:float|timestamp:float\n")
                        ff.write("user_id:token|profile:token_seq")
                        with tqdm(total=len(user_id_list)) as pbar:
                            for cur_userid in user_id_list:
                                for item in train_dict[cur_userid]:
                                    f.write("{}|{}|{}|{}\n".format(cur_userid, item[0], item[1], item[2]))
                                for item in valid_dict[cur_userid]:
                                    g.write("{}|{}|{}|{}\n".format(cur_userid, item[0], item[1], item[2]))
                                for item in test_dict[cur_userid]:
                                    h.write("{}|{}|{}|{}\n".format(cur_userid, item[0], item[1], item[2]))
                                ff.write("{}|{}\n".format(cur_userid, default))
                                pbar.update()

        shutil.copy(
            "{}/user_5k_total/user_5k_total.item".format(path),
            "{}/user_5k_{}/user_5k_{}.item".format(path2, folder, folder)
        )
    sys.exit(1)

def generate_potential(dataset):
    path = "./sortdict"
    model_list = ['FM', 'LightGCL', 'LightGCN', 'SimpleX', 'XSimGCL']
    potential_dict = {}
    cut_size_list = [50]
    
    if dataset == "amazon":
        origin_path = "./amazon/user_5k/CDs_and_Vinyl"
    elif dataset == "ml-25m":
        origin_path = "./ml-25m/user_5k"
    
    item_dict = {}
    with open("{}/u.item".format(origin_path), "r") as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        cur_itemid, cur_profile = lines[i].split("|", maxsplit=1)
        item_dict[cur_itemid] = cur_profile
    
    for model in model_list:
        with open("{}/{}_{}.pkl".format(path, dataset, model), "rb") as f:
            cur_sortdict = pickle.load(f)
        with tqdm(total=len(cur_sortdict), desc="{}".format(model)) as pbar:
            for user_id in cur_sortdict.keys():
                if user_id not in potential_dict.keys():
                    potential_dict[user_id] = {}
                for cur_tuple in cur_sortdict[user_id]:
                    if cur_tuple[0] not in potential_dict[user_id].keys():
                        potential_dict[user_id][cur_tuple[0]] = [cur_tuple[1][0], 0.0]
                    potential_dict[user_id][cur_tuple[0]][-1] += cur_tuple[1][1] / len(model_list)
                pbar.update()
    
    for user_id in potential_dict.keys():
        potential_dict[user_id] = sorted(potential_dict[user_id].items(), key=lambda x:x[1][-1], reverse=True)
    
    for cut_size in cut_size_list:
        with open("{}/{}_potential_{}.txt".format(path, dataset, cut_size), "w") as f:
            with tqdm(total=len(potential_dict), desc="Saving {}...".format(cut_size)) as pbar:
                for user_id in potential_dict.keys():
                    potential_dict[user_id] = potential_dict[user_id][:cut_size]
                    for cur_tuple in potential_dict[user_id]:
                        f.write("{}|{}|{}|{}|{}\n".format(user_id, cur_tuple[0], cur_tuple[1][0], cur_tuple[1][1], item_dict[cur_tuple[0]]))
                    pbar.update()
    sys.exit(1)

def cut_potential(dataset):
    if dataset == "ml-25m":
        with open("/*********/*****/**********/***/sortdict/ml-25m_potential_30_newdata.txt", "r") as f:
            lines = f.read().splitlines()
    elif dataset == "amazon-CDs_and_Vinyl":
        with open("/*********/*****/**********/***/sortdict/amazon_potential_30_newdata.txt", "r") as f:
            lines = f.read().splitlines()
    
    cut_len = 20
    if dataset == "ml-25m":
        file = "/*********/*****/**********/***/sortdict/ml-25m_potential_{}_newdata.txt".format(cut_len)
    elif dataset == "amazon-CDs_and_Vinyl":
        file = "/*********/*****/**********/***/sortdict/amazon_potential_{}_newdata.txt".format(cut_len)
    with open(file, "w") as f:
        for i in range(0, len(lines), 30):
            for j in range(cut_len):
                if i + j < len(lines):
                    f.write("{}\n".format(lines[i + j]))
                else:
                    break
    

def get_generate_llama(client, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
    if instruction is None:
        finalinput = input
    else:
        finalinput = instruction + input
    # print(finalinput)
    response = client.chat(model='llama3', messages=finalinput)
    # print(response)
    return response['message']['content']

def extract_profile(string):
    string = string.replace("\n", "")
    # pattern = r'(?<=Potential interest:).*'
    # pattern2 = r'(?<=Potential interest: ).*'
    patterns = [r'(?<=Potential interest:).*', r'(?<=Potential interest: ).*',
                r'(?<=Potential Interest:).*', r'(?<=Potential Interest: ).*',]
    string = re.sub('\s+',' ',string)
    result = None
    for pattern in patterns:
        match = re.search(pattern, string)
        if match:
            result = match.group(0)
    if result is not None:
        return result
    else:
        raise Exception("Cannot decode potential interest!")

def get_generate(input_dict):
    llm_obj = ChatOpenAI(
        base_url = "https://svip.xty.app/v1",
        api_key = os.getenv("UPLLM_CHAT_API_KEY", ""),
        model = "deepseek-v3",
        temperature=0.0,
        request_timeout=60,
        max_retries=100
    )
    response = llm_obj.invoke(input_dict)
    return response.content

def generate_user_pref(dataset, port):
    prompt_obj = Prompt()
    potential_prefix = prompt_obj.summarize_potential[0]
    potential_prefix, potential_postfix = potential_prefix[:-1], potential_prefix[-1:]
    if dataset == 'ml-25m':
        cut_len = 30
        # with open("./sortdict/{}_potential_{}.txt".format(dataset, cut_len), "r") as f:
        #     lines = f.read().splitlines()
        with open("./sortdict/ml-25m_potential_{}.txt".format(cut_len), "r") as f:
            lines = f.read().splitlines()
    elif dataset == 'amazon-CDs_and_Vinyl':
        cut_len = 20
        with open("./sortdict/amazon_potential_{}.txt".format(cut_len), "r") as f:
            lines = f.read().splitlines()
    generate_dict = {}
    # lenlist = []
    # idset = set()
    for i in range(len(lines)):
        cur_list = lines[i].split("|")
        assert len(cur_list) == 7
        cur_userid = cur_list[0]
        if cur_userid not in generate_dict.keys():
            generate_dict[cur_userid] = []
        cur_itemprof = "Title: {} | Category: {} | Description: {}\n".format(cur_list[-3], cur_list[-2], cur_list[-1])
        generate_dict[cur_userid].append(cur_itemprof)
        # if cur_list[1] not in idset:
        #     lenlist.append(len(cur_list[-1]))
        #     idset.add(cur_list[1])
    # lenlist.sort()
    # print(len(lenlist), sum(lenlist) / len(lenlist), lenlist[len(lenlist) // 2])
    # time.sleep(1000)

    user_pref_dict, temp_dict = {}, {}
    start_list = ['Based on', 'The user', 'This user', 'It seems', 'Given', 'It appears',
                  '**Based on', '**The user', '**This user', '**It seems', '**Given', '**It appears']
    
    ollama_client = ollama.Client(host="http://127.0.0.1:{}".format(port))
    errorid_list = []
    error_num = 0
    # mapping_dict = {11435:0, 11436:1, 11437:2, 11438:3, 11439:4, 11440:5, 11441:6, 11442:7}
    # mapping_dict = {11434:0, 11435:1, 11436:2, 11438:3, 11439:4, 11440:5, 11441:6}
    mapping_dict = {11437:0, 11439:1, 11441:2, 11442:3}
    userid_list = list(generate_dict.keys())
    userid_list = userid_list[len(userid_list) * mapping_dict[port] // len(mapping_dict) : len(userid_list) * (mapping_dict[port] + 1) // len(mapping_dict)]
    part = 4
    userid_list = userid_list[len(userid_list) * part // 5 : len(userid_list) * (part + 1) // 5]
    
    with open("./sortdict/{}_userpotential_{}_{}_{}.txt".format(dataset, cut_len, port, part), "w") as f:
        with tqdm(total=len(userid_list)) as pbar:
        # with tqdm(total=len(temp_dict)) as pbar:
            count = 0
            # userid_list = list(generate_dict.keys())
            # userid_list = list(temp_dict.keys())
            # userid_list = userid_list[len(userid_list) * mapping_dict[port] // len(mapping_dict) : len(userid_list) * (mapping_dict[port] + 1) // len(mapping_dict)]
            
            # userid_list = ['A2C83MC9APW47P', 'A2DT12NYGEY6CT', 'AP08NQGPQ321T']
            # userid_list = ['A1A2FQKRSD9GL9', 'A1K66EHHQ4MYX7', 'A1TX3VNWII1TOJ', 'A1V6TK1Z3P2U2J', 'A1YNH0LC69JYU4', 'A2AM5S09AAB83E', 'A2C83MC9APW47P', 'A2CONYQZVV07U9', 'A2DT12NYGEY6CT', 'A2I5LVVQ7WXEH6', 'A2JFDCLA9PZGQ0', 'A2L3M907XYFZ2D', 'A2O8XMNIBHZXCE', 'A2TYSPEDVIGF7W', 'A341T7E1QOHA2I', 'A38KZE4XLCHFD0', 'A3AQ35OMRJRYVK', 'A3AVJCB1ZD6ZY5', 'A3BJOYY5ZWF5VM', 'A3N0W7NSLD552L', 'A3TLB3ZQPSNOUX', 'A3UWABJ3RUDZ3E', 'AEV640MKB3I9J', 'AIVWYHNOME34I', 'AJ04DW48JANUQ', 'AOKQZVWCLONRH', 'AP08NQGPQ321T', 'ARFCORBCTKX1J']
            # userid_list = ["A1A8LTTF3V05FI", "A1XDFREDNLWW6T", "A26DOHWKRZPN8H", "A2782HW7CZX9JL", "A2CG7N2QUS9FG7", "A2KW2KWKABNYNO", "A2WZYL0HNUR5XH", "A2XXHJEWYTPZZ3", "A397ND4UT6BTXR", "A3QHN7P5Y43MDU", "A5X5EGH95H0OQ", "AF65C8QK9NNXP"]
            # userid_list = ['5329', '68124', '8550', '86037', '99662']
            # userid_list = ['113968', '21433', '44190', '8185', '83723']
            # userid_list = ['102593', '107733', '110921', '111776', '144919', '154942', '159745']
            while count < len(userid_list):
            # while count < 10:
                cur_userid = userid_list[count]
                if cur_userid in user_pref_dict:
                    f.write("{}|{}\n".format(cur_userid, user_pref_dict[cur_userid]))
                    count += 1
                    pbar.update()
                    continue
                
                # print(count)
                input_msg = ""
                for i in range(len(generate_dict[cur_userid]) - error_num):
                # for i in range(len(generate_dict[cur_userid])):
                    input_msg += "{}. {}".format(i + 1, generate_dict[cur_userid][i])
                input_dict = [{"role": "user", "content": input_msg}]
                input_dict += potential_postfix
                input_dict = potential_prefix + input_dict
                try:
                    ret = get_generate_llama(ollama_client, potential_prefix, input_dict)
                    # ret = get_generate(input_dict)
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        sys.exit(1)
                    traceback.print_exc()
                    # print("Sleep!!!")
                    # time.sleep(10)
                    continue
                # print("Ret = " + ret)
                try:
                    ret = extract_profile(ret)
                    flag = False
                    for start in start_list:
                        if ret.startswith(start) or ret.startswith(start, 1):
                            flag = True
                            break
                    if not flag:
                        # print("Ret = " + ret)
                        raise Exception
                    print("Ret = " + ret)
                except Exception as e:
                    print(ret)
                    if isinstance(e, KeyboardInterrupt):
                        sys.exit(1)
                    traceback.print_exc()
                    # time.sleep(5)
                    error_num += 1
                    if error_num < 100:
                        continue
                    else:
                        errorid_list.append(cur_userid)
                        error_num = 0
                        count += 1
                        pbar.update()
                        continue
                f.write("{}|{}\n".format(cur_userid, ret))
                count += 1
                error_num = 0
                pbar.update()
    with open("./sortdict/errorid_list_{}.pkl".format(port), "wb") as f:
        pickle.dump(errorid_list, f)
    sys.exit(1)

def combine_profile():
    path = "/*********/*****/**********/***/amazon_results/amazon_5_32_full/checkpoints"
    profile_dict = {}
    with tqdm(total=len(os.listdir(path))) as pbar:
        for file in os.listdir(path):
            pbar.update()
            if not file.endswith("final_profile.pkl"):
                continue
            cur_userid = file.split("_")[1]
            # cur_userid = str(int(float(cur_userid)))
            with open("{}/{}".format(path, file), "rb") as f:
                cur_profile = pickle.load(f)
            profile_dict[cur_userid] = cur_profile
    with open("./all_profile.pkl", "wb") as f:
        pickle.dump(profile_dict, f)
    sys.exit(1)
    
def modify_ndcg():
    ndcg_20, ndcg_20_index = [], []
    # ndcg_20_index += list(itertools.combinations(range(20), 1))
    # ndcg_20_index += list(itertools.combinations(range(20), 2))
    ndcg_20_index += list(itertools.combinations(range(20), 3))
    for i in range(len(ndcg_20_index)):
        if len(ndcg_20_index[i]) == 1:
            ndcg_20.append(float(1 / np.log2(ndcg_20_index[i][0] + 2)))
        elif len(ndcg_20_index[i]) == 2:
            dcg = float(1 / np.log2(ndcg_20_index[i][0] + 2)) + float(1 / np.log2(ndcg_20_index[i][1] + 2))
            idcg = float(1 + 1 / np.log2(3))
            ndcg_20.append(dcg / idcg)
        elif len(ndcg_20_index[i]) == 3:
            dcg = float(1 / np.log2(ndcg_20_index[i][0] + 2)) + float(1 / np.log2(ndcg_20_index[i][1] + 2)) + float(1 / np.log2(ndcg_20_index[i][2] + 2))
            idcg = float(1 + 1 / np.log2(3) + 1 / np.log2(4))
            ndcg_20.append(dcg / idcg)
    
    zipped = list(zip(ndcg_20_index, ndcg_20))
    zipped = sorted(zipped, key=lambda x:x[1], reverse=True)
    ndcg_20_index = [x[0] for x in zipped]
    ndcg_20 = [x[1] for x in zipped]
    # print(ndcg_20_index[:10])
    # print(ndcg_20[:10])
    # time.sleep(1000)
    
    path = "/*********/*****/**********/***/full_results/FM-user_5k_total-ml25m/ndcg.pkl"
    with open(path, "rb") as f:
        cur = pickle.load(f)
    
    ndcg_20_values = cur[20][0] * 5
    ndcg_rank = []
    with tqdm(total=len(ndcg_20_values)) as pbar:
        for i in range(len(ndcg_20_values)):
            find = False
            for a in range(len(ndcg_20_index)):
                if ndcg_20[a] > ndcg_20_values[i]:
                    break
                if find:
                    break
                for b in range(a, len(ndcg_20_index)):
                    if ndcg_20[a] + ndcg_20[b] > ndcg_20_values[i]:
                        break
                    if find:
                        break
                    for c in range(b, len(ndcg_20_index)):
                        if ndcg_20[a] + ndcg_20[b] + ndcg_20[c] > ndcg_20_values[i]:
                            break
                        if find:
                            break
                        for d in range(c, len(ndcg_20_index)):
                            if ndcg_20[a] + ndcg_20[b] + ndcg_20[c] + ndcg_20[d] > ndcg_20_values[i]:
                                break
                            if find:
                                break
                            for e in range(d, len(ndcg_20_index)):
                                print(a, b, c, d, e)
                                if find:
                                    break
                                # print(ndcg_20[a] + ndcg_20[b] + ndcg_20[c] + ndcg_20[d] + ndcg_20[e], ndcg_20_values[i])
                                # time.sleep(1)
                                if ndcg_20_values[i] - ndcg_20[a] + ndcg_20[b] + ndcg_20[c] + ndcg_20[d] + ndcg_20[e] > 1e-6:
                                    break
                                elif abs(ndcg_20_values[i] - ndcg_20[a] + ndcg_20[b] + ndcg_20[c] + ndcg_20[d] + ndcg_20[e]) < 1e-6:
                                    ndcg_rank.append((ndcg_20_index[a], ndcg_20_index[b], ndcg_20_index[c], ndcg_20_index[d], ndcg_20_index[e]))
                                    find = True
            pbar.update()
    sys.exit(1)

def extract_new_profile(string):
    string = string.replace("\n", "")
    string = re.sub('\s+',' ',string)
    pattern_list = [r'(?<=New user profile:)\[.*\]', r'(?<=New user profile: )\[.*\]', r'(?<=New User Profile:)\[.*\]', r'(?<=New User Profile: )\[.*\]']
    result = None
    for pattern in pattern_list:
        match_res = re.search(pattern, string)
        if match_res:
            result = match_res.group(0)
            break
    if result is not None:
        return result
    else:
        raise Exception("Cannot decode potential interest!")

def modify_user_profile(dataset, port, cut_len, part):
    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[0]
    with open("./sortdict/{}_userpotential_{}.txt".format(dataset, cut_len), "r") as f:
        lines = f.read().splitlines()
    potential_dict, final_profile_dict = {}, {}
    for i in range(len(lines)):
        user_id, potential = lines[i].split("|")
        potential_dict[user_id] = potential
    
    # for file in os.listdir("./ml-25m_results/ml-25m_5_32_full/checkpoints_new_30"):
    #     cur_userid = file.split("_")[1]
    #     potential_dict.pop(cur_userid)
    
    # if dataset == "ml-25m":
    #     if cut_len == 10 or cut_len == 30:
    #         base_folder = "/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full"
    #     else:
    #         base_folder = "/*********/*****/**********/***/ml-25m_results/ml-25m_cfscore_new"
    # elif dataset == "amazon-CDs_and_Vinyl":
    #     if cut_len == 10 or cut_len == 30:
    #         base_folder = "/*********/*****/**********/***/amazon_results/amazon_5_32_full"
    #     else:
    #         base_folder = "/*********/*****/**********/***/amazon_results/amazon_cfscore_new"
    
    if dataset == "ml-25m":
        base_folder = "./ml-25m_results/ml-25m_cfscore_3level"
    elif dataset == "amazon-CDs_and_Vinyl":
        base_folder = "/*********/*****/**********/***/amazon_results/amazon_cfscore_5level"
    
    with tqdm(total=len(potential_dict)) as pbar:
        for user_id in potential_dict.keys():
            if dataset == "ml-25m":
                cur_userid = str(float(int(user_id)))
            else:
                cur_userid = user_id
            with open("{}/checkpoints/user_{}_final_profile.pkl".format(base_folder, cur_userid), "rb") as f:
                cur_final_profile = pickle.load(f)
            final_profile_dict[user_id] = cur_final_profile
            pbar.update()
    
    # port = "11439"
    # cut_len = 10
    ollama_client = ollama.Client(host="http://127.0.0.1:{}".format(port))
    error_ids = []
    
    # base_dict = {11435:0, 11436:1, 11437:2, 11438:3, 11439:4, 11440:5, 11441:6}
    # base_dict = {11434:0, 11435:1, 11436:2, 11437:3, 11438:4, 11439:5, 11442:6}
    # base_dict = {11435:0, 11436:1, 11438:2, 11439:3, 11440:4, 11441:5}
    # base_dict = {11438:0, 11439:1, 11440:2, 11441:3}
    base_dict = {11436:0, 11437:1, 11441:2, 11442:3}
    
    userid_list = list(potential_dict.keys())
    # userid_list = userid_list[len(userid_list) * 1 // 6 : len(userid_list) * 2 // 6]
    # userid_list = ['102593', '107733', '110921', '111776', '144919', '154942', '159745']
    # userid_list = userid_list[:len(userid_list) // 7]
    # userid_list = userid_list[len(userid_list) * (int(port) - 11434) // 4 : len(userid_list) * (int(port) - 11434 + 1) // 4]
    userid_list = userid_list[len(userid_list) * base_dict[port] // len(base_dict) : len(userid_list) * (base_dict[port] + 1) // len(base_dict)]
    userid_list = userid_list[len(userid_list) * part // 5 : len(userid_list) * (part + 1) // 5]
    # userid_list = ['127421']
    # userid_list = userid_list[len(userid_list) // 3:]
    
    with tqdm(total=len(userid_list)) as pbar:
        count = 0
        error_num = 0
        while count < len(userid_list):
        # while count < 2:
            user_id = userid_list[count]
            cur_potential = potential_dict[user_id]
            cur_final_profile = final_profile_dict[user_id]
            
            # print(cur_final_profile)
            
            input_msg = """User Profile: {}.\nPotential Preference Analyses: {}.\n""".format(cur_final_profile, cur_potential)
            input_dict = [{"role": "user", "content": input_msg}]
            try:
                ret = get_generate_llama(ollama_client, modify_prefix, input_dict)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit(1)
                # print("Sleep!!!")
                # time.sleep(10)
                continue
            # print(ret)
            
            try:
                ret = extract_new_profile(ret)
                # print(ret)
                # ret = eval(ret)
                ret = ast.literal_eval(ret)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit(1)
                traceback.print_exc()
                error_num += 1
                if error_num < 100:
                    continue
                else:
                    error_ids.append(user_id)
                    error_num = 0
                    count += 1
                    continue
            
            if not os.path.exists("{}/checkpoints_new_{}".format(base_folder, cut_len)):
                os.mkdir("{}/checkpoints_new_{}".format(base_folder, cut_len))
            with open("{}/checkpoints_new_{}/user_{}_final_profile.pkl".format(base_folder, cut_len, user_id), "wb") as f:
                pickle.dump(ret, f)
            error_num = 0
            count += 1
            pbar.update()
    # with open("{}/new_profile_errorids_{}_{}.pkl".format(base_folder, port, cut_len), "wb") as f:
    #     pickle.dump(error_ids, f)
    # sys.exit(1)

def get_errorids():
    with open("/*********/*****/**********/***/ml-25m_results/ml-25m_cfscore/new_profile_errorids_11439.pkl", "rb") as f:
        erroridlist = pickle.load(f)
    print(erroridlist)
    sys.exit(1)

def get_embedding(client, sentence):
    result = client.embeddings(
        model='llama3',
        prompt=sentence
    )
    return result['embedding']

def list2sentence(profile_list):
    for i in range(len(profile_list) - 1, -1, -1):
        if profile_list[i] is Ellipsis:
            profile_list.pop(i)

    sentence = ""
    for i in range(len(profile_list)):
        sentence += str(profile_list[i])
        if i != len(profile_list) - 1:
            sentence += ", "
        else:
            sentence += "."
    return sentence

def gen_modify_userembed():
    path = "./ml-25m_results/ml-25m_cfscore_3level"
    port = 11442
    client = ollama.Client(host="http://127.0.0.1:{}".format(port))
    userembed_dict = {}
    with tqdm(total=len(os.listdir("{}/checkpoints_new_30".format(path)))) as pbar:
        for file in os.listdir("{}/checkpoints_new_30".format(path)):
            cur_userid = file.split("_")[1]
            with open("{}/checkpoints_new_30/{}".format(path, file), "rb") as f:
                cur_profile = pickle.load(f)
            if isinstance(cur_profile, tuple):
                cur_profile_new = []
                for temp in cur_profile:
                    cur_profile_new += temp
                cur_profile = cur_profile_new
            try:
                cur_embed = get_embedding(client, list2sentence(cur_profile))
            except:
                traceback.print_exc()
                sys.exit(1)
            userembed_dict[cur_userid] = cur_embed
            pbar.update()
    
    # with tqdm(total=len(os.listdir("{}/checkpoints".format(path)))) as pbar:
    #     for file in os.listdir("{}/checkpoints".format(path)):
    #         if file.endswith("final_profile.pkl"):
    #             cur_userid = file.split("_")[1]
    #             with open("{}/checkpoints/{}".format(path, file), "rb") as f:
    #                 cur_profile = pickle.load(f)
    #             if isinstance(cur_profile, tuple):
    #                 cur_profile_new = []
    #                 for temp in cur_profile:
    #                     cur_profile_new += temp
    #                 cur_profile = cur_profile_new
    #             try:
    #                 cur_embed = get_embedding(client, list2sentence(cur_profile))
    #             except:
    #                 traceback.print_exc()
    #                 sys.exit(1)
    #             userembed_dict[cur_userid] = cur_embed
    #         pbar.update()
    
    userembeds = sorted(userembed_dict.items(), key=lambda x:x[0])
    with open("{}/datasetname_{}.useremb".format(path, port), "w") as f:
        f.write("uid:token|user_emb:float_seq\n")
        with tqdm(total=len(userembeds)) as pbar:
            for i in range(len(userembeds)):
                f.write("{}|".format(userembeds[i][0]))
                for j in range(len(userembeds[i][1])):
                    f.write("{}, ".format(userembeds[i][1][j]))
                f.write("\n")
                pbar.update()
    sys.exit(1)

def direct_userembed():
    folder = "./logs/2025-05-02_10:18:41"
    embed_dict = {}
    with tqdm(total=len(os.listdir("{}/user".format(folder)))) as pbar:
        for file in os.listdir("{}/user".format(folder)):
            cur_userid = int(file.split(".")[0])
            with open("{}/checkpoints/user_{}_final_totalembed.pkl".format(folder, float(cur_userid)), "rb") as f:
                cur_embed = pickle.load(f)
                embed_dict[cur_userid] = cur_embed
            pbar.update()
    embed_dict = sorted(embed_dict.items(), key=lambda x:x[0])
    with tqdm(total=len(embed_dict)) as pbar:
        with open("{}/xxx.useremb".format(folder), "w") as f:
            for item in embed_dict:
                f.write("{}|".format(item[0]))
                for i in range(len(item[1])):
                    f.write("{}, ".format(item[1][i]))
                f.write("\n")
                pbar.update()
    sys.exit(1)

def analyze_modelembed():
    """
    ml-25m: 0.7163684368133545 -0.6944116950035095
    amazon: 0.9911388158798218 -0.7730361223220825
    https://blog.csdn.net/guyuealian/article/details/78845031
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    
    path = "/*********/*****/**********/***/ml-25m/user_5k"
    with open("{}/user_modelembed.pkl".format(path), "rb") as f:
        temp = pickle.load(f)[2022]
        user_modelembed, user_mapping = temp['embedding'], temp['mapping'].tolist()
    with open("{}/item_modelembed.pkl".format(path), "rb") as f:
        temp = pickle.load(f)[2022]
        item_modelembed, item_mapping = temp['embedding'], temp['mapping'].tolist()
    # path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl"
    # with open("{}/user_modelembed.pkl".format(path), "rb") as f:
    #     temp = pickle.load(f)[2021]
    #     user_modelembed, user_mapping = temp['embedding'], temp['mapping']
    # with open("{}/item_modelembed.pkl".format(path), "rb") as f:
    #     temp = pickle.load(f)[2021]
    #     item_modelembed, item_mapping = temp['embedding'], temp['mapping']
    
    user_modelembed = user_modelembed / user_modelembed.norm(dim=1, keepdim=True)
    item_modelembed = item_modelembed / item_modelembed.norm(dim=1, keepdim=True)
    print(user_modelembed.device, item_modelembed.device)
    res = torch.mm(user_modelembed, item_modelembed.T).cpu()
    maxcos, mincos = res.max().item(), res.min().item()
    print(maxcos, mincos)
    print(res.median())
    positive_pos = torch.where(res >= 0)
    negative_pos = torch.where(res <= 0)
    positive_k = (1 - 0) / (maxcos - 0)
    negative_k = (0 - (-1)) / (0 - mincos)
    res[positive_pos] = 0 + positive_k * (res[positive_pos] - 0)
    res[negative_pos] = -1 + negative_k * (res[negative_pos] - mincos)
    
    # time.sleep(1000)
    # k = 2 / (maxcos - mincos)
    # res = (res - mincos) * k - 1
    maxcos, mincos = res.max().item(), res.min().item()
    print(maxcos, mincos)
    print(res.median(), res.mean())
    # time.sleep(1000)
    user_mapping = zip(user_mapping, list(range(len(user_mapping))))
    user_mapping = {x[0]:x[1] for x in user_mapping}
    item_mapping = zip(item_mapping, list(range(len(item_mapping))))
    item_mapping = {x[0]:x[1] for x in item_mapping}
    res = res.cpu()
    with open("{}/cosmatrix.pkl".format(path), "wb") as f:
        pickle.dump({'cosmatrix': res, 'user_mapping': user_mapping, 'item_mapping': item_mapping}, f)
    sys.exit(1)

def temp():
    with open("/*********/*****/**********/***/amazon-CDs_and_Vinyl_full7.pkl", "rb") as f:
        groups = pickle.load(f)
    
    for i in range(len(groups)):
        print(groups[i][0])
    sys.exit(1)
    """
    A2QXK21Q1AGMVO
    A3CKW7Y7Q8NGAU
    A274EO7M9ICQU3
    A107KVRYRCU3RY
    A2STBLHVSTWJOG
    A25VIJ228PQZA
    A215P85W653CZV
    """
    
    # group 0: 4-6, 5-26, 6-32, 9-20, 10-26, 11-10, 12-21, 14-3, 17-10
    # group 1: 1-33, 2-27, 3-25, 4-6, 6-4, 7-17, 8-31, 10-19, 12-7, 13-7, 17-28, 18-26, 19-22
    # group 2: 2-19, 3-26, 4-17, 5-34, 8-11, 10-30, 11-2, 15-24, 19-23
    # group 3: 0-33, 1-17, 5-25, 7-19, 10-0, 13-27, 14-29, 15-28, 16-29, 17-9
    # group 4: 2-32, 5-9, 8-24, 10-6, 14-3, 15-34, 16-15, 17-27, 
    # group 5: 1-26, 3-31, 12-25, 15-24
    # group 6: 14-1

def generate_inputprompt(dataset):
    candidate_num = 19
    interact_max_num = 10
    if dataset == "ml-25m":
        path = "/*********/*****/**********/***/ml-25m/user_5k"
        llm_input = [
            "[UserRep] is a user representation. This user has watched ",
            " in the past. Recommend a movie for this user to watch next from the following set of movie titles, ",
            ". The recommendation is "
        ]
    elif dataset == "amazon-CDs_and_Vinyl":
        path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl"
        llm_input = [
            "[UserRep] is a user representation. This user has listened ",
            " in the past. Recommend a CDs_and_Vinyl for this user to listen next from the following set of music titles, ",
            ". The recommendation is "
        ]
    with open("{}/u_train.data".format(path), "r") as f:
        lines = f.read().splitlines()
    user_inter_dict, item_title_dict = {}, {}
    user_interacted = {}
    for i in range(len(lines)):
        user_id, item_id, _, _ = lines[i].split("\t")
        if user_id not in user_inter_dict.keys():
            user_inter_dict[user_id] = []
            user_interacted[user_id] = {}
        user_inter_dict[user_id].append(item_id)
        user_interacted[user_id][item_id] = 0
    with open("{}/u_valid.data".format(path), "r") as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        user_id, item_id, _, _ = lines[i].split("\t")
        user_interacted[user_id][item_id] = 0
    with open("{}/u_test.data".format(path), "r") as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        user_id, item_id, _, _ = lines[i].split("\t")
        user_interacted[user_id][item_id] = 0
    
    with open("{}/recbole/user_5k/user_5k_total/user_5k_total.item".format(path), "r") as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        item_id, item_title, _, _ = lines[i].split("|")
        item_title_dict[item_id] = item_title
    
    user_prompt_dict = {}
    user_history_dict, user_candidate_dict, user_target_dict = {}, {}, {}
    all_item_id = list(item_title_dict)
    np.random.seed(2024)
    random.seed(2024)
    with tqdm(total=len(user_inter_dict)) as pbar:
        for user_id in user_inter_dict.keys():
            user_history_dict[user_id], user_candidate_dict[user_id] = [], []
            history_list, candidate_list = [], []
            user_inter_dict[user_id] = user_inter_dict[user_id][-(interact_max_num + 1):]
            for i in range(len(user_inter_dict[user_id]) - 1):
                item_id = user_inter_dict[user_id][i]
                history_list.append(item_title_dict[item_id] + '[HistoryEmb]')
                user_history_dict[user_id].append(item_id)
            history_list = ','.join(history_list)
            while len(candidate_list) < candidate_num:
                index = np.random.randint(1, len(all_item_id) + 1)
                item_id = all_item_id[index]
                if item_id not in user_interacted[user_id].keys():
                    candidate_list.append(item_title_dict[item_id] + '[CandidateEmb]')
                    user_candidate_dict[user_id].append(item_id)
            target_item_id = user_inter_dict[user_id][-1]
            candidate_list.append(item_title_dict[target_item_id] + '[CandidateEmb]')
            user_candidate_dict[user_id].append(target_item_id)
            
            temp = list(zip(candidate_list, user_candidate_dict[user_id]))
            random.shuffle(temp)
            candidate_list, user_candidate_dict[user_id] = [x[0] for x in temp], [x[1] for x in temp]
            candidate_list = ','.join(candidate_list)
            input_text = llm_input[0] + history_list + llm_input[1] + candidate_list + llm_input[2]

            output_text = item_title_dict[target_item_id]
            user_target_dict[user_id] = target_item_id
            user_prompt_dict[user_id] = (input_text, output_text)
            pbar.update()
    with open("{}/user_prompt_dict.pkl".format(path), "wb") as f:
        save_dict = {
            'user_prompt_dict': user_prompt_dict, # (input_text, output_text)
            'user_history_dict': user_history_dict, # [item_ids]
            'user_candidate_dict': user_candidate_dict, # [item_ids]
            'user_target_dict': user_target_dict # item_id
        }
        pickle.dump(save_dict, f)
    # sys.exit(1)

def cal_parts_ratio(cosmatrix):
    temp = np.arange(0, 1, 0.01)
    for i in temp:
        first_ratio = torch.sum(torch.ge(cosmatrix, i)) / (cosmatrix.shape[0] * cosmatrix.shape[1])
        third_ratio = torch.sum(torch.lt(cosmatrix, -i)) / (cosmatrix.shape[0] * cosmatrix.shape[1])
        second_ratio = 1 - first_ratio - third_ratio
        print(i, first_ratio.item(), second_ratio.item(), third_ratio.item())

def analyze_cosratio2(dataset):
    if dataset == "ml-25m":
        path = "./ml-25m/user_5k/cosmatrix.pkl"
    elif dataset == "amazon-CDs_and_Vinyl":
        path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/cosmatrix.pkl"
    with open(path, "rb") as f:
        temp = pickle.load(f)
        cosmatrix = temp['cosmatrix']
    
    cal_parts_ratio(cosmatrix)
    sys.exit(1)

def analyze_cosratio(dataset):
    if dataset == "ml-25m":
        path = "/*********/*****/**********/***/ml-25m/user_5k/cosmatrix.pkl"
    elif dataset == "amazon-CDs_and_Vinyl":
        path = "/*********/*****/**********/***/amazon/user_5k/CDs_and_Vinyl/cosmatrix.pkl"
    with open(path, "rb") as f:
        temp = pickle.load(f)
        cosmatrix = temp['cosmatrix']
    
    # print(torch.sum(torch.ge(cosmatrix, 0)))
    # time.sleep(1000)
    
    # score_ratio = np.zeros(11)
    # range_list = [-1.00, -0.95, -0.80, -0.60, -0.40, -0.10, 0.10, 0.40, 0.60, 0.80, 0.95]
    # for i in range(len(range_list)):
    #     score_ratio[i] = torch.sum(torch.ge(cosmatrix, range_list[i]))
    # for i in range(score_ratio.shape[0] - 1):
    #     score_ratio[i] -= score_ratio[i + 1]
    
    # score_ratio /= (cosmatrix.shape[0] * cosmatrix.shape[1])
    # score_ratio *= 100
    # print(score_ratio.tolist())
    
    # positive_pos = torch.where(res >= 0)
    # negative_pos = torch.where(res <= 0)
    # positive_k = (1 - 0) / (maxcos - 0)
    # negative_k = (0 - (-1)) / (0 - mincos)
    # res[positive_pos] = 0 + positive_k * (res[positive_pos] - 0)
    # res[negative_pos] = -1 + negative_k * (res[negative_pos] - mincos)
    
    maxvalue1 = 0.4
    maxvalue2 = 0.22
    max_pos = torch.where(cosmatrix >= maxvalue1)
    min_pos = torch.where(cosmatrix < -maxvalue2)
    cosmatrix[max_pos] = maxvalue1
    cosmatrix[min_pos] = -maxvalue2
    positive_pos = torch.where(cosmatrix >= 0)
    negative_pos = torch.where(cosmatrix <= 0)
    positive_k = (1 - 0) / (maxvalue1 - 0)
    negative_k = (0 - (-1)) / (0 - (-maxvalue2))
    cosmatrix[positive_pos] = 0 + positive_k * (cosmatrix[positive_pos] - 0)
    cosmatrix[negative_pos] = -1 + negative_k * (cosmatrix[negative_pos] - (-maxvalue2))
    
    maxcos, mincos = cosmatrix.max().item(), cosmatrix.min().item()
    print(maxcos, mincos)
    print(cosmatrix.median(), cosmatrix.mean())
    print("")
    # cal_parts_ratio(cosmatrix)
    temp['cosmatrix'] = cosmatrix
    with open(path[:-4] + "_new.pkl", "wb") as f:
        pickle.dump(temp, f)

def testing():
    with open("/*********/*****/**********/***/sortdict/ml-25m_potential_10.txt", "r") as f:
        lines = f.read().splitlines()
    user_set = set()
    for i in range(len(lines)):
        user_id, _ = lines[i].split("|", maxsplit=1)
        if user_id not in user_set:
            user_set.add(user_id)
    
    start_list = ['Based on', 'The user', 'This user', 'It seems', 'Given', 'It appears']
    
    # user_set2 = set()
    with open("/*********/*****/**********/***/sortdict/ml-25m_userpotential_10.txt", "r") as f:
        lines = f.read().splitlines()
    for i in range(len(lines)):
        user_id, pref = lines[i].split("|")
        flag = False
        for start in start_list:
            if pref.startswith(start) or pref.startswith(start, 1):
                user_set.remove(user_id)
                flag = True
                break
        if not flag:
            print(lines[i])
        # if user_id not in user_set2:
        #     user_set2.add(user_id)
    
    # print(list(user_set - user_set2))
    # print(list(user_set2 - user_set))
        
    # print(user_set, len(user_set))
    sys.exit(1)

def testing2():
    # with open("/*********/*****/**********/***/amazon_results/amazon_cfscore_new/datasetname_11438.useremb", "r") as f:
    #     lines = f.read().splitlines()
    # userid, temp = lines[1].split("|")
    # temp = temp.split(", ")[0]
    # print(userid, temp)
    
    # with open("/*********/*****/**********/***/amazon_results/amazon_cfscore_new/checkpoints_new_30/user_A08161909WK3HU7UYTMW_final_profile.pkl", "rb") as f:
    #     profile = pickle.load(f)
    # print(type(profile))
    # client = ollama.Client(host="http://127.0.0.1:11436")
    # cur_embed = get_embedding(client, list2sentence(profile))
    # print(cur_embed[0])
    
    temp = os.listdir("/*********/*****/**********/***/ml-25m_results/ml-25m_cfscore_5level/checkpoints_new_30")
    print(len(temp))
    
    # with open("/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full/new_profile_errorids_11441_30.pkl", "rb") as f:
    #     temp = pickle.load(f)
    # print(temp)
    
    # temp = ["I appreciate classic comedies and family-friendly movies.", "Classic films with strong narratives are my favorite type of movie.", "Well-made dramas and romantic dramas are what I look for in a movie.", "Emotionally resonant films are important to me.", "I enjoy watching movies that leave a lasting impression.", "I prefer films that resonate emotionally.", "Classic dramatic films with strong narratives and Comedies with strong narratives are my favorite types of movies.", "Timeless stories with great character development and Well-crafted storytelling and characters in films are what I look for.", "I appreciate classic dramatic films with strong narratives.", "Timeless stories with great character development are what I look for in a movie.", "Action-packed films with well-crafted storytelling and characters are my favorite type of movie.", "Open to different genres and styles, including animated films, musicals, and Westerns.", "Appreciate engaging storytelling, memorable characters, and a mix of genres and themes in movies."]
    # with open("/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full/checkpoints_new_30/user_127421_final_profile.pkl", "wb") as f:
    #     pickle.dump(temp, f)
    
    # with open("/*********/*****/**********/***/ml-25m_full8.pkl", "rb") as f:
    #     temp = pickle.load(f)
    # for i in range(len(temp)):
    #     print(len(temp[i]))
    
    sys.exit(1)

def get_potential_profile_demo(dataset):
    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[0]
    with open("./sortdict/{}_userpotential_20.txt".format(dataset), "r") as f:
        lines = f.read().splitlines()
    potential_dict, final_profile_dict = {}, {}
    for i in range(len(lines)):
        user_id, potential = lines[i].split("|")
        potential_dict[user_id] = potential
        
    if dataset == "ml-25m":
        base_folder = "/*********/*****/**********/***/ml-25m_results/ml-25m_5_32_full"
    elif dataset == "amazon-CDs_and_Vinyl":
        base_folder = "/*********/*****/**********/***/amazon_results/amazon_5_32_full"
    
    with tqdm(total=len(potential_dict)) as pbar:
        for user_id in potential_dict.keys():
            if dataset == "ml-25m":
                cur_userid = str(float(int(user_id)))
            else:
                cur_userid = user_id
            with open("{}/checkpoints/user_{}_final_profile.pkl".format(base_folder, cur_userid), "rb") as f:
                cur_final_profile = pickle.load(f)
            final_profile_dict[user_id] = cur_final_profile
            pbar.update()
    
    ollama_client = ollama.Client(host="http://127.0.0.1:11435")
    userid_list = list(potential_dict.keys())
    userid_list = random.sample(userid_list, 50)
    with open("./potential_profile_demo.jsonl", "w") as f:
        with tqdm(total=len(userid_list)) as pbar:
            count = 0
            while count < len(userid_list):
                user_id = userid_list[count]
                temp = {"User_ID": user_id}
                f.write("{}\n".format(json.dumps(temp)))
                cur_potential = potential_dict[user_id]
                cur_final_profile = final_profile_dict[user_id]
                input_msg = """User Profile: {}.\nPotential Preference Analyses: {}.\n""".format(cur_final_profile, cur_potential)
                temp = {"User Profile": cur_final_profile}
                f.write("{}\n".format(json.dumps(temp)))
                temp = {"Potential Preference Analyses": cur_potential}
                f.write("{}\n".format(json.dumps(temp)))
                input_dict = [{"role": "user", "content": input_msg}]
                try:
                    ret = get_generate_llama(ollama_client, modify_prefix, input_dict)
                    temp = {"ModelOutput": ret}
                    f.write("{}\n".format(json.dumps(temp)))
                    ret = extract_new_profile(ret)
                    ret = eval(ret)
                    temp = {"New User Profile": ret}
                    f.write("{}\n".format(json.dumps(temp)))
                except:
                    temp = {"Error info": "Error!"}
                    f.write("{}\n".format(json.dumps(temp)))
                pbar.update()
                count += 1
                continue
    sys.exit(1)

def condense_desciption():
    condense_threshold = 512
    with open("./ml-25m/u.item_new_desc2", "r") as f:
        lines = f.read().splitlines()
    description_dict = {}
    for i in range(len(lines)):
        cur_itemid, cur_title, cur_category, cur_description = lines[i].split("|")
        if len(cur_description) > condense_threshold:
            description_dict[cur_itemid] = [cur_title, cur_category, cur_description]
            cur_description = cur_description[:condense_threshold]
            # print(i + 1)
    print(len(description_dict))
    time.sleep(1000)
    
    itemid_list = list(description_dict.keys())
    port_dict = {0:11434, 1:11436, 2:11434, 3:11436, 4:11434, 5:11436}
    groupid = 5
    itemid_list = itemid_list[len(itemid_list) * groupid // len(port_dict) : len(itemid_list) * (groupid + 1) // len(port_dict)]
    client = ollama.Client(host="http://127.0.0.1:{}".format(port_dict[groupid]))
    
    count = 0
    error_num = 0
    batch_size = 1
    prompt_obj = Prompt()
    prompt = prompt_obj.description_prompt[-1]
    with tqdm(total=len(itemid_list)) as pbar:
        while count < len(itemid_list):
        # while count < 4:
            message = """"""
            real_num, gen_num = 0, 0
            for i in range(batch_size):
                if count + i < len(itemid_list):
                    cur_itemid = itemid_list[count + i]
                    message += "{}\t{}\t{}\t{}\n".format(cur_itemid, description_dict[cur_itemid][0], description_dict[cur_itemid][1], description_dict[cur_itemid][2])
                    real_num += 1
                else:
                    break
            input_msg = [{"role": "user", "content": message}]
            try:
                ret = get_generate_llama(client, prompt, input_msg)
                ret = ret.split("\n")
                flag = False
                for i in range(len(ret)):
                    if "The New Descriptions:" in ret[i] or "The new descriptions:" in ret[i]:
                        for j in range(batch_size):
                            if count + j < len(itemid_list):
                                cur_itemid = itemid_list[count + j]
                                cur_description = ret[i + j + 1].split(". ")[-1]
                                description_dict[cur_itemid][-1] = cur_description
                                gen_num += 1
                            else:
                                break
                        if real_num == gen_num:
                            flag = True
                        break
                if not flag:
                    raise Exception
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit(1)
                traceback.print_exc()
                error_num += 1
                if error_num < 10:
                    continue
            count += batch_size
            error_num = 0
            if count <= len(itemid_list):
                pbar.update(batch_size)
            else:
                pbar.update(batch_size - (count - len(itemid_list)))
    with open("./ml-25m/description_dict_{}_{}.pkl".format(port_dict[groupid], groupid), "wb") as f:
        pickle.dump(description_dict, f)
    sys.exit(1)

def gen_new_file():
    description_dict = {}
    with open("./ml-25m/description_dict_11434_0.pkl", "rb") as f:
        cur_dict = pickle.load(f)
    itemid_list = list(cur_dict.keys())
    indexs = [len(itemid_list) * i // 6 for i in range(7)]
    grouped_ids = [itemid_list[indexs[i]:indexs[i + 1]] for i in range(6)]
    
    # for index, port in enumerate([11434, 11435, 11436, 11438, 11439, 11440, 11441]):
    for index, port in enumerate(["11434_0", "11436_1", "11434_2", "11436_3", "11434_4", "11436_5"]):
        with open("./ml-25m/description_dict_{}.pkl".format(port), "rb") as f:
            cur_dict = pickle.load(f)
            for cur_id in grouped_ids[index]:
                description_dict[cur_id] = cur_dict[cur_id]
    with open("./ml-25m/u.item_new_desc2", "r") as f:
        lines = f.read().splitlines()
    with open("./ml-25m/u.item_new_desc3", "w") as f:
        for i in range(len(lines)):
            cur_itemid, _ = lines[i].split("|", maxsplit=1)
            if cur_itemid in description_dict.keys():
                f.write("{}|{}|{}|{}\n".format(cur_itemid, description_dict[cur_itemid][0], description_dict[cur_itemid][1], description_dict[cur_itemid][2]))
            else:
                f.write("{}\n".format(lines[i]))
    sys.exit(1)

def gen_item_embedding():
    with open("./ml-25m/user_5k/u.item", "r") as f:
        lines = f.read().splitlines()
    group = 5
    lines = lines[len(lines) * group // 6 : len(lines) * (group + 1) // 6]
    port_dict = {0: 11435, 1:11436, 2: 11438, 3: 11439, 4:11440, 5:11441}
    client = ollama.Client(host="http://127.0.0.1:{}".format(port_dict[group]))
    
    embed_dict = {}
    with tqdm(total=len(lines)) as pbar:
        for i in range(len(lines)):
            cur_itemid, cur_title, cur_category, cur_description = lines[i].split("|")
            sentence = "{}. {}. {}".format(cur_title, cur_category, cur_description)
            embed = get_embedding(client, sentence)
            embed_dict[cur_itemid] = embed
            pbar.update()
    with open("./ml-25m/user_5k/embed_{}.pkl".format(port_dict[group]), "wb") as f:
        pickle.dump(embed_dict, f)
    sys.exit(1)

def gen_new_embed_file():
    # embed_list = []
    # with open("./ml-25m/user_5k/xxx.itememb", "w") as f:
    #     f.write("iid:token|item_emb:float_seq\n")
    #     for port in [11435, 11436, 11438, 11439, 11440, 11441]:
    #         with open("./ml-25m/user_5k/embed_{}.pkl".format(port), "rb") as g:
    #             embed_dict = pickle.load(g)
    #         with tqdm(total=len(embed_dict)) as pbar:
    #             for cur_itemid in embed_dict.keys():
    #                 embed_list.append(embed_dict[cur_itemid])
    #                 f.write("{}|".format(cur_itemid))
    #                 for value in embed_dict[cur_itemid]:
    #                     f.write("{}, ".format(value))
    #                 f.write("\n")
    #                 pbar.update()
    # with open("./Embeddings/ml-25m/item_totalembed_new.pkl", "wb") as f:
    #     pickle.dump(embed_list, f)
    
    with open("./ml-25m/user_5k/u.item", "r") as f:
        lines = f.read().splitlines()
    itemids = []
    with tqdm(total=len(lines)) as pbar:
        for i in range(len(lines)):
            itemid, _ = lines[i].split("|", maxsplit=1)
            itemids.append(itemid)
            pbar.update()
    with open("./Embeddings/ml-25m/item_totalembed.pkl", "rb") as f:
        embed_list = pickle.load(f)
    print(len(embed_list.keys()))
    with open("./ml-25m/xxx.itememb", "w") as f:
        f.write("iid:token|item_emb:float_seq\n")
        with tqdm(total=len(lines)) as pbar:
            for i in range(len(lines)):
                f.write("{}|".format(itemids[i]))
                for value in embed_list[itemids[i]]:
                    f.write("{}, ".format(value))
                f.write("\n")
                pbar.update()
    sys.exit(1)

def check():
    with open("./ml-25m/user_5k/xxx.itememb", "r") as f:
        temp1 = f.read().splitlines()[1]
    with open("./Embeddings/ml-25m/item_totalembed_new.pkl", "rb") as f:
        temp2 = pickle.load(f)[0]
    print(temp1[:1000])
    print(temp2[:100])
    sys.exit(1)

def modify_potential():
    with open("./sortdict/ml-25m_potential_30.txt", "r") as f:
        lines = f.read().splitlines()
    with open("./ml-25m/user_5k/u.item", "r") as f:
        lines2 = f.read().splitlines()
    description_dict = {}
    title_dict = {}
    for i in range(len(lines2)):
        item_id, title, _, description = lines2[i].split("|")
        title_dict[item_id] = title
        description_dict[item_id] = description
    with open("./sortdict/ml-25m_potential_30_newdata.txt", "w") as f:
        with tqdm(total=len(lines)) as pbar:
            for i in range(len(lines)):
                split_list = lines[i].split("|")
                split_list[-3] = title_dict[split_list[1]]
                split_list[-1] = description_dict[split_list[1]]
                for j in range(len(split_list)):
                    f.write(split_list[j])
                    if j != len(split_list) - 1:
                        f.write("|")
                    else:
                        f.write("\n")
                pbar.update()
    sys.exit(1)

def look_file():
    with open("./ml-25m_new/user_5k/user_validdict.pkl", "rb") as f:
        temp = pickle.load(f)
    print(list(temp['1'].items())[:10])
    sys.exit(1)

def gen_userdict():
    user_validdict, user_testdict = {}, {}
    with open("./ml-25m_new2/user_5k/recbole/user_5k/user_5k_total/user_5k_total.valid.inter", "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            userid, itemid, rating, timestamp = lines[i].split("|")
            if userid not in user_validdict.keys():
                user_validdict[userid] = {}
            if float(rating) > 0:
                rating = 1
            else:
                rating = 0
            user_validdict[userid][itemid] = [itemid, rating]
            pbar.update()
    with open("./ml-25m_new2/user_5k/recbole/user_5k/user_5k_total/user_5k_total.test.inter", "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines) - 1) as pbar:
        for i in range(1, len(lines)):
            userid, itemid, rating, timestamp = lines[i].split("|")
            if userid not in user_testdict.keys():
                user_testdict[userid] = {}
            if float(rating) > 0:
                rating = 1
            else:
                rating = 0
            user_testdict[userid][itemid] = [itemid, rating]
            pbar.update()
    with open("./ml-25m_new2/user_5k/user_validdict.pkl", "wb") as f:
        pickle.dump(user_validdict, f)
    with open("./ml-25m_new2/user_5k/user_testdict.pkl", "wb") as f:
        pickle.dump(user_testdict, f)
    sys.exit(1)

if __name__ == "__main__":
    # resample_standard("ml-25m")
    # direct_userembed()
    # generate_potential("ml-25m")
    # gen_userdict()
    
    # look_file()
    # modify_potential()
    # check()
    # gen_new_embed_file()
    # gen_item_embedding()
    
    # gen_new_file()
    
    # condense_desciption()
    # clean_description()
    # analyze_cosratio2("ml-25m")
    
    # get_potential_profile_demo("amazon-CDs_and_Vinyl")
    
    
    # analyze_cosratio("ml-25m")
    # print("")
    # analyze_cosratio("amazon-CDs_and_Vinyl")
    
    # generate_user_pref("ml-25m", 11442)
    # sys.exit(1)
    
    # testing2()
    # cut_potential("ml-25m")
    # cut_potential("amazon-CDs_and_Vinyl")
    
    # sys.exit(1)
    
    # base_dict = {11434:0, 11435:1, 11438:2, 11439:3, 11441:4}
    
    gen_modify_userembed()
    
    # for dataset, cut_len in [("amazon-CDs_and_Vinyl", 30)]:
    # # for dataset, cut_len in [("ml-25m", 30)]:
    #     modify_user_profile(dataset, 11441, cut_len)
    
    # for dataset, cut_len in [("ml-25m", 20)]:
    #     modify_user_profile(dataset, 11441, cut_len)
    
    
    modify_user_profile("ml-25m", 11442, 30, 4)
    # temp()
    # analyze_modelembed()
    # generate_inputprompt("ml-25m")
    # generate_inputprompt("amazon-CDs_and_Vinyl")
    # gen_modify_userembed()
    # get_errorids()
    # modify_user_profile("ml-25m")
    # modify_user_profile("amazon-CDs_and_Vinyl")
    
    sys.exit(1)
    # modify_ndcg()
    # load_totalembed("amazon-CDs_and_Vinyl")
    # generate_BGE_embed("amazon-CDs_and_Vinyl")
    # generate_BGE_recbole("ml-25m")
    # generate_BGE_item_embed("ml-25m")
    # saving_user_totalprofiles("amazon-CDs_and_Vinyl")
    # resample("ml-25m")
    # sys.exit(1)
    
    
    test()
    # generate_potential("amazon")
    # combine_profile()

# curl https://api.jina.ai/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer YOUR_JINA_API_KEY" \
#   -d @- <<EOFEOF
#   {
#     "model": "jina-embeddings-v3",
#     "task": "text-matching",
#     "late_chunking": false,
#     "dimensions": 1024,
#     "embedding_type": "float",
#     "input": [
#         "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
#         "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille: Erleben Sie die wohltuende Wirkung unserer Bio-Hautpflege, speziell für empfindliche Haut entwickelt. Mit den beruhigenden Eigenschaften von Aloe Vera und Kamille pflegen und schützen unsere Produkte Ihre Haut auf natürliche Weise. Verabschieden Sie sich von Hautirritationen und genießen Sie einen strahlenden Teint.",
#         "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla: Descubre el poder de la naturaleza con nuestra línea de cuidado de la piel orgánico, diseñada especialmente para pieles sensibles. Enriquecidos con aloe vera y manzanilla, estos productos ofrecen una hidratación y protección suave. Despídete de las irritaciones y saluda a una piel radiante y saludable.",
#         "针对敏感肌专门设计的天然有机护肤产品：体验由芦荟和洋甘菊提取物带来的自然呵护。我们的护肤产品特别为敏感肌设计，温和滋润，保护您的肌肤不受刺激。让您的肌肤告别不适，迎来健康光彩。",
#         "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています: 今シーズンのメイクアップトレンドは、大胆な色彩と革新的な技術に注目しています。ネオンアイライナーからホログラフィックハイライターまで、クリエイティビティを解き放ち、毎回ユニークなルックを演出しましょう。"
#     ]
#   }
# EOFEOF