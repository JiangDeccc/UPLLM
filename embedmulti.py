import pandas as pd
import numpy as np
import openai
import pickle
from scipy import spatial
import time

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

def run(dataset):
    if dataset == "ml100k":
        path = "./ml100k/user_200_3"
        embed_path = "./Embeddings/ml100k/"
    elif "amazon" in dataset:
        datatype = dataset.split("-")[1]
        path = "./amazon/user_200_3/{}".format(datatype)
        embed_path = "./Embeddings/amazon/{}".format(datatype)
    
    test_df = pd.read_csv("{}/u_test.data".format(path), header=None, sep='\t', encoding='latin-1')
    test_df.columns = ["user_id", "item_id", "rating", "timestamp"]

    user_dict = {}
    for index, row in test_df.iterrows():
        if row["user_id"] not in user_dict.keys():
            user_dict[row["user_id"]] = []
        user_dict[row["user_id"]].append((row["item_id"], row["rating"]))
    with open("{}/u.item".format(path), "r", encoding="latin-1") as f:
        item_ids = f.read().splitlines()
        for i in range(len(item_ids)):
            item_ids[i] = item_ids[i].split("|")[0]
            if dataset == "ml100k":
                item_ids[i] = int(item_ids[i])
    user_ids = list(user_dict.keys())
    # print(item_ids)
    
    with open("{}/user_totalembed.pkl".format(embed_path), "rb") as f:
        user_totalembed = pickle.load(f)
    with open("{}/item_totalembed.pkl".format(embed_path), "rb") as f:
        item_totalembed = pickle.load(f)
    
    # total_ndcg, total_mrr = 0.0, 0.0
    total_ndcg = {1: np.zeros(6), 5: np.zeros(6), 10: np.zeros(6), 20: np.zeros(6)}
    total_mrr = {1: np.zeros(6), 5: np.zeros(6), 10: np.zeros(6), 20: np.zeros(6)}
    total_bpref = {1: np.zeros(6), 5: np.zeros(6), 10: np.zeros(6), 20: np.zeros(6)}
    group_count = np.zeros(6)
    eval_length = [1, 5, 10, 20]
    if dataset == "ml100k":
        recbole_path = "./ml100k/user_200_3/recbole/user_200"
    elif "amazon" in dataset:
        recbole_path = "./amazon/user_200_3/{}/recbole/user_200".format(datatype)
    usergroup_dict = {}
    for i in range(5):
        with open("{}/user_200_{}/user_200_{}.user".format(recbole_path, i, i), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        for j in range(1, len(lines)):
            if dataset == "ml100k":
                cur_userid, _, _, _, _ = lines[j].split("|")
            elif "amazon" in dataset:
                cur_userid, _ = lines[j].split("|")
            usergroup_dict[cur_userid] = i + 1
    for key in user_dict.keys():
        cur_userindex = user_ids.index(key)
        cur_userembed = user_totalembed[cur_userindex]
        cosine_list = []
        for i in range(len(user_dict[key])):
            cur_itemindex = item_ids.index(user_dict[key][i][0])
            cur_itemembed = item_totalembed[cur_itemindex]
            cur_cos = 1 - spatial.distance.cosine(cur_userembed, cur_itemembed)

            if user_dict[key][i][1] >= 4:
                cosine_list.append((cur_cos, 1))
            elif 0 < user_dict[key][i][1] <= 3:
                cosine_list.append((cur_cos, -1))
            else:
                cosine_list.append((cur_cos, 0))
        cosine_list = sorted(cosine_list, key=lambda x:x[0], reverse=True)
        for cur_length in eval_length:
            cur_coslist = cosine_list[:cur_length]
            cur_ndcg, cur_mrr = get_ndcg_mrr(cur_coslist)
            cur_bpref = get_bpref(cur_coslist)
            total_ndcg[cur_length][0] += cur_ndcg
            total_mrr[cur_length][0] += cur_mrr
            total_bpref[cur_length][0] += cur_bpref
            total_ndcg[cur_length][usergroup_dict[str(key)]] += cur_ndcg
            total_mrr[cur_length][usergroup_dict[str(key)]] += cur_mrr
            total_bpref[cur_length][usergroup_dict[str(key)]] += cur_bpref
        group_count[0] += 1
        group_count[usergroup_dict[str(key)]] += 1

        # for i in range(len(cosine_list)):
        #     if cosine_list[i][1] >= 4:
        #         total_ndcg[0] += float(1 / np.log2(i + 2))
        #         total_mrr[0] += 1 / (i + 1)
        #         group_count[0] += 1
        #         total_ndcg[usergroup_dict[str(key)]] += float(1 / np.log2(i + 2))
        #         total_mrr[usergroup_dict[str(key)]] += 1 / (i + 1)
        #         group_count[usergroup_dict[str(key)]] += 1
        #         break
    
    for i in range(6):
        for cur_length in eval_length:
            total_ndcg[cur_length][i] /= group_count[i]
            total_mrr[cur_length][i] /= group_count[i]
            total_bpref[cur_length][i] /= group_count[i]
    
    for cur_length in eval_length:
        print("{}:".format(cur_length))
        print("ndcg: {}".format(total_ndcg[cur_length]))
        print("mrr: {}".format(total_mrr[cur_length]))
        print("bpref: {}".format(total_bpref[cur_length]))

if __name__ == "__main__":
    # run("ml100k")
    run("amazon-CDs_and_Vinyl")
    # run("amazon-Office_Products")