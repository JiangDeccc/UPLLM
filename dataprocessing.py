import pandas as pd
import numpy as np
import time
import os
import shutil
import json
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import sys
import copy
import pickle

# office数据集不合适，1正例1负例共有276129个user，只有92个满足training set可以sample50个。

def df_process(df, item_dict, neg_sample_size, pos_sample_size, not_sample_size, sample_ratio, train_size): # TODO: Office只有9个人打了19个以上的负例，先拿CDs去做，也不多，500多个，筛选性质太强了。
    def get_negative(lst, userid):
        temp1, temp2, temp3 = [], [], []
        for i in range(len(lst[0])):
            if (lst[0][i], 0.5) in user_itemcount[userid]:
                temp1.append((lst[0][i], 0.5))
            else:
                temp1.append((lst[0][i], 1.0))
        for i in range(len(lst[1])):
            if (lst[1][i], 1.5) in user_itemcount[userid]:
                temp2.append((lst[1][i], 1.5))
            else:
                temp2.append((lst[1][i], 2.0))
        for i in range(len(lst[2])):
            if (lst[2][i], 2.5) in user_itemcount[userid]:
                temp3.append((lst[2][i], 2.5))
            else:
                temp3.append((lst[2][i], 3.0))
        negative_list = temp1 + temp2 + temp3
        return negative_list
    
    # def get_positive(lst):
    #     temp4, temp5 = [], []
    #     for i in range(len(lst[3])):
    #         temp4.append((lst[3][i], 4))
    #     for i in range(len(lst[4])):
    #         temp5.append((lst[4][i], 5))
    #     positive_list = temp4 + temp5
    #     return positive_list
    
    user_totalcount = {}
    user_itemcount = {}
    user_count = {}
    target_user = []
    target_user_len = []
    total_count = np.zeros(5)
    neg_have_size = 3
    
    print("Reading")
    if (not os.path.exists("./amazon/user_totalcount.pkl")) and (not os.path.exists("./amazon/user_itemcount.pkl")):
    # if (not os.path.exists("./ml-25m/user_totalcount.pkl")) and (not os.path.exists("./ml-25m/user_itemcount.pkl")):
        with tqdm(total=df.shape[0]) as pbar: # 读取每个用户的交互历史
            for index, row in df.iterrows():
                if row['user_id'] not in user_totalcount:
                    user_totalcount[row['user_id']] = []
                if row['user_id'] not in user_itemcount:
                    user_itemcount[row['user_id']] = []
                if row['item_id'] in item_dict.keys() and (row['item_id'], row['rating']) not in user_itemcount[row['user_id']]:
                    user_totalcount[row['user_id']].append((row['item_id'], row['rating'], row['timestamp']))
                    user_itemcount[row['user_id']].append((row['item_id'], row['rating']))
                pbar.update()
        pbar.close()
        
        with open("./amazon/user_totalcount.pkl", "wb") as f:
            pickle.dump(user_totalcount, f)
        with open("./amazon/user_itemcount.pkl", "wb") as f:
            pickle.dump(user_itemcount, f)
        # with open("./ml-25m/user_totalcount.pkl", "wb") as f:
        #     pickle.dump(user_totalcount, f)
        # with open("./ml-25m/user_itemcount.pkl", "wb") as f:
        #     pickle.dump(user_itemcount, f)
    else:
        with open("./amazon/user_totalcount.pkl", "rb") as f:
            user_totalcount = pickle.load(f)
        with open("./amazon/user_itemcount.pkl", "rb") as f:
            user_itemcount = pickle.load(f)
        # with open("./ml-25m/user_totalcount.pkl", "rb") as f:
        #     user_totalcount = pickle.load(f)
        # with open("./ml-25m/user_itemcount.pkl", "rb") as f:
        #     user_itemcount = pickle.load(f)
    
    print("Sorting")
    with tqdm(total=len(user_totalcount)) as pbar: # 将每个用户的交互历史排序、按分数分类
        for user in user_totalcount.keys():
            user_totalcount[user] = sorted(user_totalcount[user], key=lambda x:x[2])
            if user not in user_count:
                user_count[user] = [[], [], [], [], [], []]
            for item in user_totalcount[user]:
                if item[1] > int(item[1]):
                    groupid = int(item[1])
                else:
                    groupid = int(item[1]) - 1
                user_count[user][groupid].append(item[0])
                user_count[user][5].append((item[0], item[1]))
                total_count[groupid] += 1
            pbar.update()
    pbar.close()

    # with open("./countlog.txt", "w") as f:
    #     f.write("{}\n\n".format(total_count))
    # print(total_count)
    # count_negative(user_count)
    # plot(user_count, total_count)
    # sys.exit(1)
    
    print("Satisfying")
    with tqdm(total=len(user_count)) as pbar: # 筛选出负样例符合条件的用户，并按照交互数量排序
        for user in user_count.keys():
            if len(user_count[user][0]) + len(user_count[user][1]) + len(user_count[user][2]) >= neg_have_size * 2 and \
            len(user_count[user][3]) + len(user_count[user][4]) >= pos_sample_size * 2:
                target_user.append(user)
                target_user_len.append(len(user_count[user][5]))
            pbar.update()
    pbar.close()
    print(len(user_totalcount), len(target_user))
    # time.sleep(1000)
    zipped_user = list(zip(target_user, target_user_len))
    zipped_user = sorted(zipped_user, key=lambda x:x[1])
    
    
    cut = -1
    for i in range(len(zipped_user)): # 筛选出有足够交互记录的用户
        if zipped_user[i][1] >= (pos_sample_size + neg_sample_size) * 2 + train_size:
            cut = i
            break
    zipped_user = zipped_user[cut:]
    print(zipped_user)
    print(len(zipped_user))
    
    # for i in range(5):
    #     print(zipped_user[len(zipped_user) * (i + 1) // 5 - 1])
    # print(zipped_user[1000], zipped_user[2000], zipped_user[3000], zipped_user[4000], zipped_user[5000])
    # time.sleep(1000)
    # ('A3LV2IOQUN652', 22) ('A23ZR1GBB8AW6E', 30) ('A2KYS0JR501ECX', 41) ('A24I3BMB9JZC7O', 64) ('A195PCNYWRB75', 125)
    
    # 将用户按照交互历史多少分为5档：[5, 10], (10, 20], (20, 30], (30, 40], (40, +\infin)
    # group_sizes = np.array([10, 20, 30, 40, 50])
    group_sizes = np.array([5, 10, 20, 35, 65])
    group_sizes += (pos_sample_size + neg_sample_size) * 2
    
    user_groups = [[], [], [], [], []]
    user_groups[0] = np.array([cur_user for cur_user in zipped_user if group_sizes[0] <= cur_user[1] <= group_sizes[1]])
    user_groups[1] = np.array([cur_user for cur_user in zipped_user if group_sizes[1] < cur_user[1] <= group_sizes[2]])
    user_groups[2] = np.array([cur_user for cur_user in zipped_user if group_sizes[2] < cur_user[1] <= group_sizes[3]])
    user_groups[3] = np.array([cur_user for cur_user in zipped_user if group_sizes[3] < cur_user[1] <= group_sizes[4]])
    user_groups[4] = np.array([cur_user for cur_user in zipped_user if group_sizes[4] < cur_user[1]])
    # rest_user = np.array([cur_user for cur_user in zipped_user if group_sizes[4] < cur_user[1]])
    
    # ml100k: 940 0 78 150 90 74 548
    # CDs: 44764 23530 11345 3812 1939 1053 3085
    # Office: 33224 24662 6420 1332 396 191 223
    
    # ml-25m: 148922 20822 14336 11741 9985 87589
    # print(user_groups[0][:40])
    
    print(len(user_groups[0]), len(user_groups[1]), len(user_groups[2]), len(user_groups[3]), len(user_groups[4]))
    # time.sleep(1000)
    
    # np.random.seed(1234)
    # np.random.shuffle(rest_user)
    
    # for i in range(5):
    #     cur_restuser = rest_user[round(rest_user.shape[0] * i / 5):round(rest_user.shape[0] * (i + 1) / 5)]
    #     # print(user_groups[i].shape, cur_restuser.shape)
    #     if user_groups[i].shape[0] == 0 and user_groups[i].ndim == 1:
    #         user_groups[i] = np.zeros((0, 2))
    #     user_groups[i] = np.concatenate([user_groups[i], cur_restuser], axis=0)
        
    # print(user_groups[0][:40])
    random_seeds = [1234, 5678, 9012, 3456, 7890]
    total_user = []
    
    for i in range(5):
        # print(user_groups[i][:40])
        np.random.seed(random_seeds[i])
        if i == 3:
            sample_num = 996
        elif i == 4:
            sample_num = 1004
        else:
            sample_num = 1000
        # sample_num = 1000
        temp = np.random.choice(a=user_groups[i].shape[0], size=sample_num, replace=False)
        user_groups[i] = user_groups[i][temp].tolist()
        for j in range(len(user_groups[i])):
            user_groups[i][j][1] = int(user_groups[i][j][1])
            if i == 4 and user_groups[i][j][1] > 80:
                # if i == 0:
                #     user_groups[i][j][1] = (train_size + 2 + group_sizes[i]) // 2
                # else:
                #     user_groups[i][j][1] = (group_sizes[i - 1] + group_sizes[i]) // 2
                user_groups[i][j][1] = 80
        # user_groups[i] = user_groups[i].tolist()
        print(user_groups[i])
        print("")
        total_user += user_groups[i]
    total_user = sorted(total_user, key=lambda x:x[0])
    
        
    # for i in range(len(total_user)):
    #     # total_user[i][0], total_user[i][1] = int(total_user[i][0]), int(total_user[i][1])
    #     total_user[i][1] = int(total_user[i][1])
    # print(len(total_user))
    # time.sleep(1000)
    # 将用户按照交互历史分为3档，在其中分别抽样，组成抽样的用户
    # cold_index = len(zipped_user) // 3
    # middle_index = len(zipped_user) * 2 // 3
    # target_user = []
    # for i in range(len(zipped_user)):
    #     target_user.append(zipped_user[i][0])
    # target_user = np.array(target_user)
    # cold_user, middle_user, warm_user = target_user[:cold_index], target_user[cold_index:middle_index], target_user[middle_index:]
    # print(len(cold_user), len(middle_user), len(warm_user))
    # time.sleep(1000)
    
    # np.random.seed(1234)
    # temp = np.random.choice(a=len(cold_user), size=sample_ratio['cold'], replace=False)
    # cold_user = cold_user[temp]
    # np.random.seed(5678)
    # temp = np.random.choice(a=len(middle_user), size=sample_ratio['middle'], replace=False)
    # middle_user = middle_user[temp]
    # np.random.seed(9012)
    # temp = np.random.choice(a=len(warm_user), size=sample_ratio['warm'], replace=False)
    # warm_user = warm_user[temp]
    # total_user = cold_user.tolist() + middle_user.tolist() + warm_user.tolist()
    # total_user.sort()
    # print(total_user)
    
    # total_user：抽样出的用户ID列表；sample_user：抽样出的数据集dict，key：用户ID；item：[负(itme_id, score)，正(item_id, score)]
    sample_user = {}
    
    # print(item_dict["B000001GP6"])
    # time.sleep(1000)
    
    with tqdm(total=len(total_user)) as pbar:
        for i in range(len(total_user)):
            sample_user[total_user[i][0]] = []
            item_dict2 = copy.deepcopy(item_dict)
            # print(item_dict2["B000001GP6"])
            # print("##### {}".format(i))
            negative_temp = get_negative(user_count[total_user[i][0]], total_user[i][0]) # [[], [], [], [], [], []]
            np.random.seed(1234)
            np.random.shuffle(negative_temp)
            np.random.seed(5678)
            temp = np.random.choice(a=len(negative_temp), size=neg_sample_size * 2, replace=False)
            negative_sample = []
            negative_sample2 = []
            negative_sample_ = []
            negative_sample2_ = []
            for j in range(temp.shape[0] // 2):
                negative_sample_.append(negative_temp[temp[j]])
            for j in range(temp.shape[0] // 2, temp.shape[0]):
                negative_sample2_.append(negative_temp[temp[j]])
            # positive_temp = get_positive(user_count[total_user[i]])
            # np.random.seed(9012)
            # np.random.shuffle(positive_temp)
            # np.random.seed(3456)
            
            # temp = np.random.choice(a=len(positive_temp), size=pos_sample_size, replace=False)
            # positive_sample = []
            # for j in range(temp.shape[0]):
            #     positive_sample.append(positive_temp[temp[j]])
            # print(item_dict)
            interactions = user_count[total_user[i][0]][5]
            # print(interactions)
            # try:
            # print(total_user[i])
            # print(interactions)
            for j in range(len(interactions)):
                # print(interactions[j])
                item_dict2.pop(interactions[j][0])
            # except:
            #     # print(item_dict)
            #     print("")
            #     print(interactions)
            #     sys.exit(1)
            item_list = list(item_dict2.items())
            
            # print(interactions)
            # print(negative_sample)
            # time.sleep(1000)
            
            for j in range(len(interactions) - 1, -1, -1):
                if interactions[j] in negative_sample_ or interactions[j] in negative_sample2_:
                    interactions.pop(j)
            
            count = 0
            for j in range(len(interactions) - 1, -1, -1):
                # print(j, interactions[j])
                if interactions[j][1] >= 4:
                    if count < pos_sample_size:
                        negative_sample.append(interactions[j])
                    else:
                        negative_sample2.append(interactions[j])
                    if count == 0:
                        interactions = interactions[:j] # 每一块的人数不为40，问题出在此处
                    else:
                        interactions.pop(j)
                    count += 1
                    if count == pos_sample_size * 2:
                        break
            
            negative_sample += negative_sample_
            negative_sample2 += negative_sample2_
            
            # not_temp = np.arange(len(item_list))
            np.random.seed(7890)
            not_temp = np.random.choice(a=len(item_list), size=not_sample_size * 2, replace=False)

            for j in range(not_temp.shape[0] // 2):
                negative_sample.append((item_list[not_temp[j]][0], 0.0))
            for j in range(not_temp.shape[0] // 2, not_temp.shape[0]):
                negative_sample2.append((item_list[not_temp[j]][0], 0.0))
            
            interactions = interactions[-total_user[i][1]:]
            interactions.reverse()

            sample_user[total_user[i][0]].append(negative_sample)
            sample_user[total_user[i][0]].append(negative_sample2)
            sample_user[total_user[i][0]].append(interactions)
            pbar.update()
    pbar.close()
    
    count = np.zeros(5)
    for key in sample_user.keys():
        if len(sample_user[key][2]) <= 10:
            count[0] += 1
        elif 10 < len(sample_user[key][2]) <= 20:
            count[1] += 1
        elif 20 < len(sample_user[key][2]) <= 35:
            count[2] += 1
        elif 35 < len(sample_user[key][2]) <= 65:
            count[3] += 1
        elif 65 < len(sample_user[key][2]):
            count[4] += 1
        # elif len(sample_user[key][2]) < 10:
        #     print(len(sample_user[key][2]))
    print(count)
    # time.sleep(1000)
    
    for i in range(len(total_user)):
        total_user[i] = total_user[i][0]
    return total_user, sample_user

def movielens_100k():
    def list2sentence(lst):
        sentence = ""
        for i in range(len(lst)):
            sentence += lst[i]
            if i != len(lst) - 1:
                sentence += ", "
            else:
                sentence += "."
        return sentence
    
    neg_sample_size = 3
    pos_sample_size = 3
    not_sample_size = 14
    sample_ratio = {'cold': 33, 'middle': 34, 'warm': 33}
    train_size = 10
    
    df = pd.read_csv("./ml100k/u.data", header=None, sep='\t', quoting=3)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    print(df)
    
    item_meta = pd.read_csv("./ml100k/u.item", header=None, sep='|', encoding='latin-1')
    item_meta.columns = ['item_id','item_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    
    item_dict = {}
    filter_list = ['item_id','item_title','release_date','video_release_date','IMDb_URL']
    category_list = []
    description_list = []
    # item_meta.columns = ['item_id', 'item_title', 'item_category', 'item_description']
    for index, row in item_meta.iterrows():
        row['release_date'] = "Release date: " + str(row['release_date'])
        temp_row = dict(row)
        temp_cate = []
        for key in temp_row.keys():
            if key not in filter_list and temp_row[key] == 1:
                temp_cate.append(key)
        temp_cate = list2sentence(temp_cate)
        item_dict[row['item_id']] = [row['item_title'], temp_cate, row['release_date']]
        category_list.append(temp_cate)
        description_list.append(row['release_date'])
    
    item_meta["item_category"] = category_list
    item_meta['item_description'] = description_list
    for col in item_meta.columns:
        if col not in ['item_id', 'item_title', 'item_category', 'item_description']:
            del item_meta[col]
    
    total_user, sample_user = df_process(df, item_dict, neg_sample_size, pos_sample_size, not_sample_size, sample_ratio, train_size)
    
    with open("./ml100k/u.user", "r", encoding="latin-1") as f:
        user_metalist = f.read().splitlines()
    if not os.path.exists("./ml100k/user_200_3"):
        os.mkdir("./ml100k/user_200_3")
    with open("./ml100k/user_200_3/u.user", "w", encoding="latin-1") as f:
        for i in range(len(total_user)):
            f.write("{}\n".format(user_metalist[total_user[i] - 1]))
            # if i != len(total_user) - 1:
            #     f.write("\n")
            # else:
            #     f.write("\x1a")
    # shutil.copyfile("./ml100k/u.item", "./ml100k/user_100/u.item")
    interacted_item = []
    with open("./ml100k/user_200_3/u_train.data", "w", encoding="latin-1") as f:
        with open("./ml100k/user_200_3/u_test.data", "w", encoding="latin-1") as g:
            with open("./ml100k/user_200_3/u_valid.data", "w", encoding="latin-1") as h:
                for user in sample_user.keys():
                    for i in range(len(sample_user[user][0])):
                        g.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][0][i][0], sample_user[user][0][i][1], 0))
                        if sample_user[user][0][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][0][i][0])
                    for i in range(len(sample_user[user][1])):
                        h.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][1][i][0], sample_user[user][1][i][1], 0))
                        if sample_user[user][1][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][1][i][0])
                    for i in range(len(sample_user[user][2])):
                        f.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][2][i][0], sample_user[user][2][i][1], 0))
                        if sample_user[user][2][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][2][i][0])
    
    with open("./ml100k/user_200_3/u.item", "w", encoding="latin-1") as f:
        for index, row in item_meta.iterrows(): # 先按时间排序
            if row['item_id'] in interacted_item:
                f.write("{}|{}|{}|{}\n".format(row['item_id'], row['item_title'], row['item_category'], row['item_description']))
    
def amazon_json(file_type):
    def list2sentence(lst):
        sentence = ""
        for i in range(len(lst)):
            sentence += lst[i]
            sentence += ". "
        return sentence
    
    def special(lst):
        lst = lst.replace("\n", " ")
        lst = lst.replace("\r", " ")
        lst = lst.replace(" | ", " ")
        lst = lst.replace("|", " ")
        lst = lst.replace("  ", " ")
        return lst
    
    json_file = "./amazon/meta_{}.json".format(file_type)
    with open(json_file, "r", encoding="utf-8") as f:
        meta_list = f.read().splitlines()
    
    itemid_dict = {}
    with open("./amazon/{}.item".format(file_type), "w", encoding="latin-1") as f:
        for i in trange(len(meta_list)):
            json_line = json.loads(meta_list[i])
            item_id = json_line['asin']
            if item_id not in itemid_dict.keys():
                itemid_dict[item_id] = 0
                item_title = json_line['title']
                item_title = special(item_title)
                item_category = json_line['category'][1:]
                item_category.append(json_line['brand'])
                item_description = json_line['feature'] + json_line['description']
                item_category = [x for x in item_category if x]
                item_description = [x for x in item_description if x]
                item_category = list2sentence(item_category)
                item_description = list2sentence(item_description)
                item_category = special(item_category)
                item_description = special(item_description)
                f.write("{}|{}|{}|{}\n".format(item_id, item_title, item_category, item_description))
            # if i != len(meta_list) - 1:
            #     f.write("\n")
            # else:
            #     f.write("\x1a")

def amazon_process(df, item_dict):
    del_index = []
    print("Checking")
    user_interacted = {}
    with tqdm(total=df.shape[0]) as pbar:
        for index, row in df.iterrows():
            # print(index, row['item_id'])
            if row['item_id'] not in item_dict.keys():
                del_index.append(index)
            elif item_dict[row['item_id']][0].startswith("<span"):
                # print(index, item_dict[row['item_id']][0])
                del item_dict[row['item_id']]
                del_index.append(index)
            
            if row['user_id'] not in user_interacted.keys():
                user_interacted[row['user_id']] = {}
            if row['item_id'] in user_interacted[row['user_id']].keys():
                del_index.append(index)
            else:
                user_interacted[row['user_id']][row['item_id']] = 0
            pbar.update()
    pbar.close()
    df.drop(del_index, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, item_dict

def amazon_csv(file):
    file_type = file.split("/")[2].split(".")[0] # CDs_and_Vinyl / Office_Products
    if not os.path.exists("./amazon/{}.item".format(file_type)):
        amazon_json(file_type)
    
    item_meta = pd.read_csv("./amazon/{}.item".format(file_type), header=None, sep='|', encoding='latin-1', quoting=3) # must quoting=3
    item_meta.columns = ['item_id', 'item_title', 'item_category', 'item_description']
    
    item_meta['item_id'] = item_meta["item_id"].astype(str)
    item_meta['item_title'] = item_meta["item_title"].astype(str)
    item_meta['item_category'] = item_meta["item_category"].astype(str)
    item_meta['item_description'] = item_meta["item_description"].astype(str)
    
    item_dict = {}
    for index, row in item_meta.iterrows():
        empty = row.isna()
        if empty['item_title']:
            continue
        item_dict[row['item_id']] = [row['item_title'], row['item_category'], row['item_description']]
    
    # print(item_dict["B000001GP6"])
    # time.sleep(1000)
    
    neg_sample_size = 3
    pos_sample_size = 3
    not_sample_size = 14
    sample_ratio = {'cold': 33, 'middle': 34, 'warm': 33}
    train_size = 5
    
    df = pd.read_csv(file, header=None, sep=',', quoting=3)
    df.columns = ['item_id', 'user_id', 'rating', 'timestamp']
    print(df)
    
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    
    if (not os.path.exists("./amazon/user_totalcount.pkl")) and (not os.path.exists("./amazon/user_itemcount.pkl")):
        df, item_dict = amazon_process(df, item_dict)
        print(df)
    
    total_user, sample_user = df_process(df, item_dict, neg_sample_size, pos_sample_size, not_sample_size, sample_ratio, train_size)
    
    if not os.path.exists("./amazon/user_5k"):
        os.mkdir("./amazon/user_5k")
    if not os.path.exists("./amazon/user_5k/{}".format(file_type)):
        os.mkdir("./amazon/user_5k/{}".format(file_type))
    
    with open("./amazon/user_5k/{}/u.user".format(file_type), "w", encoding="latin-1") as f:
        for i in range(len(total_user)):
            if file_type == "CDs_and_Vinyl":
                f.write("{}|I enjoy listening to music very much.\n".format(total_user[i]))
            elif file_type == "Office_Products":
                f.write("{}|I am interested in office products.\n".format(total_user[i]))
            # if i != len(total_user) - 1:
            #     f.write("\n")
            # else:
            #     f.write("\x1a")
    
    interacted_item = []
    with open("./amazon/user_5k/{}/u_train.data".format(file_type), "w", encoding="latin-1") as f:
        with open("./amazon/user_5k/{}/u_test.data".format(file_type), "w", encoding="latin-1") as g:
            with open("./amazon/user_5k/{}/u_valid.data".format(file_type), "w", encoding="latin-1") as h:
                for user in sample_user.keys():
                    for i in range(len(sample_user[user][0])):
                        g.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][0][i][0], sample_user[user][0][i][1], 0))
                        if sample_user[user][0][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][0][i][0])
                    for i in range(len(sample_user[user][1])):
                        h.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][1][i][0], sample_user[user][1][i][1], 0))
                        if sample_user[user][1][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][1][i][0])
                    for i in range(len(sample_user[user][2])):
                        f.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][2][i][0], sample_user[user][2][i][1], 0))
                        if sample_user[user][2][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][2][i][0])
    
    with open("./amazon/user_5k/{}/u.item".format(file_type), "w", encoding="latin-1") as f:
        with tqdm(total=item_meta.shape[0]) as pbar:
            for index, row in item_meta.iterrows(): # 先按时间排序
                if row['item_id'] in interacted_item:
                    f.write("{}|{}|{}|{}\n".format(row['item_id'], row['item_title'], row['item_category'], row['item_description']))
                    # if index != df.shape[0] - 1:
                    #     f.write("\n")
                    # else:
                    #     f.write("\x1a")
                pbar.update()
        pbar.close()

def check_dup(path):
    train_dict, valid_dict, test_dict = {}, {}, {}
    with open("{}/u_train.data".format(path), "r", encoding="latin-1") as f:
        train_lines = f.read().splitlines()
        for i in range(len(train_lines)):
            userid, itemid, rating, timestamp = train_lines[i].split("\t")
            if userid not in train_dict.keys():
                train_dict[userid] = []
            train_dict[userid].append(itemid)
    with open("{}/u_test.data".format(path), "r", encoding="latin-1") as f:
        test_lines = f.read().splitlines()
        for i in range(len(test_lines)):
            userid, itemid, rating, timestamp = test_lines[i].split("\t")
            if userid not in test_dict.keys():
                test_dict[userid] = []
            test_dict[userid].append(itemid)
    for key in train_dict.keys():
        set1 = set(train_dict[key])
        set2 = set(test_dict[key])
        inter1 = set1.intersection(set2)
        if len(inter1) != 0:
            print(key)

def preprocess_ml25m():
    def special(lst):
        lst = lst.replace("\n", " ")
        lst = lst.replace("\r", " ")
        lst = lst.replace(" | ", " ")
        lst = lst.replace("|", " ")
        lst = lst.replace("  ", " ")
        return lst
    
    # process genome tags
    genome_tags_dict, scores_dict, item_tags_dict = {}, {}, {}
    genome_tags_df = pd.read_csv("./ml-25m/genome-tags.csv", header=0) # tagId,tag
    for index, row in genome_tags_df.iterrows():
        genome_tags_dict[row["tagId"]] = row["tag"]
    genome_scores_df = pd.read_csv("./ml-25m/genome-scores.csv", header=0) # movieId,tagId,relevance
    if not os.path.exists("./ml-25m/temp_tags_dict.pkl"):
        with tqdm(total=genome_scores_df.shape[0]) as pbar:
            for index, row in genome_scores_df.iterrows():
                # print(row["movieId"], type(row["movieId"]))
                # time.sleep(0.1)
                if row["movieId"] not in scores_dict.keys():
                    scores_dict[row["movieId"]] = []
                if row["relevance"] > 0.7:
                    scores_dict[row["movieId"]].append([row["movieId"], row["tagId"], row["relevance"]])
                # if row["movieId"] > 123:
                #     break
                pbar.update()
        with tqdm(total=len(scores_dict)) as pbar:
            for key in scores_dict.keys():
                scores_dict[key] = sorted(scores_dict[key], key=lambda x:x[2], reverse=True)
                scores_dict[key] = scores_dict[key][:50]
                item_tags_dict[key] = []
                for i in range(len(scores_dict[key])):
                    cur_tag = scores_dict[key][i][1]
                    cur_tag = genome_tags_dict[cur_tag]
                    item_tags_dict[key].append(cur_tag)
                # if key == 123:
                #     print(scores_dict[key])
                #     print(item_tags_dict[key])
                #     time.sleep(1000)
                pbar.update()
        with open("./ml-25m/temp_tags_dict.pkl", "wb") as f:
            pickle.dump(item_tags_dict, f)
    else:
        print("Load cached tag dict.")
        with open("./ml-25m/temp_tags_dict.pkl", "rb") as f:
            item_tags_dict = pickle.load(f)
    
    # process tags
    tags_df = pd.read_csv("./ml-25m/tags.csv", header=0) # userId,movieId,tag,timestamp
    
    with tqdm(total=tags_df.shape[0]) as pbar:
        for index, row in tags_df.iterrows():
            if pd.isna(row["userId"]) or pd.isna(row["movieId"]) or pd.isna(row["tag"]) or pd.isna(row["timestamp"]):
                continue
            # if row["movieId"] == 123:
            #     print(row)
                # print(item_tags_dict[row["movieId"]])
            if row["movieId"] not in item_tags_dict.keys():
                item_tags_dict[row["movieId"]] = []
            item_tags_dict[row["movieId"]].append(row["tag"])
            pbar.update()
    
    # print(item_tags_dict[123])
    
    # process other files in the format of amazon
    movies_df = pd.read_csv("./ml-25m/movies.csv", header=0) # movieId,title,genres
    with tqdm(total=movies_df.shape[0]) as pbar:
        with open("./ml-25m/ml-25m.item", "w") as f:
            for index, row in movies_df.iterrows():
                f.write("{}|".format(row["movieId"]))
                if not pd.isna(row["title"]):
                    cur_title = special(row["title"])
                    f.write("{}|".format(cur_title))
                else:
                    cur_title = None
                    f.write("|")
                if not pd.isna(row["genres"]):
                    cur_genres = row["genres"].split("|")
                    for i in range(len(cur_genres)):
                        f.write("{}. ".format(cur_genres[i]))
                    f.write("|")
                else:
                    cur_genres = None
                    f.write("|")
                if row["movieId"] in item_tags_dict.keys():
                    cur_taglist = item_tags_dict[row["movieId"]]
                    cur_taglist = list(set(cur_taglist))
                    # print("**** {}, {}, {}".format(row["movieId"], row["title"], cur_taglist))
                    for cur_tag in cur_taglist:
                        # print(cur_tag)
                        cur_tag = special(cur_tag)
                        f.write("{}. ".format(cur_tag))
                f.write("\n")
                pbar.update()
    
def movielens_25m():
    if not os.path.exists("./ml-25m/ml-25m.item"):
        preprocess_ml25m()
    item_meta = pd.read_csv("./ml-25m/ml-25m.item", header=None, sep='|', encoding='latin-1', quoting=3) # must quoting=3
    item_meta.columns = ['item_id', 'item_title', 'item_category', 'item_description']
    item_meta['item_id'] = item_meta["item_id"].astype(str)
    item_meta['item_title'] = item_meta["item_title"].astype(str)
    item_meta['item_category'] = item_meta["item_category"].astype(str)
    item_meta['item_description'] = item_meta["item_description"].astype(str)
    
    item_dict = {}
    for index, row in item_meta.iterrows():
        empty = row.isna()
        if empty['item_title']:
            continue
        # row['item_id'] = int(row["item_id"])
        item_dict[row['item_id']] = [row['item_title'], row['item_category'], row['item_description']]
    
    neg_sample_size = 3
    pos_sample_size = 3
    not_sample_size = 14
    sample_ratio = None
    train_size = 5
    
    df = pd.read_csv("./ml-25m/ratings.csv", header=0, sep=',', quoting=3)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    print(df)
    
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    
    del_index = []
    if (not os.path.exists("./ml-25m/user_totalcount.pkl")) and (not os.path.exists("./ml-25m/user_itemcount.pkl")):
        with tqdm(total=df.shape[0]) as pbar:
            for index, row in df.iterrows():
                # print(index, row['item_id'])
                if row['item_id'] not in item_dict.keys():
                    del_index.append(index)
                pbar.update()
        pbar.close()
        df.drop(del_index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df)
    
    total_user, sample_user = df_process(df, item_dict, neg_sample_size, pos_sample_size, not_sample_size, sample_ratio, train_size)
    
    if not os.path.exists("./ml-25m/user_5k"):
        os.mkdir("./ml-25m/user_5k")
    
    with open("./ml-25m/user_5k/u.user", "w", encoding="latin-1") as f:
        for i in range(len(total_user)):
            f.write("{}|I enjoy watching movies very much.\n".format(total_user[i]))
    
    interacted_item = []
    with open("./ml-25m/user_5k/u_train.data", "w", encoding="latin-1") as f:
        with open("./ml-25m/user_5k/u_test.data", "w", encoding="latin-1") as g:
            with open("./ml-25m/user_5k/u_valid.data", "w", encoding="latin-1") as h:
                for user in sample_user.keys():
                    for i in range(len(sample_user[user][0])):
                        g.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][0][i][0], sample_user[user][0][i][1], 0))
                        if sample_user[user][0][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][0][i][0])
                    for i in range(len(sample_user[user][1])):
                        h.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][1][i][0], sample_user[user][1][i][1], 0))
                        if sample_user[user][1][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][1][i][0])
                    for i in range(len(sample_user[user][2])):
                        f.write("{}\t{}\t{}\t{}\n".format(user, sample_user[user][2][i][0], sample_user[user][2][i][1], 0))
                        if sample_user[user][2][i][0] not in interacted_item:
                            interacted_item.append(sample_user[user][2][i][0])
    
    with open("./ml-25m/user_5k/u.item", "w", encoding="latin-1") as f:
        with tqdm(total=item_meta.shape[0]) as pbar:
            for index, row in item_meta.iterrows(): # 先按时间排序
                if row['item_id'] in interacted_item:
                    f.write("{}|{}|{}|{}\n".format(row['item_id'], row['item_title'], row['item_category'], row['item_description']))
                pbar.update()
        pbar.close()

def check_number():
    with open("./amazon/user_5k/CDs_and_Vinyl/u_train.data", "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    sample_user = {}
    for i in range(len(lines)):
        user_id, _, _, _ = lines[i].split("\t")
        if user_id not in sample_user.keys():
            sample_user[user_id] = 0
        sample_user[user_id] += 1
    count = np.zeros(5)
    for key in sample_user.keys():
        if sample_user[key] <= 10:
            count[0] += 1
        elif 10 < sample_user[key] <= 20:
            count[1] += 1
        elif 20 < sample_user[key] <= 35:
            count[2] += 1
        elif 35 < sample_user[key] <= 65:
            count[3] += 1
        elif 65 < sample_user[key]:
            count[4] += 1
        # elif len(sample_user[key][2]) < 10:
        #     print(len(sample_user[key][2]))
    print(count)

def adjust_file(path):
    with open("{}/u.user".format(path), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    if not os.path.exists("{}_new".format(path)):
        os.mkdir("{}_new".format(path))
    with open("{}_new/u.user".format(path), "w", encoding="latin-1") as f:
        with tqdm(total=len(lines)) as pbar:
            for i in range(len(lines)):
                user_id, _ = lines[i].split("|", maxsplit=1)
                user_id = int(float(user_id))
                f.write("{}|{}\n".format(user_id, _))
                pbar.update()
    for file in ["train", "valid", "test"]:
        with open("{}/u_{}.data".format(path, file), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        with open("{}_new/u_{}.data".format(path, file), "w", encoding="latin-1") as f:
            with tqdm(total=len(lines)) as pbar:
                for i in range(len(lines)):
                    user_id, item_id, _ = lines[i].split("\t", maxsplit=2)
                    user_id = int(float(user_id))
                    item_id = int(float(user_id))
                    f.write("{}\t{}\t{}\n".format(user_id, item_id, _))
                    pbar.update()
    shutil.copy("{}/u.item".format(path), "{}_new/u.item".format(path))
    
if __name__ == "__main__":
    # check_number()
    # movielens_100k()
    amazon_csv("./amazon/CDs_and_Vinyl.csv")
    # amazon_csv("./amazon/Office_Products.csv")
    # amazon_csv("./amazon/All_Beauty.csv")
    # check_dup("./ml100k/user_200_testonly")
    
    # preprocess_ml25m()
    # movielens_25m()
    # adjust_file("./ml-25m/user_5k")
    
    """
    negative-3:
    ml100k: 885 - 144 83 70 62 484
    CDs: 6400 - 1336 863 519 385 1738
    Office: 1416 - 365 132 75 52 84
    """
