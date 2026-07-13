from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
import os
import shutil
import time
import pickle

def create(dataset):
    def read_split_data(data_type): # train, valid, test
        print("Loading {}...".format(data_type))
        with open("{}/u_{}.data".format(path, data_type), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        with tqdm(total=len(lines), desc='Processing {}...'.format(data_type)) as pbar:
            for i in range(len(lines)):
                cur_userid, cur_itemid, cur_rating, _ = lines[i].split("\t")
                cur_rating = float(cur_rating)
                if cur_userid not in user_positive_dict.keys():
                    user_positive_dict[cur_userid] = []
                if cur_userid not in user_negative_dict.keys():
                    user_negative_dict[cur_userid] = []
                if cur_rating >= 4.0:
                    cur_timestamp = timestamp_dict[cur_userid][cur_itemid]
                    user_positive_dict[cur_userid].append([cur_itemid, cur_rating, cur_timestamp])
                elif 0 < cur_rating <= 3.5:
                    cur_timestamp = timestamp_dict[cur_userid][cur_itemid]
                    user_negative_dict[cur_userid].append([cur_itemid, cur_rating, cur_timestamp])
                pbar.update()
    
    def save_split_data(data_type, user_type_dict): # train, valid, test
        with open("{}/u_{}.data".format(new_path, data_type), "w") as f:
            with tqdm(total=len(user_type_dict), desc="Saving {}...".format(data_type)) as pbar1:
                for cur_userid in user_type_dict.keys():
                    # with tqdm(total=len(user_type_dict[cur_userid], desc="Processing interaction list...")) as pbar2:
                    for i in range(len(user_type_dict[cur_userid])):
                        cur_itemid, cur_rating, cur_timestamp = user_type_dict[cur_userid][i]
                        f.write("{}\t{}\t{}\t{}\n".format(cur_userid, cur_itemid, cur_rating, cur_timestamp))
                            # pbar2.update()
                    pbar1.update()

    timestamp_dict = {}
    if dataset == "ml-25m":
        path = "./{}/user_5k".format(dataset)
        print("Loading ratings...")
        with open("{}/ratings.csv".format(path), "r") as f:
            lines = f.read().splitlines()
        with tqdm(total=len(lines) - 1, desc='Processing original dataset...') as pbar:
            for i in range(1, len(lines)):
                cur_userid, cur_itemid, _, cur_timestamp = lines[i].split(",")
                cur_timestamp = int(cur_timestamp)
                if cur_userid not in timestamp_dict.keys():
                    timestamp_dict[cur_userid] = {}
                timestamp_dict[cur_userid][cur_itemid] = cur_timestamp
                pbar.update()
                
    elif dataset == "amazon-CDs_and_Vinyl":
        temp = dataset.split("-")
        path = "./{}/user_5k/{}".format(temp[0], temp[1])
        print("Loading ratings...")
        with open("{}/ratings.csv".format(path), "r") as f:
            lines = f.read().splitlines()
        with tqdm(total=len(lines), desc='Processing original dataset...') as pbar:
            for i in range(len(lines)):
                cur_itemid, cur_userid, _, cur_timestamp = lines[i].split(",")
                cur_timestamp = int(cur_timestamp)
                if cur_userid not in timestamp_dict.keys():
                    timestamp_dict[cur_userid] = {}
                timestamp_dict[cur_userid][cur_itemid] = cur_timestamp
                pbar.update()
    
    user_positive_dict, user_negative_dict = {}, {}
    read_split_data("train")
    read_split_data("valid")
    read_split_data("test")
    
    with tqdm(total=len(user_positive_dict), desc='Sorting positive items...') as pbar:
        for cur_userid in user_positive_dict.keys():
            user_positive_dict[cur_userid] = sorted(user_positive_dict[cur_userid], key=lambda x:x[2], reverse=True)
            pbar.update()
    with tqdm(total=len(user_negative_dict), desc='Sorting negative items...') as pbar:
        for cur_userid in user_negative_dict.keys():
            user_negative_dict[cur_userid] = sorted(user_negative_dict[cur_userid], key=lambda x:x[2], reverse=True)
            pbar.update()
    
    split_ratio = {'validtest': 0.1}
    user_train_dict, user_valid_dict, user_test_dict = {}, {}, {}
    with tqdm(total=len(user_positive_dict), desc="Processing positive items in validation and test set...") as pbar:
        for cur_userid in user_positive_dict.keys():
            validtest_num = int(Decimal(str(len(user_positive_dict[cur_userid]) * split_ratio['validtest'])).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            if validtest_num == 0 and len(user_positive_dict) > 0:
                validtest_num = 1
            if cur_userid not in user_train_dict.keys():
                user_train_dict[cur_userid] = []
                user_valid_dict[cur_userid] = []
                user_test_dict[cur_userid] = []
            user_test_dict[cur_userid].extend(user_positive_dict[cur_userid][:validtest_num])
            user_valid_dict[cur_userid].extend(user_positive_dict[cur_userid][validtest_num:validtest_num * 2])
            user_train_dict[cur_userid].extend(user_positive_dict[cur_userid][validtest_num * 2:])
            user_train_dict[cur_userid].extend(user_negative_dict[cur_userid])
            user_train_dict[cur_userid] = sorted(user_train_dict[cur_userid], key=lambda x:x[2], reverse=True)
            pbar.update()
    
    print("Loading items...")
    with open("{}/u.item".format(path), "r") as f:
        lines = f.read().splitlines()
    item_set = set()
    with tqdm(total=len(lines), desc="Creating item set...") as pbar:
        for i in range(len(lines)):
            cur_itemid, _ = lines[i].split("|", maxsplit=1)
            item_set.add(cur_itemid)
            pbar.update()
    with tqdm(total=len(user_valid_dict), desc="Processing unseen items in validation and test set...") as pbar:
        for cur_userid in user_valid_dict.keys():
            for cur_itemid in item_set:
                if cur_itemid not in timestamp_dict[cur_userid].keys():
                    user_valid_dict[cur_userid].append([cur_itemid, 0, 0])
                    user_test_dict[cur_userid].append([cur_itemid, 0, 0])
            pbar.update()
    
    path_list = path.split("/")
    path_list[1] += "_new"
    new_path = os.path.join(*path_list)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.copy("{}/u.user".format(path), "{}/u.user".format(new_path))
    shutil.copy("{}/u.item".format(path), "{}/u.item".format(new_path))
    save_split_data("train", user_train_dict)
    save_split_data("valid", user_valid_dict)
    save_split_data("test", user_test_dict)

def check():
    # with open("./amazon_new/user_5k/CDs_and_Vinyl/u_valid.data", "r") as f:
    #     lines = f.read().splitlines()
    # print(lines[:100], len(lines))
    pbar = tqdm(total=100)
    for i in range(100):
        pbar.update()
        time.sleep(0.1)
    

def generate_recbole(dataset):
    def generate_group_files(group_id):
        def generate_split(data_type): # train, valid, test
            with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.{}.inter".format(new_path, group_id, group_id, data_type), "w") as f:
                f.write("{}\n".format(first_lines["inter"]))
                with open("{}/u_{}.data".format(new_path, data_type), "r") as g:
                    lines = g.read().splitlines()
                with tqdm(total=len(lines), desc="Processing {}...".format(data_type)) as pbar:
                    for i in range(len(lines)):
                        cur_userid, cur_itemid, cur_rating, cur_timestamp = lines[i].split("\t")
                        if cur_userid in user_set:
                            f.write("{}|{}|{}|{}\n".format(cur_userid, cur_itemid, cur_rating, cur_timestamp))
                        pbar.update()
        
        first_lines = {
            "user": "user_id:token|profile:token_seq",
            "item": "item_id:token|item_title:token_seq|category:token_seq|description:token_seq",
            "inter": "user_id:token|item_id:token|rating:float|timestamp:float"
        }
        print("Generating group {}...".format(group_id))
        if not os.path.exists("{}/recbole/user_5k/user_5k_{}".format(new_path, group_id)):
            os.mkdir("{}/recbole/user_5k/user_5k_{}".format(new_path, group_id))

        user_set = set()
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.user".format(path, group_id, group_id), "r") as f:
            lines = f.read().splitlines()
        with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.user".format(new_path, group_id, group_id), "w") as f:
            f.write("{}\n".format(first_lines["user"]))
            with tqdm(total=len(lines) - 1, desc="Processing users...") as pbar:
                for i in range(1, len(lines)):
                    cur_userid, cur_userprofile = lines[i].split("|")
                    user_set.add(cur_userid)
                    f.write("{}\n".format(lines[i]))
                    pbar.update()
        
        if group_id != "0":
            shutil.copy("{}/recbole/user_5k/user_5k_0/user_5k_0.item".format(new_path), "{}/recbole/user_5k/user_5k_{}/user_5k_{}.item".format(new_path, group_id, group_id))
        else:
            with open("{}/u.item".format(new_path), "r") as f:
                lines = f.read().splitlines()
            with open("{}/recbole/user_5k/user_5k_{}/user_5k_{}.item".format(new_path, group_id, group_id), "w") as f:
                f.write("{}\n".format(first_lines["item"]))
                with tqdm(total=len(lines), desc="Processing item...") as pbar:
                    for i in range(len(lines)):
                        f.write("{}\n".format(lines[i]))
                        pbar.update()

        generate_split("train")
        generate_split("valid")
        generate_split("test")
    
    if dataset == "ml-25m":
        path = "./{}/user_5k".format(dataset)
        new_path = "./{}_new/user_5k".format(dataset)
    elif dataset == "amazon-CDs_and_Vinyl":
        uniset, subset = dataset.split("-")
        path = "./{}/user_5k/{}".format(uniset, subset)
        new_path = "./{}_new/user_5k/{}".format(uniset, subset)
    if not os.path.exists("{}/recbole/user_5k/user_5k_total".format(new_path)):
        os.makedirs("{}/recbole/user_5k/user_5k_total".format(new_path))
    
    for group_id in ["0", "1", "2", "3", "4", "total"]:
        generate_group_files(group_id)

def generate_vt_posnum(dataset):
    if dataset == "ml-25m":
        new_path = "./{}_new/user_5k".format(dataset)
    elif dataset == "amazon-CDs_and_Vinyl":
        uniset, subset = dataset.split("-")
        new_path = "./{}_new/user_5k/{}".format(uniset, subset)
    with open("{}/u_valid.data".format(new_path), "r") as f:
        lines = f.read().splitlines()
    positive_num_dict = {}
    with tqdm(total=len(lines), desc="Processing...") as pbar:
        for i in range(len(lines)):
            cur_userid, _, cur_rating, _ = lines[i].split("\t")
            cur_rating = float(cur_rating)
            if cur_userid not in positive_num_dict.keys():
                positive_num_dict[cur_userid] = 0
            if cur_rating >= 4.0:
                positive_num_dict[cur_userid] += 1
            pbar.update()
    with open("{}/vt_posnum.dict".format(new_path), "wb") as f:
        pickle.dump(positive_num_dict, f)

def generate_userinter(dataset):
    if dataset == "ml-25m":
        new_path = "./{}_new/user_5k".format(dataset)
    elif dataset == "amazon-CDs_and_Vinyl":
        uniset, subset = dataset.split("-")
        new_path = "./{}_new/user_5k/{}".format(uniset, subset)
    
    user_validdict, user_testdict = {}, {}
    with open("{}/u_valid.data".format(new_path), "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines), desc="Processing validation set") as pbar:
        for i in range(len(lines)):
            user_id, item_id, rating, timestamp = lines[i].split("\t")
            if user_id not in user_validdict.keys():
                user_validdict[user_id] = {}
            if float(rating) >= 4.0:
                is_pos = 1
            elif 0.0 < float(rating) < 4.0:
                is_pos = -1
            else:
                is_pos = 0
            user_validdict[user_id][item_id] = [item_id, is_pos]
            pbar.update()
    
    with open("{}/u_test.data".format(new_path), "r") as f:
        lines = f.read().splitlines()
    with tqdm(total=len(lines), desc="Processing testing set") as pbar:
        for i in range(len(lines)):
            user_id, item_id, rating, timestamp = lines[i].split("\t")
            if user_id not in user_testdict.keys():
                user_testdict[user_id] = {}
            if float(rating) >= 4.0:
                is_pos = 1
            elif 0.0 < float(rating) < 4.0:
                is_pos = -1
            else:
                is_pos = 0
            user_testdict[user_id][item_id] = [item_id, is_pos]
            pbar.update()
    
    with open("{}/user_validdict.pkl".format(new_path), "wb") as f:
        pickle.dump(user_validdict, f)
    with open("{}/user_testdict.pkl".format(new_path), "wb") as f:
        pickle.dump(user_testdict, f)

if __name__ == "__main__":
    # create("ml-25m")
    # check()
    # generate_recbole("ml-25m")
    # generate_vt_posnum("ml-25m")
    generate_userinter("ml-25m")