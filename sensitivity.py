from scipy import stats
import os
import numpy as np
import json
from ast import literal_eval
from tqdm import tqdm
import time
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from prompt import *
import tiktoken
import copy
import pickle
import sys

def check_results():
    base_folder = "./baseline_results"
    for folder in os.listdir(base_folder):
        full_path = os.path.join(base_folder, folder)
        for file in os.listdir(full_path):
            file_name = file.split(".")[0]
            model, dataset, method = file_name.split("-")
            use_llm_embed, merge_embed = False, False
            string_dict = {
                "none": ("use_llm_embed = False", "merge_embed"),
                "init": ("use_llm_embed = True", "merge_embed = none"),
                "add": ("use_llm_embed = True", "merge_embed = add"),
                "cldot": ("use_llm_embed = True", "merge_embed = cl-dot"),
                "clcos": ("use_llm_embed = True", "merge_embed = cl-cos")
            }
            use_string, merge_string = string_dict[method]
            path = os.path.join(full_path, file)
            with open(path, "r") as f:
                lines = f.read().splitlines()
            for i in range(len(lines)):
                if use_string in lines[i]:
                    use_llm_embed = True
                if merge_string in lines[i]:
                    merge_embed = True
                if use_llm_embed == True and merge_embed == True:
                    break
            
            if use_llm_embed == True and merge_embed == True:
                print("OK! {}".format(file))
            else:
                print("Wrong! {}".format(file))
                
def analysis_seedlist(seedlist):
    ndcg_list, bpref_list = {}, {}
    # ndcg_list, bpref_list = np.zeros(6), np.zeros(6)
    for i in range(len(seedlist)):
        if i == 0:
            _, cur_dict = seedlist[i].split("total test result: ")
        else:
            _, cur_dict = seedlist[i].split("test result of group {}: ".format(i))
        # cur_dict = json.loads(cur_dict)
        cur_dict = literal_eval(cur_dict)
        # for key in cur_dict.keys():
        #     if key.startswith("ndcg"):
        #         ndcg_list[i] = cur_dict[key]
        #     if key.startswith("bpref"):
        #         bpref_list[i] = cur_dict[key]
        for topk in cur_dict.keys():
            if topk not in ndcg_list.keys() and topk not in bpref_list.keys():
                ndcg_list[topk], bpref_list[topk] = np.zeros(6), np.zeros(6)
            for key in cur_dict[topk].keys():
                if key.startswith("ndcg"):
                    ndcg_list[topk][i] = cur_dict[topk][key]
                if key.startswith("bpref"):
                    bpref_list[topk][i] = cur_dict[topk][key]
            
    return ndcg_list, bpref_list # key: topk, value: the result of one seed, np.array(6)

def read_logfile(file):
    # total_ndcg_mat, total_bpref_mat = np.zeros((5, 6)), np.zeros((5, 6))
    total_ndcg_mat, total_bpref_mat = {}, {}
    with open(file, "r") as f:
        lines = f.read().splitlines()
    count = 0
    for i in range(len(lines)):
        if "Best results of seed" in lines[i]:
            cur_seedlist = []
            # for j in range(1, 7):
            for j in range(1, 2):
                cur_seedlist.append(lines[i + j])
            cur_ndcg, cur_bpref = analysis_seedlist(cur_seedlist)
            for topk in cur_ndcg.keys():
                if topk not in total_ndcg_mat.keys() and topk not in total_bpref_mat.keys():
                    # total_ndcg_mat[topk], total_bpref_mat[topk] = np.zeros((10, 6)), np.zeros((10, 6))
                    total_ndcg_mat[topk], total_bpref_mat[topk] = np.zeros((5, 6)), np.zeros((5, 6))
                total_ndcg_mat[topk][count], total_bpref_mat[topk][count] = cur_ndcg[topk], cur_bpref[topk]
            count += 1
    # key: topk, value: the result of 5 seeds, np.array((5, 6))
    return total_ndcg_mat, total_bpref_mat

def cal_ttest(file1, file2): # file1: cos/dot, file2: none
    ndcg1, bpref1 = read_logfile(file1)
    ndcg2, bpref2 = read_logfile(file2)
    ndcg_p, bpref_p = {}, {}
    for topk in ndcg1.keys():
        ndcg_p[topk] = np.zeros(6)
        bpref_p[topk] = np.zeros(6)
        for i in range(1):
            cur_ndcg1, cur_ndcg2 = ndcg1[topk][:, i], ndcg2[topk][:, i]
            cur_bpref1, cur_bpref2 = bpref1[topk][:, i], bpref2[topk][:, i]
            
            print(cur_ndcg1)
            print(cur_ndcg2)
            print(cur_bpref1)
            print(cur_bpref2)
            _, ndcg_p[topk][i] = stats.ttest_ind(cur_ndcg1, cur_ndcg2, equal_var=False)
            _, bpref_p[topk][i] = stats.ttest_ind(cur_bpref1, cur_bpref2, equal_var=False)
        with open("./history_length_results/group 5/sensitivity.log", "a") as f:
            f.write("Results of top-{}".format(topk))
            print("Results of top-{}".format(topk))
            ndcg_list= ndcg_p[topk].tolist()
            f.write("ndcg: {}\n".format(ndcg_list))
            print("ndcg: {}\n".format(ndcg_list))
            for j in range(len(ndcg_list)):
                if ndcg_list[j] <= 0.05:
                    f.write("* ")
                else:
                    f.write("- ")
            f.write("\n")
    for topk in ndcg1.keys():
        bpref_list =  bpref_p[topk].tolist()
        with open("./history_length_results/group 5/sensitivity.log", "a") as f:
            f.write("Results of top-{}".format(topk))
            print("Results of top-{}".format(topk))
            f.write("bpref: {}\n".format(bpref_list))
            print("bpref: {}\n".format(bpref_list))
            for j in range(len(bpref_list)):
                if bpref_list[j] <= 0.05:
                    f.write("* ")
                else:
                    f.write("- ")
            f.write("\n\n")
            
            
    # time.sleep(1000)
    
    # ndcg_t, ndcg_p, bpref_t, bpref_p = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
    # for i in range(6):
    #     cur_ndcg1, cur_ndcg2 = ndcg1[:, i], ndcg2[:, i]
    #     cur_bpref1, cur_bpref2 = bpref1[:, i], bpref2[:, i]
    #     ndcg_t[i], ndcg_p[i] = stats.ttest_ind(cur_ndcg1, cur_ndcg2, equal_var=False)
    #     bpref_t[i], bpref_p[i] = stats.ttest_ind(cur_bpref1, cur_bpref2, equal_var=False)
    # with open("./baseline_results/sensitivity.log", "a") as f:
    #     f.write("ndcg: {}\n".format(ndcg_p.tolist()))
    #     f.write("bpref: {}\n".format(bpref_p.tolist()))

def cal_special_sensitivity():
    path = "./history_length_results/group 5"
    with open("{}/sensitivity.log".format(path), "w") as f:
        pass
    
    file1 = "./history_length_results/group 5/NeuMF/NeuMF-user_5k_total_ml-25m_10-Sep-12-2024_08-03-17-ebd6cf.log"
    file2 = "./history_length_results/group 5/NeuMF/NeuMF-user_5k_total_ml-25m_5-Sep-12-2024_06-33-24-35ce22.log"
    cal_ttest(file1, file2)

def cal_sensitivity():
    path = "./history_length_results/group 5"
    with open("{}/sensitivity.log".format(path), "w") as f:
        pass
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'SGL', 'SimGCL', 'XSimGCL', 'LightGCL']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'XSimGCL', 'LightGCL']
    model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'XSimGCL']
    # dataset_list = ['ml100k']
    # dataset_list = ['ml100k', 'CDs_and_Vinyl']
    dataset_list = ['amazon']
    # filetype_list = ['clcos', 'cldot']
    # filetype_list = ['clcos_real',
    #                  'clcos_chatinfer_llama3con_chatembed',
    #                  'clcos_llama3infer_llama3con_chatembed',
    #                  # 'clcos_llama3full'
    #                  ]
    filetype_list = [1, 3]
    with tqdm(total=len(model_list) * len(dataset_list) * len(filetype_list)) as pbar:
        for dataset in dataset_list:
            for model in model_list:
                for filetype in filetype_list:
                    file1 = "{}/{}/{}-{}-{}.log".format(path, model, model, dataset, filetype)
                    file2 = "{}/{}/{}-{}-none.log".format(path, model, model, dataset)
                    with open("{}/sensitivity.log".format(path), "a") as f:
                        f.write("{}-{}-{}: \n".format(model, dataset, filetype))
                    cal_ttest(file1, file2)
                    pbar.update()
    pbar.close()

def generate_csv():
    def cal_improve_rate(a, b):
        if a == 0 and b == 0:
            return 0
        else:
            return (b - a) * 100 / a
    
    # dataset_list = ['ml100k', 'CDs_and_Vinyl']
    dataset_list = ['CDs_and_Vinyl']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'SimGCL', 'XSimGCL', 'LightGCL']
    model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'XSimGCL', 'LightGCL']
    filetype_list = ['none',
                     'clcos_real',
                     'clcos_chatinfer_llama3con_chatembed',
                     'clcos_llama3infer_llama3con_chatembed',
                     # 'clcos_llama3full'
                     ]
    # dataset_list = ['CDs_and_Vinyl']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'SGL', 'SimGCL', 'XSimGCL', 'LightGCL']
    # filetype_list = ['none', 'base', 'cldot', 'clcos']
    nums = [1, 5, 10, 20]
    result_dict, best_improve = {}, {}
    for dataset in dataset_list:
        result_dict[dataset], best_improve[dataset] = {}, {}
        for model in model_list:
            result_dict[dataset][model], best_improve[dataset][model] = {}, {}
            for filetype in filetype_list:
                print("{}-{}-{}".format(dataset, model, filetype))
                result_dict[dataset][model][filetype] = {}
                for num in nums:
                    result_dict[dataset][model][filetype][num] = {}
                    result_dict[dataset][model][filetype][num]['ndcg'] = []
                    result_dict[dataset][model][filetype][num]['bpref'] = []
                with open("./baseline_results_10_new/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "r") as f:
                    lines = f.read().splitlines()
                for i in range(len(lines)):
                    if "total average test results" in lines[i]:
                        startline = i
                        break
                for i in range(4, 0, -1): # In the order of 20, 10, 5, 1
                    cur_num = nums[i - 1]
                    for j in range(6):
                        cur_dict = lines[startline + i + j * 5].split("INFO  ")[1]
                        cur_dict = literal_eval(cur_dict)
                        cur_ndcg, cur_bpref = cur_dict['ndcg@{}'.format(cur_num)], cur_dict['bpref@{}'.format(cur_num)]
                        cur_ndcg = Decimal(str(cur_ndcg)).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                        cur_bpref = Decimal(str(cur_bpref)).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                        result_dict[dataset][model][filetype][cur_num]['ndcg'].append(cur_ndcg)
                        result_dict[dataset][model][filetype][cur_num]['bpref'].append(cur_bpref)
            for num in nums:
                best_improve[dataset][model][num] = {}
                best_improve[dataset][model][num]['ndcg'] = []
                best_improve[dataset][model][num]['bpref'] = []
                # print(result_dict[dataset][model]['none'][cur_num]['ndcg'])
                # print(result_dict[dataset][model]['cldot_noadd'][cur_num]['ndcg'])
                # print(result_dict[dataset][model]['clcos_noadd'][cur_num]['ndcg'])
                # time.sleep(1000)
                for i in range(6):
                    # cur_ndcg_improve = max(
                    #     cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                    #                      result_dict[dataset][model]['cldot'][num]['ndcg'][i]),
                    #     cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                    #                      result_dict[dataset][model]['clcos'][num]['ndcg'][i]),
                    # )
                    # cur_ndcg_improve = cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                    #                                     result_dict[dataset][model]['clcos'][num]['ndcg'][i])
                    cur_ndcg_improve = max(
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                                         result_dict[dataset][model]['clcos_real'][num]['ndcg'][i]),
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                                         result_dict[dataset][model]['clcos_chatinfer_llama3con_chatembed'][num]['ndcg'][i]),
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                                         result_dict[dataset][model]['clcos_llama3infer_llama3con_chatembed'][num]['ndcg'][i]),
                        # cal_improve_rate(result_dict[dataset][model]['none'][num]['ndcg'][i],
                        #                  result_dict[dataset][model]['clcos_llama3full'][num]['ndcg'][i]),
                    )
                    cur_ndcg_improve = Decimal(str(cur_ndcg_improve)).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
                    # cur_bpref_improve = max(
                    #     cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                    #                      result_dict[dataset][model]['cldot'][num]['bpref'][i]),
                    #     cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                    #                      result_dict[dataset][model]['clcos'][num]['bpref'][i]),
                    # )
                    # cur_bpref_improve = cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                    #                                      result_dict[dataset][model]['clcos'][num]['bpref'][i])
                    cur_bpref_improve = max(
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                                         result_dict[dataset][model]['clcos_real'][num]['bpref'][i]),
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                                         result_dict[dataset][model]['clcos_chatinfer_llama3con_chatembed'][num]['bpref'][i]),
                        cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                                         result_dict[dataset][model]['clcos_llama3infer_llama3con_chatembed'][num]['bpref'][i]),
                        # cal_improve_rate(result_dict[dataset][model]['none'][num]['bpref'][i],
                        #                  result_dict[dataset][model]['clcos_llama3full'][num]['bpref'][i]),
                    )
                    cur_bpref_improve = Decimal(str(cur_bpref_improve)).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
                    best_improve[dataset][model][num]['ndcg'].append(cur_ndcg_improve)
                    best_improve[dataset][model][num]['bpref'].append(cur_bpref_improve)
                # print(best_improve[dataset][model][num]['ndcg'])
                # time.sleep(1000)
    
    for dataset in dataset_list:
        with open("./tables/{}.csv".format(dataset), "w") as f:
            for num in nums[::-1]:
                f.write('NDCG@{},Total,"[10,20]","(20,30]","(30,40]","(40,50]","(50,60]"\n'.format(num))
                for model in model_list:
                    for filetype in filetype_list:
                        f.write("{}-{}".format(model, filetype))
                        cur_list = result_dict[dataset][model][filetype][num]['ndcg']
                        for i in range(len(cur_list)):
                            f.write(",{}".format(cur_list[i]))
                        f.write("\n")
                    f.write("Best Improve")
                    cur_list = best_improve[dataset][model][num]['ndcg']
                    for i in range(len(cur_list)):
                        f.write(",{}%".format(cur_list[i]))
                    f.write("\n")
                f.write(",\n")
                f.write('BPREF@{},Total,"[10,20]","(20,30]","(30,40]","(40,50]","(50,60]"\n'.format(num))
                for model in model_list:
                    for filetype in filetype_list:
                        f.write("{}-{}".format(model, filetype))
                        cur_list = result_dict[dataset][model][filetype][num]['bpref']
                        for i in range(len(cur_list)):
                            f.write(",{}".format(cur_list[i]))
                        f.write("\n")
                    f.write("Best Improve")
                    cur_list = best_improve[dataset][model][num]['bpref']
                    for i in range(len(cur_list)):
                        f.write(",{}%".format(cur_list[i]))
                    f.write("\n")
                f.write(",\n")
    
    with open("./tables/latex_main.txt", "w") as f:
        f.write("& N@1 & N@5 & N@10 & N@20 & B@5 & B@10 & B@20 & N@1 & N@5 & N@10 & N@20 & B@5 & B@10 & B@20 \\\\\n")
        for model in model_list:
            for filetype in filetype_list:
                f.write("{}-{}".format(model, filetype))
                for dataset in dataset_list:
                    for num in nums:
                        f.write(" & {}".format(result_dict[dataset][model][filetype][num]['ndcg'][0]))
                    for num in nums[1:]:
                        f.write(" & {}".format(result_dict[dataset][model][filetype][num]['bpref'][0]))
                f.write(" \\\\\n")
            f.write("Best Improve")
            for dataset in dataset_list:
                for num in nums:
                    cur_improve = best_improve[dataset][model][num]['ndcg'][0]
                    if cur_improve > 0:
                        f.write(" & $\\uparrow${}\\%".format(cur_improve))
                    elif cur_improve < 0:
                        f.write(" & $\\downarrow${}\\%".format(cur_improve))
                    else:
                        f.write(" & {}\\%".format(cur_improve))
                for num in nums[1:]:
                    cur_improve = best_improve[dataset][model][num]['bpref'][0]
                    if cur_improve > 0:
                        f.write(" & $\\uparrow${}\\%".format(cur_improve))
                    elif cur_improve < 0:
                        f.write(" & $\\downarrow${}\\%".format(cur_improve))
                    else:
                        f.write(" & {}\\%".format(cur_improve))
            f.write(" \\\\\n")
    
    return result_dict, best_improve

def plot_improve(best_improve):
    # dataset_list = ['ml100k', 'CDs_and_Vinyl']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'SGL', 'SimGCL', 'XSimGCL', 'LightGCL']
    # filetype_list = ['none', 'cldot_noadd', 'clcos_noadd']
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'SGL', 'XSimGCL', 'LightGCL']
    nums = [1, 5, 10, 20]
    
    range_dict = {
        'ml100k': {
            'NDCG': {
                1: (-14, 31),
                5: (-7, 9),
                10: (-9, 6),
                20: (-4, 4)
            },
            'BPREF': {
                1: (-1, 1),
                5: (-23, 21),
                10: (-21, 17),
                20: (-6, 12)
            }
        },
        'CDs_and_Vinyl': {
            'NDCG': {
                1: (-10, 40),
                5: (-7, 18),
                10: (-4, 9),
                20: (-2, 6)
            },
            'BPREF': {
                1: (-1, 1),
                5: (-16, 30),
                10: (-9, 22),
                20: (-5, 8)
            }
        }
    }
    
    x = np.arange(5)
    xlabels = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5"]
    with tqdm(total=len(dataset_list) * len(nums)) as pbar:
        for dataset in dataset_list:
            for num in nums:
                plt.figure(figsize=(24, 24), dpi=300)
                plt.tight_layout()
                plt.suptitle("Best improvement of {}-NDCG@{}".format(dataset, num))
                for i in range(len(model_list)):
                    plt.subplot(3, 3, i + 1)
                    model = model_list[i]
                    y = best_improve[dataset][model][num]['ndcg']
                    plt.plot(x, y[1:])
                    plt.axhline(y=y[0], xmin=x[0], xmax=x[-1], linestyle="--")
                    plt.axhline(y=0, xmin=x[0], xmax=x[-1], linestyle="--", color='orange')
                    plt.title(model)
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Groups")
                    plt.ylabel("NDCG@{}".format(num))
                    plt.ylim(range_dict[dataset]['NDCG'][num])
                plt.savefig("./img2/BestImprovement-{}-NDCG@{}.png".format(dataset, num))
                
                plt.figure(figsize=(24, 24), dpi=300)
                plt.tight_layout()
                plt.suptitle("Best improvement of {}-BPREF@{}".format(dataset, num))
                for i in range(len(model_list)):
                    plt.subplot(3, 3, i + 1)
                    model = model_list[i]
                    y = best_improve[dataset][model][num]['bpref']
                    plt.plot(x, y[1:])
                    plt.axhline(y=y[0], xmin=x[0], xmax=x[-1], linestyle="--")
                    plt.axhline(y=0, xmin=x[0], xmax=x[-1], linestyle="--", color='orange')
                    plt.title(model)
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Groups")
                    plt.ylabel("NDCG@{}".format(num))
                    plt.ylim(range_dict[dataset]['BPREF'][num])
                plt.savefig("./img2/BestImprovement-{}-BPREF@{}.png".format(dataset, num))
                pbar.update()
    pbar.close()

def get_group_ablation(best_improve):
    font = fm.FontProperties(size=28)
    model_list = ['XSimGCL', 'LightGCN', 'FM']
    # y1 = best_improve['ml100k']['XSimGCL'][5]
    # y2 = best_improve['ml100k']['LightGCL'][5]
    # y3 = best_improve['ml100k']['FM'][5]
    x = np.arange(5)
    xlabels = ["1", "2", "3", "4", "5"]
    
    ylim_list = [(-2, 4), (-6, 12)]
    plt.figure(figsize=(20, 14), dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.07, hspace=0.3, wspace=0.3)
    for i in range(len(model_list)):
        plt.subplot(2, 3, i + 1)
        model = model_list[i]
        y = best_improve['ml100k'][model][20]['ndcg']
        plt.plot(x, y[1:])
        plt.axhline(y=y[0], xmin=x[0], xmax=x[-1], linestyle="--")
        plt.axhline(y=0, xmin=x[0], xmax=x[-1], linestyle="--", color='orange')
        plt.title("{}-NDCG@20".format(model), fontproperties=font)
        plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
        plt.yticks(fontproperties=font)
        plt.xlabel("Group ID", fontproperties=font)
        plt.ylabel("Imprv. (%)", fontproperties=font, labelpad=-10)
        plt.ylim(ylim_list[0])
        
        plt.subplot(2, 3, i + 4)
        model = model_list[i]
        y = best_improve['ml100k'][model][20]['bpref']
        plt.plot(x, y[1:])
        plt.axhline(y=y[0], xmin=x[0], xmax=x[-1], linestyle="--")
        plt.axhline(y=0, xmin=x[0], xmax=x[-1], linestyle="--", color='orange')
        plt.title("{}-BPREF@20".format(model), fontproperties=font)
        plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
        plt.yticks(fontproperties=font)
        plt.xlabel("Group ID", fontproperties=font)
        plt.ylabel("Imprv. (%)", fontproperties=font, labelpad=-10)
        plt.ylim(ylim_list[1])
    plt.savefig("./img2/group_ablation.png")

def plot_condense_length():
    result_dict = {
        'XSimGCL': {
            'ml100k-cldot': {
                'ndcg': [0.5995, 0.6004, 0.6018, 0.5965],
                'bpref': [0.5014, 0.5060, 0.5051, 0.5114]
            },
            'ml100k-clcos': {
                'ndcg': [0.6031, 0.6005, 0.6026, 0.6003],
                'bpref': [0.5000, 0.4989, 0.4940, 0.4934]
            },
            'CDs_and_Vinyl-cldot': {
                'ndcg': [0.5551, 0.5539, 0.5543, 0.5556],
                'bpref': [0.5900, 0.5912, 0.5923, 0.5940],
            },
            'CDs_and_Vinyl-clcos': {
                'ndcg': [0.5539, 0.5541, 0.5533, 0.5541],
                'bpref': [0.5801, 0.5803, 0.5772, 0.5812],
            }
        },
        'LightGCN': {
            'ml100k-cldot': {
                'ndcg': [0.5800, 0.5690, 0.5795, 0.5662],
                'bpref': [0.4866, 0.4840, 0.4906, 0.4912],
            },
            'ml100k-clcos': {
                'ndcg': [0.6057, 0.6049, 0.6042, 0.6017],
                'bpref': [0.4954, 0.4943, 0.4946, 0.4917],
            },
            'CDs_and_Vinyl-cldot': {
                'ndcg': [0.5087, 0.5097, 0.5103, 0.5149],
                'bpref': [0.4940, 0.4949, 0.4952, 0.4969],
            },
            'CDs_and_Vinyl-clcos': {
                'ndcg': [0.5130, 0.5145, 0.5137, 0.5137],
                'bpref': [0.4897, 0.4909, 0.4875, 0.4875],
            }
        },
        'LightGCL': {
            'ml100k-cldot': {
                'ndcg': [0.5697, 0.5706, 0.5671, 0.5715],
                'bpref': [0.5074, 0.5051, 0.4989, 0.5042],
            },
            'ml100k-clcos': {
                'ndcg': [0.5902, 0.5922, 0.5866, 0.5907],
                'bpref': [0.4872, 0.4883, 0.4886, 0.4889],
            },
            'CDs_and_Vinyl-cldot': {
                'ndcg': [0.5099, 0.5058, 0.5119, 0.5114],
                'bpref': [0.4698, 0.4746, 0.4624, 0.4801],
            },
            'CDs_and_Vinyl-clcos': {
                'ndcg': [0.5011, 0.5169, 0.5102, 0.5123],
                'bpref': [0.4801, 0.5131, 0.4846, 0.4943]
            }
        }
    }
    x = np.arange(4)
    xlabels = ["16", "32", "48", "no condense"]
    for model in result_dict.keys():
        plt.figure(figsize=(24, 12), dpi=300)
        plt.tight_layout()
        plt.suptitle("NDCG@20 and BPREF@20 of {} at different condense length".format(model))
        count = 1
        for dataset_type in result_dict[model].keys():
            plt.subplot(2, 4, count)
            y = result_dict[model][dataset_type]['ndcg']
            plt.plot(x, y)
            plt.title("{}-{}-ndcg@20".format(model, dataset_type))
            plt.xticks(ticks=x, labels=xlabels)
            plt.xlabel("Length")
            plt.ylabel("NDCG@20")
            plt.subplot(2, 4, count + 4)
            y = result_dict[model][dataset_type]['bpref']
            plt.plot(x, y)
            plt.title("{}-{}-bpref@20".format(model, dataset_type))
            plt.xticks(ticks=x, labels=xlabels)
            plt.xlabel("Length")
            plt.ylabel("BPREF@20")
            count += 1
        plt.savefig("./img2/condense_length_{}.png".format(model))

def plot_history_length():
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    model_list = ['XSimGCL', 'LightGCN', 'LightGCL', 'FM']
    filetype_list = ['clcos']
    suffix = [x for x in range(5, 65, 5)]
    suffix.remove(55)
    xlabels = [str(x) for x in suffix]
    xlabels[-1] = 'Full'
    x = np.arange(len(xlabels))
    
    length = 20
    
    # startpoint = {
    #     'XSimGCL': [0.6, 0.6, 0.55, 0.55, 0.495, 0.49, 0.575, 0.586],
    #     'LightGCN': [0.6, 0.578, 0.507, 0.508, 0.494, 0.488, 0.482, 0.489],
    #     'LightGCL': [0.590, 0.555, 0.501, 0.507, 0.485, 0.495, 0.480, 0.464]
    # }
    # height = {
    #     'XSimGCL': 0.022,
    #     'LightGCN': 0.016,
    #     'LightGCL': 0.035
    # }
    
    for model in model_list:
        plt.figure(figsize=(24, 12), dpi=300)
        plt.tight_layout()
        plt.suptitle("NDCG@{} and BPREF@{} of {} at different user history length".format(length, length, model))
        count = 1
        for dataset in dataset_list:
            for filetype in filetype_list:
                if model == 'FM' and (dataset == 'ml100k' or filetype == 'cldot'):
                    continue
                ndcg_list, bpref_list = [], []
                for cur_suffix in suffix:
                    with open("./ablation_results2/{}/{}-{}-{}-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                        lines = f.read().splitlines()
                    for i in range(len(lines)):
                        if "total average test results" in lines[i]:
                            cur_dict = lines[i + 4].split("INFO  ")[1]
                            cur_dict = literal_eval(cur_dict)
                            ndcg_list.append(cur_dict['ndcg@{}'.format(length)])
                            bpref_list.append(cur_dict['bpref@{}'.format(length)])
                            break
                plt.subplot(2, 4, count)
                plt.plot(x, ndcg_list)
                plt.title("{}-{}-ndcg@{}".format(dataset, filetype, length))
                plt.xticks(ticks=x, labels=xlabels)
                plt.xlabel("Length")
                plt.ylabel("NDCG@{}".format(length))
                # plt.ylim(startpoint[model][count - 1], startpoint[model][count - 1] + height[model])
                plt.subplot(2, 4, count + 4)
                plt.plot(x, bpref_list)
                plt.title("{}-{}-bpref@{}".format(dataset, filetype, length))
                plt.xticks(ticks=x, labels=xlabels)
                plt.xlabel("Length")
                plt.ylabel("BPREF@{}".format(length))
                # plt.ylim(startpoint[model][count + 3], startpoint[model][count + 3] + height[model])
                count += 1
        plt.savefig("./img2/history_length_{}_@{}_v2.png".format(model, length))

def get_history_ablation():
    # file = [('ml100k', 'XSimGCL', 'clcos'), ('ml100k', 'LightGCN', 'clcos'), ('CDs_and_Vinyl', 'FM', 'clcos')]
    file_20 = [('ml100k', 'XSimGCL', '', 'clcos'), ('ml100k', 'LightGCN', '', 'clcos'), ('CDs_and_Vinyl', 'FM', '2', 'clcos')]
    file_10 = [('ml100k', 'XSimGCL', '', 'clcos'), ('ml100k', 'LightGCN', '', 'clcos'),
               ('CDs_and_Vinyl', 'LightGCN', '', 'clcos'), ('CDs_and_Vinyl', 'FM', '2', 'clcos')]
    file_5 = [('ml100k', 'XSimGCL', '', 'clcos'), ('ml100k', 'LightGCL', '2', 'clcos'),
              ('ml100k', 'LightGCN', '2', 'clcos'), ('CDs_and_Vinyl', 'FM', '2', 'clcos')]
    
    font = fm.FontProperties(size=28)
    
    
    suffix = [x for x in range(5, 65, 5)]
    suffix.remove(55)
    xlabels = [str(x) for x in suffix]
    xlabels[-1] = 'Full'
    x = np.arange(len(xlabels))
    
    startpoint = [0.6, 0.6, 0.643, 0.49, 0.49, 0.466]
    height = [0.012, 0.023]
    
    # startpoint = [0.6, 0.6, 0.48, 0.65, 0.4, 0.43, 0.3, 0.38]
    # height = [0.025, 0.05]
    
    # startpoint = [0.57, 0.57, 0.555, 0.635, 0.28, 0.275, 0.27, 0.34]
    # height = [0.03, 0.065]
    
    plt.figure(figsize=(33, 14), dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.07, hspace=0.3, wspace=0.2)
    for i in range(len(file_20)):
        dataset, model, foldertype, filetype = file_20[i]
        ndcg_list, bpref_list = [], []
        for cur_suffix in suffix:
            with open("./ablation_results{}/{}/{}-{}-{}-{}.log".format(foldertype, model, model, dataset, filetype, cur_suffix), "r") as f:
                lines = f.read().splitlines()
            for j in range(len(lines)):
                if "total average test results" in lines[j]:
                    cur_dict = lines[j + 4].split("INFO  ")[1]
                    cur_dict = literal_eval(cur_dict)
                    ndcg_list.append(cur_dict['ndcg@20'])
                    bpref_list.append(cur_dict['bpref@20'])
                    break
        plt.subplot(2, 3, i + 1)
        plt.plot(x, ndcg_list)
        plt.title("{}-{}-ndcg@20".format(model, dataset), fontproperties=font)
        plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
        plt.xlabel("Learned User History Length", fontproperties=font)
        plt.ylabel("NDCG@20", fontproperties=font)
        plt.ylim(startpoint[i], startpoint[i] + height[0])
        plt.yticks(fontproperties=font)
        plt.subplot(2, 3, i + 4)
        plt.plot(x, bpref_list)
        plt.title("{}-{}-bpref@20".format(model, dataset), fontproperties=font)
        plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
        plt.xlabel("Learned User History Length", fontproperties=font)
        plt.ylabel("BPREF@20", fontproperties=font)
        plt.ylim(startpoint[i + 3], startpoint[i + 3] + height[1])
        plt.yticks(fontproperties=font)
    plt.savefig("./img2/history_ablation.png")
    
    
def get_bestparam():
    # dataset_list = ['ml100k', 'CDs_and_Vinyl', 'Office_Products']
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'SimGCL', 'XSimGCL', 'LightGCL', 'SGL']
    # model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'FM', 'WideDeep', 'LightGCL']
    model_list = ['SimGCL', 'XSimGCL']
    # model_list = ['XSimGCL', 'SimGCL', 'SGL']
    # filetype_list = ['cldot_noadd', 'clcos_noadd']
    filetype_list = ['clcos_real']
    with open("./tables/bestparams.txt", "w") as f:
        pass
    for filetype in filetype_list:
        for dataset in dataset_list:
            for model in model_list:
                cur_dict = {}
                with open("./baseline_results_10_new/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "r") as f:
                    lines = f.read().splitlines()
                for i in range(len(lines)):
                    if "Best results of seed" in lines[i]:
                        cur_seed = lines[i].split(" ")[-1][:-1]
                        cur_param = lines[i + 7].split("With parameters: ")[-1]
                        cur_dict[cur_seed] = cur_param
                with open("./tables/bestparams.txt", "a") as f:
                    f.write("{}-{}-{}\n".format(dataset, model, filetype))
                    for key in cur_dict.keys():
                        f.write("\'{}\': {},\n".format(key, cur_dict[key]))
                    f.write("\n")

def merge_results():
    dataset_list = ['ml100k', 'CDs_and_Vinyl', 'Office_Products']
    model_list = ['LightGCN', 'SimpleX', 'NeuMF', 'SpectralCF', 'FM', 'WideDeep', 'SGL', 'SimGCL', 'XSimGCL', 'LightGCL']
    filetype_list = ['none', 'cldot_noadd', 'clcos_noadd']
    if not os.path.exists("./baseline_results_10"):
        os.mkdir("./baseline_results_10")
    for model in model_list:
        if not os.path.exists("./baseline_results_10/{}".format(model)):
            os.mkdir("./baseline_results_10/{}".format(model))
        for dataset in dataset_list:
            for filetype in filetype_list:
                with open("./baseline_results_10/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "w") as f:
                    pass
                with open("./baseline_results_10/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "r") as f:
                    lines = f.read().splitlines()
                result_lines = []
                for i in range(len(lines)):
                    if "Best results of seed" in lines[i]:
                        with open("./baseline_results_10/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "a") as f:
                            for j in range(8):
                                f.write("{}\n".format(lines[i + j]))
                            f.write("\n")
                    if "total average test results" in lines[i]:
                        for j in range(30):
                            try:
                                cur_line = literal_eval(lines[i + j])
                            except:
                                cur_line = lines[i + j]
                            result_lines.append(cur_line)
                        break
                with open("./baseline_results_10/{}/{}-{}-{}2.log".format(model, model, dataset, filetype), "r") as f:
                    lines = f.read().splitlines()
                for i in range(len(lines)):
                    if "Best results of seed" in lines[i]:
                        with open("./baseline_results_10/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "a") as f:
                            for j in range(8):
                                f.write("{}\n".format(lines[i + j]))
                            f.write("\n")
                    if "total average test results" in lines[i]:
                        for j in range(30):
                            if isinstance(result_lines[i + j], dict):
                                cur_line = literal_eval(lines[i + j])
                                for key in result_lines[i + j].keys():
                                    result_lines[i + j][key] = (result_lines[i + j][key] + lines[i + j][key]) / 2
                        with open("./baseline_results_10/{}/{}-{}-{}.log".format(model, model, dataset, filetype), "a") as f:
                            for j in range(len(result_lines)):
                                f.write("{}\n".format(result_lines[j]))
                        break

def get_vector_mod():
    path = "./ml100k/user_200_3/recbole/user_200/user_200_total/"
    with open("{}/user_200_total.useremb".format(path), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    max_length = -1
    for i in range(1, len(lines)):
        cur_embed = lines[i].split("|")[1]
        cur_embed = cur_embed.split(", ")[:-1]
        for j in range(len(cur_embed)):
            cur_embed[j] = float(cur_embed[j])
        cur_embed = np.array(cur_embed)
        mod = np.linalg.norm(cur_embed)
        if mod > max_length:
            max_length = mod
    print(max_length)

def calculate_token(dataset, condense):
    def get_num_tokens(messages, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    prompt_obj = Prompt()
    # if condense:
    #     if dataset == "ml100k":
    #         path = "./ml100k/user_200_3"
    #         llm_path = "5-1"
    #         system_msg = prompt_obj.system_msg[0]
    #     elif "amazon" in dataset:
    #         subset = dataset.split("-")[1]
    #         path = "./amazon/user_200_3/{}".format(subset)
    #         if subset == "CDs_and_Vinyl":
    #             llm_path = "5-2"
    #             system_msg = prompt_obj.system_msg[1]
    #         elif subset == "Office_Products":
    #             llm_path = "5-3"
    #             system_msg = prompt_obj.system_msg[2]
    # else:
    #     if dataset == "ml100k":
    #         path = "./ml100k/user_200_3"
    #         llm_path = "99-1"
    #         system_msg = prompt_obj.system_msg[0]
    #     elif "amazon" in dataset:
    #         subset = dataset.split("-")[1]
    #         path = "./amazon/user_200_3/{}".format(subset)
    #         if subset == "CDs_and_Vinyl":
    #             llm_path = "99-2-new"
    #             system_msg = prompt_obj.system_msg[1]
    
    if condense:
        if dataset == "ml100k":
            path = "./ml100k/user_200_3"
            llm_path = "ml100k-gpt4"
            system_msg = prompt_obj.system_msg[0]
        elif "amazon" in dataset:
            subset = dataset.split("-")[1]
            path = "./amazon/user_200_3/{}".format(subset)
            if subset == "CDs_and_Vinyl":
                llm_path = "amazon-CDs_and_Vinyl-gpt4"
                system_msg = prompt_obj.system_msg[1]
    else:
        if dataset == "ml100k":
            path = "./ml100k/user_200_3"
            llm_path = "ml100k-gpt4-nocondense"
            system_msg = prompt_obj.system_msg[0]
        elif "amazon" in dataset:
            subset = dataset.split("-")[1]
            path = "./amazon/user_200_3/{}".format(subset)
            if subset == "CDs_and_Vinyl":
                llm_path = "amazon-CDs_and_Vinyl-gpt4-nocondense"
                system_msg = prompt_obj.system_msg[1]
    
    with open("{}/recbole/user_200/user_200_4/user_200_4.user".format(path), "r") as f:
        lines = f.read().splitlines()
    userid_list = []
    for i in range(1, len(lines)):
        cur_userid = lines[i].split("|", 1)[0]
        userid_list.append(cur_userid)
    
    # print(userid_list)
    # time.sleep(1000)
    
    total_input_tokens, total_output_tokens = 0, 0
    # with tqdm(total=len(userid_list)) as pbar:
    for i in range(len(userid_list)):
        cur_input_tokens, cur_output_tokens = 0, 0
        with open("./LLM_results/{}/logs/{}.jsonl".format(llm_path, userid_list[i]), "r") as f:
            lines = f.read().splitlines()
        for j in range(len(lines)):
            lines[j] = lines[j].replace(" NaN", " \"NaN\"")
        for j in range(len(lines) - 2):
            # print(lines[j])
            dicts = []
            dicts.append(literal_eval(lines[j]))
            dicts.append(literal_eval(lines[j + 1]))
            dicts.append(literal_eval(lines[j + 2]))
            # dicts = [literal_eval(lines[j]), literal_eval(lines[j + 1]), literal_eval(lines[j + 2])]
            if ("Profile" in dicts[0].keys() and "Interaction" in dicts[1].keys() and "ModelOutput" in dicts[2].keys()) or \
                ("***[User condense]***" in dicts[0].keys() and "Interaction" in dicts[1].keys() and "ModelOutput" in dicts[2].keys()):
                np.random.seed(int(time.time() * 1e6 % 1e6))
                cur_user_example = prompt_obj.user_example[np.random.randint(0, len(prompt_obj.user_example))]
                cur_prefix = copy.deepcopy(prompt_obj.user_prompt[-1])
                cur_prefix = system_msg + cur_prefix
                cur_prefix[1]['content'] += cur_user_example
                input_msg, output_msg = "", ""
                try:
                    cur_profile = "User Original Profile: {}\n".format(dicts[0]["Profile"])
                except:
                    cur_profile = "User Original Profile: {}\n".format(dicts[0]["***[User condense]***"])
                cur_inter = "{}\n".format(dicts[1]["Interaction"])
                input_msg = cur_profile + cur_inter
                output_msg = dicts[2]["ModelOutput"]
                input_msg = [{"role": "user", "content": input_msg}]
                output_msg = [{"role": "assistant", "content": output_msg}]
                cur_input_tokens += get_num_tokens(messages=cur_prefix + input_msg, model="gpt-4-turbo-20240409")
                cur_output_tokens += get_num_tokens(messages=output_msg, model="gpt-4-turbo-20240409")
                j += 3
            elif ("Profile" in dicts[0].keys() and "***[User condense]***" in dicts[1].keys()):
                np.random.seed(int(time.time() * 1e6 % 1e6))
                cur_condense_example = prompt_obj.condense_example[np.random.randint(0, len(prompt_obj.condense_example))]
                cur_prefix = copy.deepcopy(prompt_obj.condense_prompt[-1])
                cur_prefix = system_msg + cur_prefix
                cur_prefix[1]['content'] += cur_condense_example
                
                cur_profile = "User Original Profile: {}\n".format(dicts[0]["Profile"])
                input_msg = cur_profile
                output_msg = dicts[1]["***[User condense]***"]
                input_msg = [{"role": "user", "content": input_msg}]
                output_msg = [{"role": "assistant", "content": output_msg}]
                cur_input_tokens += get_num_tokens(messages=cur_prefix + input_msg, model="gpt-4-turbo-20240409")
                cur_output_tokens += get_num_tokens(messages=output_msg, model="gpt-4-turbo-20240409")
                j += 2
            elif "Error info" in dicts[0].keys() and dicts[0]["Error info"] == "Fatal error! Rollback 10 times! Restart this user!":
                cur_input_tokens, cur_output_tokens = 0, 0
        total_input_tokens += cur_input_tokens
        total_output_tokens += cur_output_tokens
            # pbar.update()
    # pbar.close()
    print("Input tokens: {}, Output tokens: {}".format(total_input_tokens, total_output_tokens))
    
    input_money, output_money = 10, 30
    total_cost = total_input_tokens / 1000000 * input_money + total_output_tokens / 1000000 * output_money
    print("Total cost: ${}".format(total_cost))


def plot_history_length_new():
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    # model_list = ['XSimGCL', 'LightGCN', 'LightGCL', 'FM']
    # model_list = ['XSimGCL', 'LightGCN', 'LightGCL']
    model_list = ['FM']
    filetype_list = ['clcos']
    
    suffix = [x for x in range(5, 65, 5)]
    xlabels = [str(x) for x in suffix]
    xlabels[-1] = 'Full'
    x = np.arange(len(xlabels))
    
    add_dict = {20: 4, 10: 3, 5: 2}
    
    for length in [20, 10, 5]:
        for model in model_list:
            plt.figure(figsize=(24, 12), dpi=300)
            plt.tight_layout()
            plt.suptitle("NDCG@{} and BPREF@{} of {} at different user history length".format(length, length, model))
            count = 1
            for dataset in dataset_list:
                for filetype in filetype_list:
                    # if model == 'FM' and (dataset == 'ml100k' or filetype == 'cldot'):
                    #     continue
                    ndcg_list, bpref_list = [], []
                    ndcg_list_nc, bpref_list_nc = [], []
                    for cur_suffix in suffix:
                        with open("./ablation_results/{}/{}-{}-{}-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                            lines = f.read().splitlines()
                        for i in range(len(lines)):
                            if "total average test results" in lines[i]:
                                cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                                cur_dict = literal_eval(cur_dict)
                                ndcg_list.append(cur_dict['ndcg@{}'.format(length)])
                                bpref_list.append(cur_dict['bpref@{}'.format(length)])
                                break
                        with open("./ablation_results/{}/{}-{}-{}_nc-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                            lines = f.read().splitlines()
                        for i in range(len(lines)):
                            if "total average test results" in lines[i]:
                                cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                                cur_dict = literal_eval(cur_dict)
                                ndcg_list_nc.append(cur_dict['ndcg@{}'.format(length)])
                                bpref_list_nc.append(cur_dict['bpref@{}'.format(length)])
                                break
                    plt.subplot(2, 4, count)
                    plt.plot(x, ndcg_list, label='condense')
                    plt.plot(x, ndcg_list_nc, label='no condense')
                    plt.title("{}-{}-ndcg@{}".format(model, dataset, length))
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Length")
                    plt.ylabel("NDCG@{}".format(length))
                    # plt.legend()
                    # plt.ylim(startpoint[model][count - 1], startpoint[model][count - 1] + height[model])
                    plt.subplot(2, 4, count + 4)
                    plt.plot(x, bpref_list, label='condense')
                    plt.plot(x, bpref_list_nc, label='no condense')
                    plt.title("{}-{}-bpref@{}".format(model, dataset, length))
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Length")
                    plt.ylabel("BPREF@{}".format(length))
                    # plt.legend()
                    # plt.ylim(startpoint[model][count + 3], startpoint[model][count + 3] + height[model])
                    count += 1
            plt.savefig("./img3/history_length_{}_@{}.png".format(model, length))
            
            plt.figure(figsize=(24, 12), dpi=300)
            plt.tight_layout()
            plt.suptitle("NDCG@{} and BPREF@{} of {} at different user history length".format(length, length, model))
            count = 1
            for dataset in dataset_list:
                for filetype in filetype_list:
                    if filetype == 'cldot':
                        continue
                    ndcg_list, bpref_list = [], []
                    ndcg_list_nc, bpref_list_nc = [], []
                    for cur_suffix in suffix:
                        with open("./ablation_results2/{}/{}-{}-{}-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                            lines = f.read().splitlines()
                        for i in range(len(lines)):
                            if "total average test results" in lines[i]:
                                cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                                cur_dict = literal_eval(cur_dict)
                                ndcg_list.append(cur_dict['ndcg@{}'.format(length)])
                                bpref_list.append(cur_dict['bpref@{}'.format(length)])
                                break
                        with open("./ablation_results2/{}/{}-{}-{}_nc-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                            lines = f.read().splitlines()
                        for i in range(len(lines)):
                            if "total average test results" in lines[i]:
                                cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                                cur_dict = literal_eval(cur_dict)
                                ndcg_list_nc.append(cur_dict['ndcg@{}'.format(length)])
                                bpref_list_nc.append(cur_dict['bpref@{}'.format(length)])
                                break
                    plt.subplot(2, 4, count)
                    plt.plot(x, ndcg_list, label='condense')
                    plt.plot(x, ndcg_list_nc, label='no condense')
                    plt.title("{}-{}-ndcg@{}".format(dataset, filetype, length))
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Length")
                    plt.ylabel("NDCG@{}".format(length))
                    plt.legend()
                    # plt.ylim(startpoint[model][count - 1], startpoint[model][count - 1] + height[model])
                    plt.subplot(2, 4, count + 4)
                    plt.plot(x, bpref_list, label='condense')
                    plt.plot(x, bpref_list_nc, label='no condense')
                    plt.title("{}-{}-bpref@{}".format(dataset, filetype, length))
                    plt.xticks(ticks=x, labels=xlabels)
                    plt.xlabel("Length")
                    plt.ylabel("BPREF@{}".format(length))
                    plt.legend()
                    # plt.ylim(startpoint[model][count + 3], startpoint[model][count + 3] + height[model])
                    count += 1
            plt.savefig("./img3/history_length_{}_@{}_new.png".format(model, length))


def get_history_ablation_new():
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    model_list = ['XSimGCL', 'LightGCN']
    filetype_list = ['clcos']
    length = 20
    
    suffix = [x for x in range(5, 65, 5)]
    xlabels = [str(x) for x in suffix]
    xlabels[-1] = 'Full'
    x = np.arange(len(xlabels))
    add_dict = {20: 4, 10: 3, 5: 2}
    font = fm.FontProperties(size=28)
    
    startpoint = [0.6, 0.55, 0.6, 0.507]
    height = [0.009, 0.009, 0.009, 0.009]
    
    count = 1
    plt.figure(figsize=(17, 17), dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.07, hspace=0.25, wspace=0.25)
    for model in model_list:
        for dataset in dataset_list:
            for filetype in filetype_list:
                ndcg_list, bpref_list = [], []
                ndcg_list_nc, bpref_list_nc = [], []
                for cur_suffix in suffix:
                    with open("./ablation_results/{}/{}-{}-{}-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                        lines = f.read().splitlines()
                    for i in range(len(lines)):
                        if "total average test results" in lines[i]:
                            cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                            cur_dict = literal_eval(cur_dict)
                            ndcg_list.append(cur_dict['ndcg@{}'.format(length)])
                            bpref_list.append(cur_dict['bpref@{}'.format(length)])
                            break
                    with open("./ablation_results/{}/{}-{}-{}_nc-{}.log".format(model, model, dataset, filetype, cur_suffix), "r") as f:
                        lines = f.read().splitlines()
                    for i in range(len(lines)):
                        if "total average test results" in lines[i]:
                            cur_dict = lines[i + add_dict[length]].split("INFO  ")[1]
                            cur_dict = literal_eval(cur_dict)
                            ndcg_list_nc.append(cur_dict['ndcg@{}'.format(length)])
                            bpref_list_nc.append(cur_dict['bpref@{}'.format(length)])
                            break
                plt.subplot(2, 2, count)
                plt.plot(x, ndcg_list, label='condense')
                plt.plot(x, ndcg_list_nc, label='no condense')
                plt.title("{}-{}-ndcg@{}".format(model, dataset, length), fontproperties=font)
                plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
                plt.xlabel("Learned User History Length", fontproperties=font)
                plt.ylabel("NDCG@{}".format(length), fontproperties=font)
                plt.yticks(fontproperties=font)
                # plt.legend()
                plt.ylim(startpoint[count - 1], startpoint[count - 1] + height[count - 1])
                # plt.subplot(2, 2, count + 2)
                # plt.plot(x, bpref_list, label='condense')
                # plt.plot(x, bpref_list_nc, label='no condense')
                # plt.title("{}-{}-bpref@{}".format(model, dataset, length), fontproperties=font)
                # plt.xticks(ticks=x, labels=xlabels, fontproperties=font)
                # plt.xlabel("Learned User History Length", fontproperties=font)
                # plt.ylabel("BPREF@{}".format(length), fontproperties=font)
                # plt.yticks(fontproperties=font)
                # plt.ylim(startpoint[count + 1], startpoint[count + 1] + height[count + 1])
                # plt.legend()
                count += 1
        plt.savefig("./img3/history_length_final_ablation.png")

def merge_logs():
    def processing_logs(file):
        length = [1, 5, 10, 20]
        result_list = [{}, {}, {}, {}, {}, {}]
        if file is None:
            return result_list
        for i in range(len(result_list)):
            cur_dict = result_list[i]
            for j in range(len(length)):
                cur_dict[length[j]] = {}
                cur_dict[length[j]]['ndcg@{}'.format(length[j])] = 0
                cur_dict[length[j]]['mrr@{}'.format(length[j])] = 0
                cur_dict[length[j]]['bpref@{}'.format(length[j])] = 0
        
        with open(file, "r") as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            if "Best results of seed" in lines[i]:
                with open("./merged.log", "a") as f:
                    f.write("{}\n".format(lines[i]))
                    for j in range(1, 2):
                        f.write("{}\n".format(lines[i + j]))
                        cur_dict = lines[i + j].split(": ", maxsplit=1)[1]
                        cur_dict = literal_eval(cur_dict)
                        for key in cur_dict.keys():
                            for key2 in cur_dict[key].keys():
                                result_list[j - 1][key][key2] += cur_dict[key][key2] / 5
                    f.write("{}\n".format(lines[i + 2]))
        return result_list

    with open("./merged.log", "w") as f:
        pass
    
    info_prefix = "Sun 19 May 2024 01:23:15 INFO  "
    
    file1 = "./group 5/XSimGCL/old/XSimGCL-amazon-5.log"
    # file2 = "./deepest_res/WideDeep-ml100k-clcos_real-2022-2029.log"
    file2 = None
    result_list1 = processing_logs(file1)
    result_list2 = processing_logs(file2)
    
    for i in range(len(result_list2)):
        for key in result_list2[i].keys():
            for key2 in result_list2[i][key].keys():
                result_list1[i][key][key2] += result_list2[i][key][key2]
                result_list1[i][key][key2] /= 5
    
    with open("./merged.log", "a") as f:
        for i in range(6):
            if i == 0:
                f.write("{}total average test results\n".format(info_prefix))
            else:
                f.write("{}average test result of group {}\n".format(info_prefix, i))
            for key in result_list1[i].keys():
                f.write("{}{{'ndcg@{}': {}, 'mrr@{}': {}, 'bpref@{}': {}}}\n".format(info_prefix, key, result_list1[i][key]['ndcg@{}'.format(key)], key, result_list1[i][key]['mrr@{}'.format(key)], key, result_list1[i][key]['bpref@{}'.format(key)]))

def compare_condense():
    dataset_list = ['ml100k', 'CDs_and_Vinyl']
    model_list = ['FM', 'LightGCN', 'XSimGCL']
    filetype_list = ['clcos']
    length = 20
    suffix = [x for x in range(5, 65, 5)]
    
    result, result_nc = {}, {}
    
    with open("./tables/compare_condense.csv", "w") as f:
        pass
    
    font = fm.FontProperties(size=28)
    
    for model in model_list:
        result[model] = {}
        result_nc[model] = {}
        for dataset in dataset_list:
            # if dataset == 'ml100k' and model == 'FM':
            #     continue
            result[model][dataset] = {}
            result[model][dataset]['ndcg'] = np.zeros(len(suffix) + 1)
            result[model][dataset]['bpref'] = np.zeros(len(suffix) + 1)
            result_nc[model][dataset] = {}
            result_nc[model][dataset]['ndcg'] = np.zeros(len(suffix) + 1)
            result_nc[model][dataset]['bpref'] = np.zeros(len(suffix) + 1)
            for cur_suffix in suffix:
                with open("./ablation_results/{}/{}-{}-clcos-{}.log".format(model, model, dataset, cur_suffix), "r") as f:
                    lines = f.read().splitlines()
                for i in range(len(lines)):
                    if "total average test results" in lines[i]:
                        cur_dict = lines[i + 4].split("INFO  ")[1]
                        cur_dict = literal_eval(cur_dict)
                        result[model][dataset]['ndcg'][0] += cur_dict['ndcg@20'] / len(suffix)
                        result[model][dataset]['ndcg'][cur_suffix // 5] = cur_dict['ndcg@20']
                        result[model][dataset]['bpref'][0] += cur_dict['bpref@20'] / len(suffix)
                        result[model][dataset]['bpref'][cur_suffix // 5] = cur_dict['bpref@20']
                        break
                
                with open("./ablation_results/{}/{}-{}-clcos_nc-{}.log".format(model, model, dataset, cur_suffix), "r") as f:
                    lines = f.read().splitlines()
                for i in range(len(lines)):
                    if "total average test results" in lines[i]:
                        cur_dict = lines[i + 4].split("INFO  ")[1]
                        cur_dict = literal_eval(cur_dict)
                        result_nc[model][dataset]['ndcg'][0] += cur_dict['ndcg@20'] / len(suffix)
                        result_nc[model][dataset]['ndcg'][cur_suffix // 5] = cur_dict['ndcg@20']
                        result_nc[model][dataset]['bpref'][0] += cur_dict['bpref@20'] / len(suffix)
                        result_nc[model][dataset]['bpref'][cur_suffix // 5] = cur_dict['bpref@20']
                        break

            cur_ndcg = result[model][dataset]['ndcg'] - result_nc[model][dataset]['ndcg']
            cur_bpref = result[model][dataset]['bpref'] - result_nc[model][dataset]['bpref']
            with open("./tables/compare_condense.csv", "a") as f:
                f.write("{}-{},average".format(model, dataset))
                for i in range(len(suffix)):
                    f.write(",{}".format(suffix[i]))
                f.write("\nndcg")
                for i in range(cur_ndcg.shape[0]):
                    f.write(",{}".format(cur_ndcg[i]))
                f.write("\nbpref")
                for i in range(cur_bpref.shape[0]):
                    f.write(",{}".format(cur_bpref[i]))
                f.write("\n,\n")
    
    x = np.arange(6)
    y1, y2 = [], []
    xtick_list = []
    for model in model_list:
        for dataset in dataset_list:
            y1.append(result_nc[model][dataset]['ndcg'][8])
            y2.append(result[model][dataset]['ndcg'][8])
            if dataset == "ml100k":
                xtick_list.append("{}-ml".format(model))
            else:
                xtick_list.append("{}-CD".format(model))
    bar_width = 0.35
    plt.figure(figsize=(10, 10), dpi=300)
    plt.subplots_adjust(left=0.18, right=0.99, top=0.99, bottom=0.27)
    plt.tight_layout()
    plt.bar(x, y1, bar_width, label='w/o condense')
    plt.bar(x + bar_width, y2, bar_width, label='condense')
    plt.ylabel('NDCG@20', fontproperties=font)
    plt.ylim(0.48, 0.68)
    plt.xticks(x + bar_width / 2, xtick_list, fontproperties=font, rotation=90)
    plt.yticks(fontproperties=font)
    plt.legend(fontsize=28)
    for i in range(6):
        improve = round((y2[i] - y1[i]) / y1[i] * 100, 2)
        plt.text(i - 0.4, y2[i]+ 0.002, str("+{}%".format(improve)), fontproperties=font)
    plt.savefig("./img3/condense_ablation_ndcg.png")
    
    x = np.arange(6)
    y1, y2 = [], []
    xtick_list = []
    for model in model_list:
        for dataset in dataset_list:
            y1.append(result_nc[model][dataset]['bpref'][8])
            y2.append(result[model][dataset]['bpref'][8])
            if dataset == "ml100k":
                xtick_list.append("{}-ml".format(model))
            else:
                xtick_list.append("{}-CD".format(model))
    bar_width = 0.35
    plt.figure(figsize=(10, 10), dpi=300)
    plt.subplots_adjust(left=0.18, right=0.99, top=0.97, bottom=0.27)
    plt.tight_layout()
    plt.bar(x, y1, bar_width, label='w/o condense')
    plt.bar(x + bar_width, y2, bar_width, label='condense')
    plt.ylabel('BPREF@20', fontproperties=font)
    plt.ylim(0.45, 0.62)
    plt.xticks(x + bar_width / 2, xtick_list, fontproperties=font, rotation=90)
    plt.yticks(fontproperties=font)
    plt.legend(fontsize=28, loc=2)
    for i in range(6):
        improve = round((y2[i] - y1[i]) / y1[i] * 100, 2)
        plt.text(i - 0.4, y2[i]+ 0.002, str("+{}%".format(improve)), fontproperties=font)
    plt.savefig("./img3/condense_ablation_bpref.png")

def test():
    path = "./full_results/FM-cos_3_64-ml25m/ndcg.pkl"
    with open(path, "rb") as f:
        temp_dict = pickle.load(f)
    print(temp_dict[5].keys())
    sys.exit(1)

def pointwise_sensitivity():
    path1 = "./log/FM/FM-user_5k_total_amazon_1-Sep-24-2024_09-28-45-4fb58f"
    path2 = "./log/FM/FM-user_5k_total_amazon_1-Sep-28-2024_07-47-12-728700"
    with open("{}/ndcg.pkl".format(path1), "rb") as f:
        ndcg1 = pickle.load(f)
    with open("{}/ndcg.pkl".format(path2), "rb") as f:
        ndcg2 = pickle.load(f)
    with open("{}/bpref.pkl".format(path1), "rb") as f:
        bpref1 = pickle.load(f)
    with open("{}/bpref.pkl".format(path2), "rb") as f:
        bpref2 = pickle.load(f)
    
    ndcg_p, bpref_p = {}, {}
    for key in ndcg1.keys():
        ndcg_p[key], bpref_p[key] = np.zeros(6), np.zeros(6)
        for i in range(6):
            cur_ndcg1 = ndcg1[key][i]
            cur_ndcg2 = ndcg2[key][i]
            cur_bpref1 = bpref1[key][i]
            cur_bpref2 = bpref2[key][i]
        with open("./sensitivity.log", "a") as f:
            f.write("Results of top-{}".format(key))
            print("Results of top-{}".format(key))
            ndcg_list= ndcg_p[key].tolist()
            f.write("ndcg: {}\n".format(ndcg_list))
            print("ndcg: {}\n".format(ndcg_list))
            for j in range(len(ndcg_list)):
                if ndcg_list[j] <= 0.05:
                    f.write("* ")
                else:
                    f.write("- ")
            f.write("\n")

def find_best():
    path = "./full_results/temp.log"
    with open(path, "r") as f:
        lines = f.read().splitlines()
    best_ndcg, seed, best_param_dict = 0.0, 0, None
    best_dict_list = None
    for i in range(len(lines)):
        if "seed = " in lines[i]:
            seed = int(lines[i].split(" ")[-1])
        if "Above is the result of parameter" in lines[i]:
            cur_param = lines[i].split(": ", maxsplit=1)[1]
            dict_list = []
            try:
                for j in range(-7, -1):
                    curdict = lines[i + j].split("  ")[1]
                    curdict = curdict.split(": ", maxsplit=1)[1]
                    curdict = eval(curdict)
                    dict_list.append(curdict)
                if dict_list[0][20]['ndcg@20'] > best_ndcg:
                    best_ndcg = dict_list[0][20]['ndcg@20']
                    best_dict_list = copy.deepcopy(dict_list)
                    best_param_dict = copy.deepcopy(cur_param)
            except:
                continue
                
    cur_time = lines[-1].split("  ")[0]
    with open(path, "a") as f:
        f.write("{}  Best results of seed {}:\n".format(cur_time, seed))
        for i in range(6):
            if i == 0:
                f.write("{}  total test result: {}\n".format(cur_time, best_dict_list[i]))
            else:
                f.write("{}  test result of group {}: {}\n".format(cur_time, i, best_dict_list[i]))
        f.write("{}  With parameters: {}\n".format(cur_time, best_param_dict))

def calculate_average():
    path = "./full_results/FM-cos_5_64-ml25m.log"
    with open(path, "r") as f:
        lines = f.read().splitlines()
    results = [{} for _ in range(6)]
    for i in range(6):
        for length in [1, 5, 10, 20]:
            results[i][length] = {}
            results[i][length]["ndcg@{}".format(length)] = 0.0
            results[i][length]["mrr@{}".format(length)] = 0.0
            results[i][length]["bpref@{}".format(length)] = 0.0
    
    seed_num = 0
    for i in range(len(lines)):
        if "Best results of seed" in lines[i]:
            for j in range(1, 7):
                curdict = lines[i + j].split(": ", maxsplit=1)[1]
                curdict = eval(curdict)
                for length in curdict.keys():
                    for key in curdict[length].keys():
                        results[j - 1][length][key] += curdict[length][key]
            seed_num += 1
    
    for i in range(6):
        for length in results[i].keys():
            for key in results[i][length].keys():
                results[i][length][key] /= seed_num
    
    cur_time = lines[-1].split("  ")[0]
    with open(path, "a") as f:
        for i in range(6):
            if i == 0:
                f.write("{}  total average test results\n".format(cur_time))
            else:
                f.write("{}  average test result of group {}\n".format(cur_time, i))
            for length in results[i].keys():
                f.write("{}  {}\n".format(cur_time, results[i][length]))
    
if __name__ == "__main__":
    # find_best()
    # calculate_average()
    test()
    # check_results()
    # cal_ttest("./baseline_results/LightGCN/LightGCN-ml100k-cldot.log", "./baseline_results/LightGCN/LightGCN-ml100k-none.log")
    # cal_sensitivity()
    # generate_csv()
    # get_bestparam()
    
    # result_dict, best_improve = generate_csv()
    # print(best_improve['ml100k']['LightGCN'][20]['bpref'])
    # plot_improve(best_improve)
    # get_group_ablation(best_improve)
    # plot_condense_length()
    # plot_history_length()
    # get_vector_mod()
    # get_history_ablation()
    
    # plot_history_length_new()
    # calculate_token("ml100k", False)
    # calculate_token("ml100k", True)
    # calculate_token("amazon-CDs_and_Vinyl", False)
    # calculate_token("amazon-CDs_and_Vinyl", True)
    
    # get_history_ablation_new()
    
    # merge_logs()
    # compare_condense()
    
    # cal_special_sensitivity()