import sys
sys.path.append("../..")
import importlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7" # Please check GPU status before start running this file.
import requests
import yaml
from mylogging import get_logger
import requests
import json
# from alpaca_lora.generate import *
from prompt import *
import re
import openai
# import openai.error
import random
import numpy as np
from scipy import spatial
import pickle
from tqdm import trange, tqdm
import pandas as pd
import copy
import signal
from timeout_decorator import timeout, TimeoutError
import tiktoken
import shutil
import traceback
import paramiko
# import ollama
from transformers import AutoTokenizer
import threading
import ast
from openai import OpenAI
import re
import gc
vllm_client = OpenAI(
    api_key=os.getenv("UPLLM_API_KEY", "EMPTY"),
    base_url="http://localhost:8003/v1"
)
MODEL_NAME = "./models/Meta-Llama-3.1-8B-Instruct"

# 初始化embedding client（和vllm_client并列放在__main__或全局）
embed_client = OpenAI(
    api_key=os.getenv("UPLLM_API_KEY", "EMPTY"),
    base_url="http://localhost:8001/v1"
)
EMBED_MODEL_NAME = "./models/Meta-Llama-3.1-8B-Instruct"


# os.environ['OLLAMA_MAX_QUEUE'] = "4096"


def get_embedding_with_retry(client, text, max_retries=5, sleep_sec=2):
    last_err = None
    for attempt in range(max_retries):
        try:
            return get_embedding(client, text)
        except Exception as e:
            last_err = e
            print(f"[embedding retry {attempt + 1}/{max_retries}] {repr(e)}")
            time.sleep(sleep_sec)
    raise RuntimeError(f"get_embedding failed after {max_retries} retries: {repr(last_err)}")


class FunctionTimeoutError(TimeoutError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def timeout_decorator(timeout_time):
    def timeout_handler(signum, frame):
        raise FunctionTimeoutError("Function execution timed out.")
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_time)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

logger = get_logger(__name__)

logger.info = print

import time
def getTime():
    return time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))

# Embedding实际的RPM只有30，不知道为什么

class Agent:
    prompter, tokenizer, model = None, None, None
    def __init__(self, stream_output=False, llm='LLaMA-13B'):
        self.llm_model = llm
        self.set_model()
    
    def set_model(self):
        if self.llm_model == "ChatGPT":
            self.generate = self.generate_chatgpt
            openai.api_key = "Your key"
            openai.api_base = "Your base"
        elif self.llm_model == "gpt-4":
            self.generate = self.generate_gpt4
            openai.api_key = "Your key"
            openai.api_base = "Your base"
        elif self.llm_model == "llama3":
            self.generate = self.generate_llama3
            # self.client = ollama.Client(host="http://127.0.0.1:11434")
            openai.api_key = "Your key"
            openai.api_base = "Your base"
        
    
    def judge_relevance(self, embed1, embed2):
        threshold = 0.98
        cos_sim = 1 - spatial.distance.cosine(embed1, embed2)
        if cos_sim >= threshold:
            return False
        else:
            return True
    
    def generate_chatgpt(self, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
        # max_tokens = 1536
        logger.info("Query {}".format(self.llm_model))
        return get_generate(instruction, input, max_tokens)
    
    def generate_gpt4(self, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
        logger.info("Query {}".format(self.llm_model))
        return get_generate2(instruction, input, max_tokens)
    
    # def generate_llama3(self, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
    #     logger.info("Query {}".format(self.llm_model))
    #     return get_generate_llama(ollama_client, instruction, input, max_tokens)
    
    def generate_llama3(self, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
        # logger.info("Query {}".format(self.llm_model))
        return get_generate_llama(vllm_client, instruction, input, max_tokens)

@timeout_decorator(30)
def get_generate(instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
    if instruction is None:
        finalinput = input
    else:
        finalinput = instruction + input
    # time.sleep(10)
    # print("#####")
    # for i in range(len(finalinput)):
    #     print(finalinput[i])
    if max_tokens is None:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=finalinput,
            # max_tokens=1536,
            temperature=0.0
        )
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=finalinput,
            max_tokens=max_tokens,
            temperature=0.0
        )
    # time.sleep(1)
    return response.choices[0].message["content"]


# # @timeout_decorator(30)
# def get_generate_llama(client, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
#     if instruction is None:
#         finalinput = input
#     else:
#         finalinput = instruction + input
    
#     print(finalinput)
    
#     response = client.chat(model='llama3', messages=finalinput)
#     return response['message']['content']

def get_generate_llama(client, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
    if instruction is None:
        finalinput = input
    else:
        finalinput = instruction + input

    # print('##### Final Input')
    # print(finalinput)
    # print('##### End Final Input')

    kwargs = dict(
        model=MODEL_NAME,
        messages=finalinput,
        temperature=0.6,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

@timeout_decorator(120)
def get_generate2(instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
    if instruction is None:
        finalinput = input
    else:
        finalinput = instruction + input
    if max_tokens is None:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",
            messages=finalinput,
            temperature=0.0
        )
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",
            messages=finalinput,
            max_tokens=max_tokens,
            temperature=0.0
        )
    return response.choices[0].message["content"]

def get_num_tokens(messages, model="gpt-4-turbo-2024-04-09"):
    if model.startswith("gpt"):
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
    elif model.startswith("llama"):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        msgstr = ""
        for i in range(len(messages)):
            msgstr += messages[i]["content"]
        tokens = tokenizer.tokenize(msgstr)
        num_tokens = len(tokens)
    return num_tokens

class ConnectSSH:
    def __init__(self):
        jumpbox_host_ip = "101.6.41.59"
        jump_user = "your_username"
        jump_port = 11517
        target_user = "your_username"
        ssh_key_filename = os.getenv('HOME') + '/.ssh/id_rsa'
        target_host_ip = "192.168.56.26"
        
        jumpbox_ssh = paramiko.SSHClient()
        jumpbox_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jumpbox_ssh.connect(hostname=jumpbox_host_ip, port=jump_port, username=jump_user, key_filename=ssh_key_filename)
        
        jumpbox_transport = jumpbox_ssh.get_transport()
        src_addr = (jumpbox_host_ip, 11517)
        dest_addr = (target_host_ip, 22)
        jumpbox_channel = jumpbox_transport.open_channel(kind="direct-tcpip", dest_addr=dest_addr, src_addr=src_addr)
        
        target_ssh = paramiko.SSHClient()
        target_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        target_ssh.connect(hostname=target_host_ip, username=target_user, password="fit1-507", sock=jumpbox_channel)
        
        trans = target_ssh.get_transport()
        sftp = paramiko.SFTPClient.from_transport(trans)
        
        self.jumpbox_ssh = jumpbox_ssh
        self.target_ssh = target_ssh
        self.sftp = sftp
        # output = stdout.read()
        # print(output.decode('utf-8'))
    
    def execute(self, cmd):
        stdin, stdout, stderr = self.target_ssh.exec_command(command=cmd, get_pty=True)
        output = stdout.read()
        err = stderr.read()
        print("output: {}".format(output.decode('utf-8')))
        print("error: {}".format(err.decode('utf-8')))
        return output.decode('utf-8')
    
    def upload_file(self, local_path, remote_path):
        self.sftp.put(localpath=local_path, remotepath=remote_path)
    
    def download_file(self, remote_path, local_path):
        self.sftp.get(remotepath=remote_path, localpath=local_path)
    
    def close(self):
        self.target_ssh.close()
        self.jumpbox_ssh.close()
        self.sftp.close()

def connect_ssh():
    connect_obj = ConnectSSH()
    # prompt = [
    #     {"role": "system", "content": "You are a helpful assistant on movie recommendation."},
    #     {"role": "user", "content": "Please recommend 5 action movies in 2020s to me. You only need to output their names, directors and released date."},
    # ]
    # prompt = """{}""".format(prompt)
    
    # cur_index = 0
    # while cur_index < len(prompt):
    #     if prompt[cur_index] == "'":
    #         prompt = prompt[:cur_index] + "\\\"" + prompt[cur_index + 1:]
    #         cur_index += 1
    #     cur_index += 1
    # prompt = "\"\"\"" + prompt + "\"\"\""
    prompt_obj = Prompt()
    condense_prompt = prompt_obj.condense_prompt[0]
    condense_example = prompt_obj.condense_example[0]
    condense_prompt[1]["content"] += condense_example
    
    condense_list = ["age: 53", "gender: female", "occupation: other", "moderately likes Drama genre", "moderately likes Musical genre", "appreciates movies with compelling drama", "moderately likes Crime genre", "appreciates movies that combine elements of Crime and Drama genres", "moderately likes romance genre", "moderately likes thriller genre", "appreciates movies with compelling drama and romance", "likes movies that combine elements of drama and romance genres", "appreciates comedy movies with elements of drama and romance", "appreciates movies with compelling drama and romance in sci-fi and war settings", "likes movies that combine elements of drama, romance, and action genres", "appreciates comedy genre", "appreciates Drama genre", "enjoys Romance genre", "likes movies with compelling drama and romance", "appreciates comedy movies with elements of romance", "enjoys romantic comedies", "enjoys comedy movies with elements of drama and romance", "appreciates movies with elements of drama and romance", "enjoys movies with elements of drama and romance", "appreciates movies with elements of drama and romance", "appreciates movies that combine elements of drama and romance", "dislikes movies that lack compelling drama and romance", "appreciates movies that combine elements of drama, romance, and action genres", "enjoys intense action movies", "appreciates suspenseful plots", "appreciates movies with intense action and suspenseful plots", "enjoys comedy movies", "may prefer movies that combine elements of Drama, Romance, and Action genres", "may prefer movies with more elements of romance", "may prefer movies that combine multiple genres"]
    message = ""
    message += "User Original Profile: {}\n".format(condense_list)
    condense_input = [{"role": "user", "content": message}]
    condense_prompt += condense_input
    
    with open("./condense_profile.pkl", "wb") as f:
        pickle.dump(condense_prompt, f)
    connect_obj.upload_file("./condense_profile.pkl", "./llama3-main/condense_profile.pkl")
    
    output = connect_obj.execute("python ./llama3-main/connect_llama3.py")
    real_output = output['result'][0]['generation']['content']
    print(output)
    print(real_output)
    # connect_obj.download_file("./llama_output.txt", "./llama_output.txt")


def safe_parse_profile(profile_str):
    """
    处理包含撇号、特殊字符的profile字符串
    """
    if profile_str is None:
        return []
    
    # 先尝试ast.literal_eval
    try:
        return ast.literal_eval(profile_str)
    except:
        pass
    
    # 尝试修复后再parse：把内部撇号转义
    try:
        # 找到所有['...', '...']中的内容，逐个提取
        # 匹配单引号包裹的内容，允许内部有撇号（贪婪匹配两个单引号之间的内容）
        pattern = r"'((?:[^'\\]|(?<=\w)'(?=\w)|\\.)*)'"
        matches = re.findall(pattern, profile_str)
        if matches:
            return matches
    except:
        pass
    
    # 兜底：按逗号分割，手动清理
    try:
        # 去掉首尾的[]
        stripped = profile_str.strip()
        if stripped.startswith('['):
            stripped = stripped[1:]
        if stripped.endswith(']'):
            stripped = stripped[:-1]
        
        # 按', '分割后清理引号
        items = re.split(r",\s*", stripped)
        result = []
        for item in items:
            item = item.strip()
            # 去掉首尾引号（单引号或双引号）
            if (item.startswith("'") and item.endswith("'")) or \
               (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            if item:
                result.append(item)
        return result
    except:
        return []


from pathlib import Path


class History:
    def __init__(self):
        pass
    
    @staticmethod
    def create_train(file_name, time):
        obj = History()
        obj.content = []
        obj.time = time
        obj.file_name = f"logs/{obj.time}/{file_name}"

        Path(f"logs/{obj.time}/user").mkdir(parents=True, exist_ok=True)
        Path(f"logs/{obj.time}/item").mkdir(parents=True, exist_ok=True)
        Path(f"logs/{obj.time}/checkpoints").mkdir(parents=True, exist_ok=True)

        Path(obj.file_name).parent.mkdir(parents=True, exist_ok=True)
        Path(obj.file_name).write_text("")
        return obj

    # @staticmethod
    # def create_train(file_name, time):
    #     obj = History()
    #     obj.content = []
    #     obj.time=time
    #     obj.file_name =f"logs/{obj.time}/{file_name}"
    #     # print(obj.file_name)
    #     if not os.path.exists(f"logs/{obj.time}"):
    #         os.mkdir(f"logs/{obj.time}")
    #         os.mkdir(f"logs/{obj.time}/user")
    #         os.mkdir(f"logs/{obj.time}/item")
    #         os.mkdir(f"logs/{obj.time}/checkpoints")
    #     obj.f = open(f'{obj.file_name}', 'w')
    #     obj.f.close()
    #     return obj

    @staticmethod
    def create_eval(file_name):
        obj = History()
        obj.content = []
        obj.file_name = file_name
        print(obj.file_name)
        with open(obj.file_name, "w") as f:
            pass
        return obj

    # def append(self, object):
    #     str_obj = json.dumps(object)
    #     logger.info(str_obj)
    #     self.content.append(object)
    #     # from IPython import embed; embed()
    #     try:
    #         with open(f"{self.file_name}", 'a') as f:
    #             f.write(str_obj+"\n")
    #     except TypeError:
    #         from IPython import embed; embed()
    def append(self, object):
        str_obj = json.dumps(object)
        # logger.info(str_obj)
        try:
            with open(self.file_name, 'a') as f:
                f.write(str_obj + "\n")
        except TypeError as e:
            print("TypeError in History.append:", repr(e))
            print("file_name:", self.file_name)
            print("object:", object)
            raise e

    def append_batch(self, objects):
        lines = []
        for obj in objects:
            try:
                lines.append(json.dumps(obj))
            except TypeError as e:
                print("TypeError in History.append_batch:", repr(e))
                print("file_name:", self.file_name)
                print("object:", obj)
                raise e

        with open(self.file_name, 'a') as f:
            f.write("\n".join(lines) + "\n")

class User(Agent):
    def __init__(self, meta, log=True, time=getTime(), llm='LLaMA-13B', singleembed=[], totalembed=[], dataset="ml100k", msg_index=0): # TODO: 拆分一下构造函数，初次初始化和再初始化分开。
        super().__init__(llm=llm)
        # print(meta)
        self.profile_max_length = profile_max_length
        self.properties=meta.to_dict()
        self.log=log
        # print('[user init]: ',self.properties)
        self.user_id = meta['user_id']
        self.dataset = dataset
        
        if self.log:
            # print("user/{}.jsonl".format(self.user_id))
            self.history = History.create_train(file_name="user/{}.jsonl".format(self.user_id), time=time)
        
        if dataset == "ml100k":
            self.init_profile_ml100k()
        elif "amazon" in dataset:
            self.init_profile_amazon()
        elif dataset == "ml-25m":
            self.init_profile_ml25m()
        
        self.prompt = Prompt()
        
        self.singleembed = singleembed
        self.totalembed = totalembed
        
        self.system_msg = self.prompt.system_msg[msg_index]
        
        # if msg_index == 2:
        #     self.max_token = 1536
        # else:
        #     self.max_token = None
        
        # self.max_token = 1536
        self.max_token = 2048
        
        self.initial_profile = copy.deepcopy(self.profile)
        self.initial_singleembed = copy.deepcopy(self.singleembed)
        self.initial_totalembed = copy.deepcopy(self.totalembed)
        
        if llm == 'ChatGPT':
            self.condense = self.condense_traditional
        else:
            self.condense = self.condense_llm
        
        # print(self.system_msg, self.max_token)
        # sys.exit(1)

    def init_profile_ml100k(self):
        self.profile = []
        self.profile.append("age: {}".format(self.properties['age']))
        self.profile.append("gender: {}".format("male" if self.properties['gender'] == "M" else "female"))
        self.profile.append("occupation: {}".format(self.properties['occupation']))
        if self.log:
            self.history.append({"Profile": self.profile})
    
    def init_profile_amazon(self):
        self.profile = []
        self.profile.append(self.properties['profile'])
        # if self.log:
        #     self.history.append({"Profile": self.profile})
    
    def init_profile_ml25m(self):
        self.profile = []
        self.profile.append(self.properties['profile'])

    # def condense_traditional(self):
    #     def cal_minmax(eval_dict:dict) -> dict:
    #         maxnum, minnum = -1, 99999
    #         for key in eval_dict.keys():
    #             if eval_dict[key] > maxnum:
    #                 maxnum = eval_dict[key]
    #             if eval_dict[key] < minnum:
    #                 minnum = eval_dict[key]
    #         if maxnum == minnum:
    #             return eval_dict
    #         for key in eval_dict.keys():
    #             eval_dict[key] = (eval_dict[key] - minnum) / (maxnum - minnum)
    #         return eval_dict
        
    #     def cal_score(index:int, freq_dict:dict, unique_dict:dict, history_dict:list[dict]) -> float:
    #         score = freq_dict[index] / 3 + unique_dict[index] / 3
    #         score2 = 0.0
    #         for i in range(len(history_dict)):
    #             score2 += history_dict[i][index] / 3
    #         score += score2 / 3
    #         return score
        
    #     condensing_rate = 0.3
    #     condensing_len = int(len(self.profile) * (1 - condensing_rate))
        
    #     # remove conflicting terms
    #     np.random.seed(int(time.time() * 1e6 % 1e6))
    #     self.conflict_example = self.prompt.conflict_example[np.random.randint(0, len(self.prompt.conflict_example))]
    #     self.conflict_prefix = copy.deepcopy(self.prompt.conflict_prompt[-1])
    #     self.conflict_prefix = self.system_msg + self.conflict_prefix
    #     self.conflict_prefix[1]['content'] += self.conflict_example
        
    #     message = ""
    #     message += "User Original Profile: {}\n".format(self.profile)
    #     condense_input = [{"role": "user", "content": message}]
    #     self.conflict_prefix += condense_input
        
    #     with open("./condense_profile.pkl", "wb") as f:
    #         pickle.dump(self.conflict_prefix, f)
    #     connect_obj.upload_file("./condense_profile.pkl", "./llama3-main/condense_profile.pkl")
    #     connect_obj.execute("python ./llama3-main/connect_llama3.py")
    #     connect_obj.download_file("./llama3-main/condensed_profile.pkl", "./condensed_profile.pkl")
    #     with open("./condensed_profile.pkl", "rb") as f:
    #         output = pickle.load(f)
    #     output = output['result'][0]['generation']['content']
    #     print(output)
        
    #     updated_profile = self.extract_profile(output)
    #     updated_profile = updated_profile.replace(".]","]")
    #     updated_profile = eval(updated_profile)
    #     updated_singleembed = []
    #     for i in range(len(updated_profile)):
    #         if updated_profile[i] in self.profile:
    #             index = self.profile.index(updated_profile[i])
    #             updated_singleembed.append(self.singleembed[index])
    #         else:
    #             updated_singleembed.append(get_embedding(updated_profile[i]))

    #     while (len(updated_profile) > condensing_len):
    #         sim_dict, freq_dict, unique_dict, history_dict = {}, {}, {}, [{}, {}, {}]
    #         # 基于sim_dict，同时考虑到frequency（越低越好）、uniqueness（越低越好）和与history的匹配程度（越高越好）
    #         for i in range(len(updated_profile)):
    #             for j in range(i + 1, len(updated_profile)):
    #                 sim_dict[(i, j)] = 1 - spatial.distance.cosine(updated_singleembed[i], updated_singleembed[j])
    #         sim_dict2 = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)
    #         high1, high2 = sim_dict2[0][0][0], sim_dict2[0][0][1]
            
    #         for i in range(len(updated_profile)):
    #             freq_dict[i] = 0
    #             for j in range(len(updated_profile)):
    #                 if j < i:
    #                     freq_dict[i] += sim_dict[(j, i)]
    #                 elif j > i:
    #                     freq_dict[i] += sim_dict[(i, j)]
    #                 else:
    #                     continue
    #             freq_dict[i] /= (len(updated_profile) - 1)
    #         freq_dict = cal_minmax(freq_dict)
            
    #         # for i in range(len(sim_dict) // 2):
    #         #     if sim_dict[i][0][0] not in freq_dict.keys():
    #         #         freq_dict[sim_dict[i][0][0]] = 0
    #         #     if sim_dict[i][0][1] not in freq_dict.keys():
    #         #         freq_dict[sim_dict[i][0][1]] = 0
    #         #     freq_dict[sim_dict[i][0][0]] += 1
    #         #     freq_dict[sim_dict[i][0][1]] += 1
    #         # freq_dict = cal_minmax(freq_dict)
            
    #         for i in range(len(updated_profile)):
    #             cur_profile = updated_profile[i].split(" ")
    #             count_dict = {}
    #             for k1 in range(len(cur_profile)):
    #                 count_dict[cur_profile[k1].lower()] = 0
    #                 for j in range(len(updated_profile)): # 包括自身
    #                     other_profile = updated_profile[j].split(" ")
    #                     for k2 in range(len(other_profile)):
    #                         if cur_profile[k1].lower() == other_profile[k2].lower():
    #                             count_dict[cur_profile[k1].lower()] += 1
    #                             break
    #             min_freq = 99999
    #             for key in count_dict.keys():
    #                 if count_dict[key] < min_freq:
    #                     min_freq = count_dict[key]
    #             unique_dict[i] = min_freq
    #         unique_dict = cal_minmax(unique_dict)
            
    #         inter_category = []
    #         for i in range(user_count):
    #             cur_itemid = train_list[user_count]['item_id']
    #             if self.dataset == "ml100k":
    #                 temp_list = item_pool[cur_itemid].properties["category"].split(".")[0].split(",")
    #             elif "amazon" in self.dataset:
    #                 temp_list = str(item_pool[cur_itemid].properties["category"]).split(".")
    #             temp_list = [x.strip() for x in temp_list]
    #             temp_list = list(filter(None, temp_list))
    #             temp_list = [x.lower() for x in temp_list]
    #             inter_category.append(temp_list)
    #         for i in range(len(updated_profile)):
    #             phrase_level, word_level, word_totalmatch = 0, 0, 0
    #             # cur_profile = updated_profile[i].split(" ")
    #             # cur_profile = [x.lower() for x in cur_profile]
    #             cur_profile = updated_profile[i].lower()
    #             for j in range(len(inter_category)):
    #                 for k in range(len(inter_category[j])):
    #                     if inter_category[j][k] in cur_profile:
    #                         phrase_level += 1
    #                     separate = inter_category[j][k].split(" ")
    #                     flag = False
    #                     for l in range(len(separate)):
    #                         if separate[l] in cur_profile:
    #                             flag = True
    #                             word_totalmatch += 1
    #                     if flag:
    #                         word_level += 1
    #             history_dict[0][i] = phrase_level
    #             history_dict[1][i] = word_level
    #             history_dict[2][i] = word_totalmatch
    #         for i in range(len(history_dict)):
    #             history_dict[i] = cal_minmax(history_dict[i])
    #             for key in history_dict[i].keys():
    #                 history_dict[i][key] = 1 - history_dict[i][key]
            
    #         h1_score, h2_score = cal_score(high1, freq_dict, unique_dict, history_dict), cal_score(high2, freq_dict, unique_dict, history_dict)
    #         if h1_score < h2_score:
    #             high1, high2 = high2, high1
    #         updated_profile.pop(high1)
    #         updated_singleembed.pop(high1)
        
    #     self.profile = copy.deepcopy(updated_profile)
    #     self.singleembed = copy.deepcopy(updated_singleembed)
    #     self.profile_next = ''
    #     self.totalembed = get_embedding(list2sentence(self.profile))
    #     self.history.append({
    #             "***[User condense]***": self.profile,
    #         })
        
    def condense_llm(self, ):
        np.random.seed(int(time.time() * 1e6 % 1e6))
        # self.condense_example = self.prompt.condense_example[np.random.randint(0, len(self.prompt.condense_example))]
        self.condense_example = self.prompt.condense_example[-1]

        self.condense_prefix = copy.deepcopy(self.prompt.condense_prompt[0])
        self.condense_prefix = self.system_msg + self.condense_prefix
        self.condense_prefix[1]['content'] += self.condense_example
        
        message = ""
        message += "User Original Profile: {}\n".format(self.profile)
        condense_input = [{"role": "user", "content": message}]
        # self.condense_instance_prompt = self.condense_prefix + condense_input #message
        # ret =self.generate(self.condense_instance_prompt)
        ret = self.generate(self.condense_prefix, condense_input, self.max_token)

        # if self.log:
        #     self.history.append({
        #         "***[User condense]***": ret,
        #     })

        updated_profile = self.extract_profile(ret)
        self.profile_next = ''
        updated_profile = updated_profile.replace(".]","]")

        self.profile = safe_parse_profile(updated_profile)
        if not self.profile:
            retry_times = 0
            while retry_times < 3:
                logger.info("Retrying to parse profile, attempt #{}".format(retry_times + 1))
                ret = self.generate(self.condense_prefix, condense_input, self.max_token)
                updated_profile = self.extract_profile(ret)
                updated_profile = updated_profile.replace(".]","]")
                self.profile = safe_parse_profile(updated_profile)
                if self.profile:
                    break
                retry_times += 1
            error_infos = [{"Error Type": "Condensing Error"}]
            error_infos.append({"Error Info": "Failed to parse profile: {}".format(updated_profile)})
            error_infos.append({"Error Info": "Model Input: {}".format(condense_input)})
            error_infos.append({"Error Info": "Model Output: {}".format(ret)})
            self.history.append_batch(error_infos)
            # self.history.append({"Error Info": "Failed to parse profile: {}".format(updated_profile)})
            # self.history.append({"Error Info": "Model Output: {}".format(ret)})
            logger.info("Failed to parse profile: {}".format(updated_profile))
            logger.info("Model Output: {}".format(ret))
            self.profile = self.last_profile  # 回退到上一个profile
        # try:
        #     self.profile = eval(updated_profile)
        # except:
        #     self.profile = ast.literal_eval(updated_profile)
        
        self.singleembed = []
        for i in range(len(self.profile)):
            if self.profile[i] not in self.last_profile:
                self.singleembed.append(get_embedding(ollama_client, self.profile[i]))
            else:
                last_index = self.last_profile.index(self.profile[i])
                self.singleembed.append(self.last_singleembed[last_index])
                # self.singleembed.append(self.last_singleembed[i])
        self.totalembed = get_embedding(ollama_client, list2sentence(self.profile))
            
        self.history.append({"ModelInput:": self.condense_prefix + condense_input})
        self.history.append({
                "***[User condense]***": self.profile,
            })

    def extract_profile(self, string):
        marker = "New User Profile:"
        start = string.find(marker)
        if start == -1:
            return None

        # 从 marker 后的第一个 '[' 开始，提取到与之匹配的第一个完整 ']'
        left = string.find("[", start + len(marker))
        if left == -1:
            return None

        in_quote = False
        escape = False
        depth = 0
        for i in range(left, len(string)):
            ch = string[i]
            if in_quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_quote = False
                continue

            if ch == '"':
                in_quote = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return string[left:i + 1]

        return None

    def view_item(self, movie_list, score_list):
        self.last_profile = copy.deepcopy(self.profile)
        self.last_singleembed = copy.deepcopy(self.singleembed)
        self.last_totalembed = copy.deepcopy(self.totalembed)
        
        self.prefix = copy.deepcopy(self.prompt.user_prompt[-1])
        # self.prefix = copy.deepcopy(self.prompt.baseline_user_prompt2[-1])
        np.random.seed(int(time.time() * 1e6 % 1e6))
        self.user_example = self.prompt.user_example[np.random.randint(0, len(self.prompt.user_example))]

        self.prefix[1]['content'] = self.prefix[1]['content'].replace('{example}', self.user_example)
        # self.prefix = self.system_msg + self.prefix
        # self.prefix[1]['content'] += self.user_example
        
        # add viewing history
        log_entries = []
        if self.log:
            for i in range(len(movie_list)):
                log_entries.append({
                # self.history.append({
                    "Interaction": {
                        "item.name": movie_list[i].name, 
                        "item.profile": movie_list[i].profile,
                        "rating": str(score_list[i]),
                    }
                })

        # def cfscore_index(a):
        #     if 0.95 <= a:
        #         scale = "Greatest Imaginable Like"
        #     elif 0.80 <= a < 0.95:
        #         scale = "Like Extremely"
        #     elif 0.60 <= a < 0.80:
        #         scale = "Like Very Much"
        #     elif 0.40 <= a < 0.60:
        #         scale = "Like Moderately"
        #     elif 0.10 <= a < 0.40:
        #         scale = "Like Slightly"
        #     elif -0.10 <= a < 0.10:
        #         scale = "Neither Like nor Dislike"
        #     elif -0.40 <= a < -0.10:
        #         scale = "Dislike Slightly"
        #     elif -0.60 <= a < -0.40:
        #         scale = "Dislike Moderately"
        #     elif -0.80 <= a < -0.60:
        #         scale = "Dislike Very Much"
        #     elif -0.95 <= a < -0.80:
        #         scale = "Dislike Extremely"
        #     elif a <= -0.95:
        #         scale = "Greatest Imaginable Dislike"
        #     return scale
        def cfscore_index(a):
            if 0.10 <= a:
                scale = "Like this item"
            elif 0 <= a < 0.10:
                scale = "Neutral towards like"
            elif -0.10 <= a < 0:
                scale = "Neutral towards dislike"
            elif a < -0.10:
                scale = "Dislike this item"
            return scale

        message = ""
        message += "User Original Profile:\n {}\n\n".format(self.profile)
        message += "Current session interactions:\n\n"
        for i in range(len(movie_list)):
            message += "Item #{}: {}\n".format(i + 1, movie_list[i].name)
            message += "Item Profile: {}\n".format(movie_list[i].profile)
            message += "Rating: {}/5.0\n\n".format(str(score_list[i]))
            # cur_userindex, cur_itemindex = user_mapping[str(self.user_id)], item_mapping[str(movie_list[i].item_id)]
            # # message += "Collaborative Score: {}\n".format(cosmatrix[cur_userindex][cur_itemindex])
            # message += "Collaborative Hedonic Scale: {}\n\n".format(cfscore_index(cosmatrix[cur_userindex][cur_itemindex]))


        instance_input = [{"role": "user", "content": message}]
        # input_msg = self.prefix[1]['content'].format(content=message)
        # instance_input = [{"role": "user", "content": input_msg}]
        # self.prefix = self.prefix[:-1]

        # self.instance_prompt = self.prefix + [{"role": "user", "content": message}]
        # ret = self.generate(self.instance_prompt)
        # print("##### {}".format(self.prefix))
        # print("##### {}".format(instance_input))
        # print(type(self.prefix), type(instance_input))
        ret = self.generate(self.prefix, instance_input, self.max_token)
        # print("##### {}".format(ret))
        # time.sleep(1000)
        if self.log:
            log_entries.append({
                "ModelInput": self.prefix + instance_input,
            })
            log_entries.append({
            # self.history.append({
                "ModelOutput": ret,
            })
            self.history.append_batch(log_entries)
            
        updated_profile = self.extract_profile(ret)
        # updated_profile = updated_profile.replace('\'', '"')
        # print("@@@@@ {}".format(updated_profile))

        self.profile_next = safe_parse_profile(updated_profile)
        if not self.profile_next:
            retry_times = 0
            while retry_times < 3:
                logger.info("Retrying to parse profile, attempt #{}".format(retry_times + 1))
                ret = self.generate(self.prefix, instance_input, self.max_token)
                updated_profile = self.extract_profile(ret)
                # updated_profile = updated_profile.replace(".]","]")
                self.profile_next = safe_parse_profile(updated_profile)
                if self.profile_next:
                    break
                retry_times += 1
                logger.info("Failed to parse profile: {}".format(updated_profile))
                logger.info("Model Output: {}".format(ret))
            if not self.profile_next:
                error_infos = []
                error_infos.append({"Error Info": "Failed to parse profile: {}".format(updated_profile)})
                error_infos.append({"Error Info": "Model Input: {}".format(instance_input)})
                error_infos.append({"Error Info": "Model Output: {}".format(ret)})
                self.history.append_batch(error_infos)
            # self.history.append({"Error Info": "Failed to parse profile: {}".format(updated_profile)})
            # self.history.append({"Error Info": "Model Output: {}".format(ret)})

        # try:
        #     self.profile_next = eval(updated_profile) #json.loads(updated_profile)
        # except Exception as e:
        #     if isinstance(e, json.decoder.JSONDecodeError):
        #         print(updated_profile)
        #         from IPython import embed; embed()
        #     else:
        #         self.profile_next = ast.literal_eval(updated_profile)

        # self.condense()
    
    def update_profile(self,):
        # if len(self.profile)>0 and self.profile[0] in self.profile_next:
        #     print("[duplicate] profile: ",self.profile,"profile_next: ",self.profile_next)
        #     self.profile=self.profile_next
        # else:
        #     self.profile = self.profile + self.profile_next
        # print(type(self.profile)) # list
        # time.sleep(1000)
        # print(self.profile, self.profile_next)
        
        
        if self.profile is not None and len(self.profile) > 0:
            # for i in range(len(self.profile)):
            #     self.profile[i] = self.profile[i].strip()
            for i in range(len(self.profile_next)):
                self.profile_next[i] = self.profile_next[i].strip()
                temp_embed = get_embedding(ollama_client, self.profile_next[i])
                add = True
                for j in range(len(self.singleembed)):
                    if not self.judge_relevance(self.singleembed[j], temp_embed):
                        add = False
                        break
                if add:
                    self.profile.append(self.profile_next[i])
                    self.singleembed.append(temp_embed)
            # self.profile = list(set(self.profile).union(set(self.profile_next)))
            # self.profile.sort()
            
        else:
            self.profile = self.profile_next
            self.singleembed = []
            for i in range(len(self.profile)):
                self.singleembed.append(get_embedding(ollama_client, self.profile[i]))
        
        self.totalembed = get_embedding(ollama_client, list2sentence(self.profile))
            
        self.profile_next = ''
        if self.log:
            self.history.append({
                "Profile": self.profile,
            })

class Item(Agent):
    def __init__(self, meta, log=True, time=getTime(), llm='LLaMA-13B', singleembed=[], totalembed=[], dataset="ml100k"):
        super().__init__(llm=llm)
        # print(meta)
        self.properties = meta.to_dict() # ["item_id", "name", "category", "description"]
        self.log=log
        # print('[item init]: ',self.properties)
        self.item_id = meta['item_id']
        self.dataset = dataset
        
        if self.log:
            self.history = History.create_train(file_name="item/{}.jsonl".format(self.item_id),time=time)
        self.init_profile()

        self.prefix = [
{"role": "system", "content": "You are a helpful assistant on recommendation."},
{"role": "user", "content": """Now I will give you a profile of a movie. The relevance degree of movie profile and user profile will help the recommendation system to determine whether a user would like the movie.  Each time a user will interact with the movie, you will have a chance to see (1) the rating that the user give to the movie and (2) the user's attribute. These two information will help you update the profile of the movie to be more accurate. The profile is a list of short descriptive terms that can describe the movie. Some terms with keyword:argument structure like "Comedy:1" are the initial profile. '1' denote the movie has the feature, '0' denotes the movie doesn't have the feature. An example would be:

Movie Profile: ["title:Toy Story (1995)", "release_date:01-Jan-1995", "Action:0", "Adventure:0", "Children:1", "Comedy:1", "Crime:0", "Documentary:0", "Drama:0", "Fantasy:0"]
User Profile: ["13", "male", "child", "love science", "dislike history"]
Rating: 3/5
Thought: This user is a boy, he gives this movie a moderate score, which means he might not very like the movie. Therefore, a children who like science might not prefer this movie".
Updated Movie Profile: ["title:Toy Story (1995)", "release_date:01-Jan-1995", "Action:0", "Adventure:0", "Children:1", "Comedy:1", "Crime:0", "Documentary:0", "Drama:0", "Fantasy:0", "A child who like science might not prefer"]
"""},
{"role": "assistant", "content": "Sure, I will update the movie's profile after I see the rating and the user's attribute. I will firstly give a Thought, and then give the Updated Movie Profile. Now can you give me a movie's profile, the user who watch this movie profile, and the rating?"}
]
        # self.init_term(file_list)
        
        if singleembed is not None:
            self.singleembed = singleembed
        else:
            self.singleembed = None
        self.totalembed = totalembed
        
        self.initial_profile = copy.deepcopy(self.profile)
        if self.singleembed is not None:
            self.initial_singleembed = copy.deepcopy(self.singleembed)
        self.initial_totalembed = copy.deepcopy(self.totalembed)

    def init_profile(self):
        self.name = self.properties["name"]
        self.profile = []
        self.profile.append(self.name)
        
        if self.dataset == "ml100k":
            temp_list = self.properties["category"].split(".")[0].split(",")
        elif "amazon" in self.dataset:
            temp_list = str(self.properties["category"]).split(".")
        elif self.dataset == "ml-25m":
            temp_list = str(self.properties["category"]).split(".")
        
        temp_list = [x.strip() for x in temp_list]
        temp_list = list(filter(None, temp_list))
        self.profile += temp_list
        
        if "amazon" in self.dataset:
            self.profile += [self.properties["description"]]
        elif "ml-25m" in self.dataset:
            self.profile += [self.properties["description"]]
        
        if self.log:
            self.history.append({"Profile": self.profile})

    def condense(self, ): # Item profile is temporarily not been condensed.
        return
        self.condense_prefix = [
{"role": "system", "content": "You are a helpful assistant on recommendation."},
{"role": "user", "content": """Now I will give you a profile of a user. The profile might be too long because each time the user interact with an item, we will append some description terms. Please help me summarize the profile into a shorter one. Note that the attribute at the end of the profile might be more updated and important. Here is an example:

Original profile: ["xxx"]
Thought: this person has xxx
Updated profile: ["yyyy"]
"""},
{"role": "assistant", "content": "Sure, I will try my best to summarize the user's profile. "}
]
        message = ""
        message += "User profile: {}\n".format(self.profile)
        self.condense_instance_prompt = self.condense_prefix + message
        ret = self.generate(self.condense_instance_prompt)




    def extract_profile(self, string):
        pattern = r'(?<=Updated Movie Profile:).*'

        match = re.search(pattern, string)

        if match:
            result = match.group(0)
            logger.info("Updated Movie Profile !!! {}".format(result))
        else:
            result = None
        return result



    def view_by_user(self, user, score):
        self.last_profile = copy.deepcopy(self.profile)
        # if self.singleembed is not None:
        #     self.last_singleembed = copy.deepcopy(self.singleembed)
        # self.last_totalembed = copy.deepcopy(self.totalembed)
        # print("Next function")
        # time.sleep(1000)
        # add viewing history
        if self.log:
            self.history.append({
                "Interaction": {
                    "user.profile": user.profile,
                    "rating": str(score),
                }
            })

        message = ""
        message += "Movie profile: {}\n".format(self.profile)
        message += "User profile: {}\n".format(user.profile)
        message += "Rating: {}/5".format(score)


        self.instance_prompt = self.prefix + [{"role": "user", "content": message}]
        ret = self.generate(self.instance_prompt)

        if self.log:
            self.history.append({
                "ModelOutput": ret,
            })

        updated_profile = self.extract_profile(ret)
        self.profile_next = updated_profile # 防止同一轮中，movie看到的是 user看过这个movie之后的profile
   
        

        self.condense()
    
    def update_profile(self,):
        self.profile = self.profile_next
        self.profile_next = ""

        if self.log:
            self.history.append({
                "Profile": self.profile,
            })

def interaction(user: User, item:list[Item], score: list[float], update_terms=["user", "item"]):
    if "user" in update_terms:
        user.view_item(item, score)   
    if "item" in update_terms:
        item.view_by_user(user, score)
    if "user" in update_terms:
        user.update_profile()
    if "item" in update_terms:
        item.update_profile()
    if "user" in update_terms and len(user.profile) > user.profile_max_length:
        user.condense()

def explode_condense(user: User, update_terms=["user", "item"]):
    if "user" in update_terms:
        user.condense()

def rollback(user:User, item:Item, update_terms=["user", "item"]):
    if "user" in update_terms:
        user.profile = copy.deepcopy(user.last_profile)
        user.singleembed = copy.deepcopy(user.last_singleembed)
        user.totalembed = copy.deepcopy(user.last_totalembed)
    if "item" in update_terms:
        item.profile = copy.deepcopy(item.last_profile)
        # item.singleembed = copy.deepcopy(item.last_singleembed)
        # item.totalembed = copy.deepcopy(item.last_totalembed)

def rollback_all(user:User, item:Item, update_terms=["user", "item"]):
    if "user" in update_terms:
        user.profile = copy.deepcopy(user.initial_profile)
        user.singleembed = copy.deepcopy(user.initial_singleembed)
        user.totalembed = copy.deepcopy(user.initial_totalembed)
    if "item" in update_terms:
        item.profile = copy.deepcopy(item.initial_profile)
        item.singleembed = copy.deepcopy(item.initial_singleembed)
        item.totalembed = copy.deepcopy(item.initial_totalembed)

def load_meta_data(update_terms=["user", "item"], runtime=getTime(), llm='LLaMA-13B', dataset_tuple=(0, "ml100k")):
    msg_index, dataset = dataset_tuple
    
    if not os.path.exists("temp"):
        os.mkdir("temp")
    
    from load import read_log, read_meta

    item_meta, user_meta = read_meta(dataset)

    user_pool = {}
    # log_user = False
    log_user=bool('user' in update_terms)
    
    if dataset == "ml100k":
        with open("./Embeddings/ml100k/user_singleembed.pkl", "rb") as f:
            user_singleembed = pickle.load(f)
        with open("./Embeddings/ml100k/user_totalembed.pkl", "rb") as f:
            user_totalembed = pickle.load(f)
        # with open("./Embeddings/ml100k/item_singleembed.pkl", "rb") as f:
        #     item_singleembed = pickle.load(f)
        with open("./Embeddings/ml100k/item_totalembed.pkl", "rb") as f:
            item_totalembed = pickle.load(f)
    elif "amazon" in dataset:
        dataset_type = dataset.split("-")[1]
        with open("./Embeddings/amazon/{}/user_singleembed.pkl".format(dataset_type), "rb") as f:
            user_singleembed = pickle.load(f)
        with open("./Embeddings/amazon/{}/user_totalembed.pkl".format(dataset_type), "rb") as f:
            user_totalembed = pickle.load(f)
        # with open("./Embeddings/amazon/{}/item_singleembed.pkl".format(dataset_type), "rb") as f:
        #     item_singleembed = pickle.load(f)
        with open("./Embeddings/amazon/{}/item_totalembed.pkl".format(dataset_type), "rb") as f:
            item_totalembed = pickle.load(f)
    elif dataset == "ml-25m":
        with open("./Embeddings/ml-25m/user_singleembed.pkl", "rb") as f:
            user_singleembed = pickle.load(f)
        with open("./Embeddings/ml-25m/user_totalembed.pkl", "rb") as f:
            user_totalembed = pickle.load(f)
        with open("./Embeddings/ml-25m/item_totalembed.pkl", "rb") as f:
            item_totalembed = pickle.load(f)
    
    print("Loading user meta data")
    with tqdm(total=user_meta.shape[0]) as pbar:
        for index, row in user_meta.iterrows():
            cur_singleembed = user_singleembed[row["user_id"]]
            cur_totalembed = user_totalembed[index]
            # cur_singleembed, cur_totalembed = None, None
            u = User(row, log=log_user, time=runtime, llm=llm, singleembed=cur_singleembed, totalembed=cur_totalembed, dataset=dataset, msg_index=msg_index)
            user_pool[row["user_id"]] = u
            pbar.update()
    
    item_pool={}
    log_item=bool('item' in update_terms)
    
    print("Loading item meta data")
    with tqdm(total=item_meta.shape[0]) as pbar:
        for index, row in item_meta.iterrows():
            if dataset == "ml100k":
                # print(item_singleembed)
                # cur_singleembed = item_singleembed[row["item_id"]]
                cur_singleembed = None
                cur_totalembed = item_totalembed[index]
            elif "amazon" in dataset:
                cur_singleembed = None
                cur_totalembed = item_totalembed[index]
            # cur_singleembed, cur_totalembed = None, None
            i = Item(row, log=log_item, time=runtime, llm=llm, singleembed=cur_singleembed, totalembed=cur_totalembed, dataset=dataset)
            item_pool[row["item_id"]] = i
            pbar.update()

    interaction_train, interaction_test = read_log(dataset) # every users' rating history
    
    return user_pool, item_pool, interaction_train, interaction_test

def extract_same_user_interaction(interaction_train: pd.DataFrame, interaction_test: pd.DataFrame, user_id: int):
    ret_train = interaction_train[interaction_train["user_id"] == user_id]
    ret_train.reset_index(drop=True, inplace=True)
    ret_test = interaction_test[interaction_test["user_id"] == user_id]
    ret_test.reset_index(drop=True, inplace=True)
    # logger.info("Length of interation: {}".format(len(ret)))
    return ret_train, ret_test

def get_negative(interaction_log_user: pd.DataFrame):
    indexes = []
    for index, row in interaction_log_user.iterrows():
        if row['rating'] <= 2:
            indexes.append(index)
    negative_interaction = interaction_log_user.iloc[indexes]
    negative_interaction.reset_index(drop=True, inplace=True)
    return negative_interaction

# @timeout_decorator(10)
# def get_embedding(sentence):
#     result = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=sentence
#     )
#     return result["data"][0]["embedding"]

# @timeout_decorator(10)
# def get_embedding2(sentence):
#     result = openai.Embedding.create(
#         model="text-embedding-3-large",
#         input=sentence
#     )
#     return result["data"][0]["embedding"]

# # @timeout_decorator(10)
# def get_embedding(client, sentence):
#     result = client.embeddings(
#         model='llama3',
#         prompt=sentence
#     )
#     return result['embedding']

# def get_embedding(client, sentence):
#     # client 参数保留以兼容现有调用，实际不再使用
#     model = get_embedding_model()
#     embedding = model.encode(sentence, normalize_embeddings=True)
#     return embedding.tolist()


def get_embedding(client, sentence):
    # client参数保留兼容现有调用签名，实际使用全局embed_client
    response = embed_client.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=sentence
    )
    return response.data[0].embedding

def list2sentence(profile_list):
    sentence = ""
    for i in range(len(profile_list)):
        sentence += profile_list[i]
        if i != len(profile_list) - 1:
            sentence += ", "
        else:
            sentence += "."
    return sentence

def ndcg_naive(test_tag_sorted): # Using when there is only one positive item, and use 0/1 to represent the score.
    for i in range(len(test_tag_sorted)):
        if test_tag_sorted[i] == 1:
            return float(1 / np.log2(i + 2))

def mrr(test_tag_sorted):
    for i in range(len(test_tag_sorted)):
        if test_tag_sorted[i] == 1:
            return 1 / (i + 1)

def get_ndcg_mrr(zipped_list):
    count = 0
    dcg, idcg, mrr = 0.0, 0.0, 0.0
    for i in range(len(zipped_list)):
        if zipped_list[i][0] >= 4:
            count += 1
            dcg += float(1 / np.log2(i + 2))
            mrr += float(1 / (i + 1))
    for i in range(count):
        idcg += float(1 / np.log2(i + 2))
    mrr /= count
    return dcg / idcg, mrr

def get_bpref(zipped_list):
    bpref_score = 0.0
    pos_num, neg_num = 0, 0
    negnum_list = np.zeros(len(zipped_list))
    for i in range(len(zipped_list)):
        if i >= 1 and 0 < zipped_list[i - 1][0] <= 3:
            negnum_list[i] = negnum_list[i - 1] + 1
        else:
            negnum_list[i] = negnum_list[i - 1]
        if zipped_list[i][0] >= 4:
            pos_num += 1
        else:
            neg_num += 1
    denominator = min(pos_num, neg_num)
    for i in range(len(zipped_list)):
        if zipped_list[i][0] >= 4:
            bpref_score += (1 - negnum_list[i] / denominator)
    bpref_score /= pos_num
    return bpref_score

def get_ndcg_mrr2(zipped_list):
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

def get_bpref2(zipped_list):
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


def llm_evaluate(dataset):
    openai.api_key = "Your key"
    openai.api_base = "Your base"
    mode = "score_ranking" # direct_ranking / score_ranking
    
    embed_dir2 = "./logs/2024-01-18_13:47:01/checkpoints"
    last_path = embed_dir2.split("checkpoints")[0]
    history = History.create_eval(file_name="{}/llm_evaluate.jsonl".format(last_path)) # self.history.append({"Profile": self.profile})
    
    data_dir = "./{}/user_200/recbole/user_200".format(dataset)
    with open("{}/user_200_total/user_200_total.item".format(data_dir), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    itemprof_dict = {}
    for i in range(1, len(lines)):
        cur_itemid, cur_itemname, cur_description, _ = lines[i].split("|")
        itemprof_dict[cur_itemid] = (cur_itemname, cur_itemname + ", " + cur_description)
    with open("{}/user_200_total/user_200_total.test.inter".format(data_dir), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    interaction_dict, positivename_dict = {}, {}
    for i in range(1, len(lines)):
        cur_userid, cur_itemid, cur_rating, _ = lines[i].split("|")
        if cur_userid not in interaction_dict.keys():
            interaction_dict[cur_userid] = []
        interaction_dict[cur_userid].append([cur_itemid, cur_rating, itemprof_dict[cur_itemid][0], itemprof_dict[cur_itemid][1]])
        if int(cur_rating) >= 4:
            positivename_dict[cur_userid] = itemprof_dict[cur_itemid][0]
        random.seed(2024)
        random.shuffle(interaction_dict[cur_userid])
    
    train_count = {}
    with open("{}/user_200_total/user_200_total.train.inter".format(data_dir), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        cur_userid, _, _, _ = lines[i].split("|")
        if cur_userid not in train_count.keys():
            train_count[cur_userid] = 0
        train_count[cur_userid] += 1
    
    userprofile_dict = {}
    for file in os.listdir(embed_dir2):
        if file.endswith("final_profile.pkl"):
            cur_userid = file.split("_")[1]
            with open("{}/{}".format(embed_dir2, file), "rb") as f:
                cur_userprofile = pickle.load(f)
            userprofile_dict[cur_userid] = cur_userprofile
    
    ndcg_scores = np.zeros(6)
    mrr_scores = np.zeros(6)
    group_count = np.zeros(6)
    prompt_obj = Prompt()
    
    if mode == "direct_ranking":
        evaluate_prompt = prompt_obj.evaluate_prompt[0]
    elif mode == "score_ranking":
        evaluate_prompt = prompt_obj.score_prompt[0]
    
    count = 0
    users = list(interaction_dict.keys())
    while True:
        if count == len(users):
        # if count == 1:
            break
        key = users[count]
        message = ""
        message += "Final User Profile: {}\n".format(userprofile_dict[key])
        history.append({"Final User Profile": userprofile_dict[key]})
        message += "Items in the testing set:\n"
        itemsinfo = ""
        for i in range(len(interaction_dict[key])):
            temp = "{}\t{}\t{}\n".format(i + 1, interaction_dict[key][i][2], interaction_dict[key][i][3])
            message += temp
            itemsinfo += temp
        history.append({"Item in the testing set": itemsinfo})
        instance_input = [{"role": "user", "content": message}]
        try:
            ret = get_generate(evaluate_prompt, instance_input, 2048)
            print("##### {}".format(ret))
            history.append({"Output": ret})
            
            if train_count[key] <= 20:
                group_id = 1
            elif train_count[key] <= 30:
                group_id = 2
            elif train_count[key] <= 40:
                group_id = 3
            elif train_count[key] <= 50:
                group_id = 4
            elif train_count[key] <= 60:
                group_id = 5
            group_count[0] += 1
            group_count[group_id] += 1
            
            ret_lines = ret.split("\n")
            itemsinfo = ""
            
            if mode == "direct_ranking":
                start_index = ret_lines.index("The ranking of the items:")
                for i in range(start_index + 1, start_index + 21):
                    print(ret_lines[i])
                    itemsinfo += ret_lines[i]
                    cur_index, cur_itemname = ret_lines[i].split(sep=". ", maxsplit=1)
                    if cur_itemname == positivename_dict[key]:
                        cur_ndcg = float(1 / np.log2(i + 1 - start_index))
                        cur_mrr = 1 / (i - start_index)
                        ndcg_scores[0] += cur_ndcg
                        mrr_scores[0] += cur_mrr
                        ndcg_scores[group_id] += cur_ndcg
                        mrr_scores[group_id] += cur_mrr
                        cur_rank = i - start_index
            elif mode == "score_ranking":
                start_index = ret_lines.index("The matching scores of the items:")
                scores = eval(ret_lines[start_index + 1])
                print(scores)
                time.sleep(1000)
                cur_ratinglist = []
                for i in range(len(interaction_dict[key])):
                    cur_ratinglist.append(interaction_dict[key][i][1])
                zipped_list = list(zip(cur_ratinglist, scores))
                for i in range(len(zipped_list)):
                    if zipped_list[i][1] >= 4:
                        cur_ndcg = float(1 / np.log2(i + 2))
                        cur_mrr = 1 / (i + 1)
                        ndcg_scores[0] += cur_ndcg
                        mrr_scores[0] += cur_mrr
                        ndcg_scores[group_id] += cur_ndcg
                        mrr_scores[group_id] += cur_mrr
                        cur_rank = i + 1

            history.append({"The ranking of the items": itemsinfo})
            history.append({"Positive Item": positivename_dict[key], "Ranking": cur_rank})
        except Exception as e:
            print(e)
            time.sleep(1000)
            if isinstance(e, KeyboardInterrupt):
                sys.exit(1)
            continue
        count += 1
    for i in range(6):
        ndcg_scores[i] /= group_count[i]
        mrr_scores[i] /= group_count[i]
    print(ndcg_scores)
    print(mrr_scores)
    ndcg_scores = ndcg_scores.tolist()
    mrr_scores = mrr_scores.tolist()
    history.append({"NDCG@20": ndcg_scores, "MRR@20": mrr_scores})
    sys.exit(1)
 

def verify_description(): # adjust the time to 2 seconds.
    data_path = "./amazon/user_200_3/CDs_and_Vinyl"
    openai.api_key = "Your key"
    openai.api_base = "Your base"
    
    with open("{}/u.item".format(data_path), "r") as f:
        lines = f.read().splitlines()
    for i in trange(10185, len(lines)):
    # for i in range(1):
        _, _, _, cur_description = lines[i].split("|")
        cur_input = [{"role": "user", "content": cur_description}]
        # cur_input = [{"role": "user", "content": "how can i keep from singing, Pie Jesu, Panis Angelicus, O Come All Ye Faithful, Postlude on Adeste fideles"}]
        # print(cur_description, type(cur_input))
        # time.sleep(1000)
        try:
            ret = get_generate(input=cur_input)
        except Exception as e:
            if not isinstance(e, FunctionTimeoutError):
                print("Line {}, {}, {}".format(i + 1, e, type(e)))
    sys.exit(1)

def manual_eval(dataset): # used as interrupt in the middle
    userid_list = []
    path_list = ["./logs/2024-04-09_05:55:43", "./logs/2024-04-07_16:11:14"]
    split_list = [27, 39]
    ndcg_list = np.zeros(6)
    mrr_list = np.zeros(6)
    bpref_list = np.zeros(6)
    count_list = np.zeros(6)
    
    if not os.path.exists("./logs/combined"):
        os.mkdir("./logs/combined")
        os.mkdir("./logs/combined/logs")
        os.mkdir("./logs/combined/checkpoints")
    else:
        os.system("rm -rf ./logs/combined")
        os.mkdir("./logs/combined")
        os.mkdir("./logs/combined/logs")
        os.mkdir("./logs/combined/checkpoints")
    
    assert len(path_list) == len(split_list)
    if "amazon" in dataset:
        data_type = dataset.split("-")[1]
        with open("./amazon/user_200_3/{}/u.user".format(data_type), "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            user_id, _ = lines[i].split("|")
            userid_list.append(user_id)
        usergroup_dict = {}
        for i in range(5):
            with open("./amazon/user_200_3/{}/recbole/user_200/user_200_{}/user_200_{}.user".format(data_type, i, i), "r", encoding="latin-1") as f:
                lines = f.read().splitlines()
            for j in range(1, len(lines)):
                user_id, _ = lines[j].split("|")
                usergroup_dict[user_id] = i + 1
    elif dataset == "ml100k":
        with open("./ml100k/user_200_3/u.user", "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        for i in range(len(lines)):
            user_id, _, _, _, _ = lines[i].split("|")
            userid_list.append(user_id)
        usergroup_dict = {}
        for i in range(5):
            with open("./ml100k/user_200_3/recbole/user_200/user_200_{}/user_200_{}.user".format(i, i), "r", encoding="latin-1") as f:
                lines = f.read().splitlines()
            for j in range(1, len(lines)):
                user_id, _, _, _, _ = lines[j].split("|")
                usergroup_dict[user_id] = i + 1
    
    # print(userid_list)
    # print(usergroup_dict)
    # time.sleep(1000)
    userid_list2, usergroup_dict2 = [], {}
    usergroup_list = list(usergroup_dict.items())
    # print(usergroup_list)
    for i in range(len(usergroup_list)):
        if usergroup_list[i][1] == 5:
            userid_list2.append(usergroup_list[i][0])
            usergroup_dict2[usergroup_list[i][0]] = 5
    # print(userid_list2)
    # print(len(userid_list2))
    # time.sleep(1000)
    userid_list = copy.deepcopy(userid_list2)
    usergroup_dict = copy.deepcopy(usergroup_dict2)
    
    cur_point = 0
    for i in trange(39):
        if i == split_list[cur_point]:
            cur_point += 1
        with open("{}/user/{}.jsonl".format(path_list[cur_point], userid_list[i]), "r") as f:
            lines = f.read().splitlines()
        ndcg, mrr, bpref = json.loads(lines[-3]), json.loads(lines[-2]), json.loads(lines[-1])
        ndcg, mrr, bpref = ndcg["ndcg@20"], mrr["mrr@20"], bpref["bpref@20"]
        ndcg_list[0], ndcg_list[usergroup_dict[userid_list[i]]] = ndcg_list[0] + ndcg, ndcg_list[usergroup_dict[userid_list[i]]] + ndcg
        mrr_list[0], mrr_list[usergroup_dict[userid_list[i]]] = mrr_list[0] + mrr, mrr_list[usergroup_dict[userid_list[i]]] + mrr
        bpref_list[0], bpref_list[usergroup_dict[userid_list[i]]] = bpref_list[0] + bpref, bpref_list[usergroup_dict[userid_list[i]]] + bpref
        count_list[0], count_list[usergroup_dict[userid_list[i]]] = count_list[0] + 1, count_list[usergroup_dict[userid_list[i]]] + 1
        # print(userid_list[i], usergroup_dict[userid_list[i]], ndcg, mrr)
        shutil.copy("{}/user/{}.jsonl".format(path_list[cur_point], userid_list[i]), "./logs/combined/logs/{}.jsonl".format(userid_list[i]))
        for path in os.listdir("{}/checkpoints".format(path_list[cur_point])):
            if path.startswith("user_{}".format(userid_list[i])):
                shutil.copy("{}/checkpoints/{}".format(path_list[cur_point], path), "./logs/combined/checkpoints/{}".format(path))
    
    for i in range(6):
        ndcg_list[i] /= count_list[i]
        mrr_list[i] /= count_list[i]
        bpref_list[i] /= count_list[i]
    with open("{}/final_results.txt".format(path_list[-1]), "w") as f:
        f.write("Final results:\n")
        f.write("ndcg: {}\n".format(ndcg_list))
        f.write("mrr: {}\n".format(mrr_list))
        f.write("bpref: {}\n".format(bpref_list))
    sys.exit(1)

def run_one_thread(userid_list, interaction_train, interaction_test, user_pool, item_pool, updated_object, checkpoints, runtime, statusobj):
    
    i = 0
    error_num = 0
    explode_opt = False
    
    # print(userid_list)
    # print(userid_list.index("A3NHUQ33CFH3VM"))
    # time.sleep(1000)
    with tqdm(total=len(userid_list), colour='magenta', ascii=True) as pbar1:
        while i < len(userid_list):
        # while i < 1:
        # while i < 35:
            
        # while i < 150:
        # while i < 27:
        # while i < 7:
            # print(userid_list[i])
            # time.sleep(1000)
        
        # for i in range(len(userid_list)):
        # for i in [single_index]:
        # for i in range(2):
            df_train, df_test = extract_same_user_interaction(interaction_train, interaction_test, user_id=userid_list[i])
            # statusobj.append({"Status": "Current user id: {}".format(userid_list[i])})
            # print("Initialized! Start training!")
            # statusobj.append({"Status": "Initialized! Start training!"})
            
            # global train_list
            train_list = []
            for index, row in df_train.iterrows():
                train_list.append(row)
            
            # global user_count
            user_count = 0
            # train_list = train_list[0:1]
            
            # while user_count < 5:
            with tqdm(total=len(train_list), colour='yellow', ascii=True) as pbar2:
                # while user_count < 1:
                while user_count < len(train_list):
                # while user_count < 11:
                    if error_num == 10:
                        if profile_max_length < 100:
                            user_pool[train_list[user_count]['user_id']].history.append({"Error info": "Fatal error! Rollback 10 times! Restart this user!"})
                            rollback_all(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], update_terms=updated_object)
                        else:
                            user_pool[train_list[user_count]['user_id']].history.append({"Error info": "Reach maximum I/O length! Stop this user!"})
                            # rollback_all(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], update_terms=updated_object)
                        break
                    
                    # logger.info("{}".format(train_list[user_count]))
                    try:
                        if user_count + batch_size <= len(train_list):
                            batch_indices = list(range(user_count, user_count + batch_size))
                        else:
                            batch_indices = list(range(user_count, len(train_list)))
                        item_list = [item_pool[train_list[index]['item_id']] for index in batch_indices]
                        score_list = [train_list[index]['rating'] for index in batch_indices]
                        # interaction(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], score=train_list[user_count]['rating'], update_terms=updated_object)
                        interaction(user_pool[train_list[user_count]['user_id']], item_list, score=score_list, update_terms=updated_object)
                        # raise FunctionTimeoutError("test")
                    except Exception as e:
                        # print(e)
                        traceback.print_exc()
                        # global connect_obj
                        # connect_obj.close()
                        # del connect_obj
                        # connect_obj = ConnectSSH()
                        user_pool[train_list[user_count]['user_id']].history.append({"Error info": str(e)})
                        statusobj.append({"Error info": str(e)})
                        if isinstance(e, KeyboardInterrupt):
                            sys.exit(1)
                        elif isinstance(e, AttributeError) and explode_opt == True:
                        # elif (isinstance(e, AttributeError) or isinstance(e, openai.error.InvalidRequestError)) and explode_opt == True:
                            error_num2 = 0
                            while True:
                                if error_num2 == 10:
                                    break
                                try:
                                    explode_condense(user_pool[train_list[user_count]['user_id']])
                                except:
                                    error_num2 += 1
                            if error_num2 == 10:
                                error_num += 1
                                # time.sleep(5)
                                rollback(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], update_terms=updated_object)
                                continue
                        else:
                            error_num += 1
                            # time.sleep(5)
                            rollback(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], update_terms=updated_object)
                            continue
                    # interaction(user_pool[train_list[user_count]['user_id']], item_pool[train_list[user_count]['item_id']], score=train_list[user_count]['rating'], update_terms=updated_object)
                    # user_count += 1
                    user_count += batch_size
                    error_num = 0
                    if user_count >= len(train_list):
                        break
                    if user_count in checkpoints:
                        if "user" in updated_object:
                            with open("./logs/{}/checkpoints/user_{}_{}_profile.pkl".format(runtime, train_list[user_count]['user_id'], user_count), "wb") as f:
                                pickle.dump(user_pool[train_list[user_count]['user_id']].profile, f)
                            with open("./logs/{}/checkpoints/user_{}_{}_singleembed.pkl".format(runtime, train_list[user_count]['user_id'], user_count), "wb") as f:
                                pickle.dump(user_pool[train_list[user_count]['user_id']].singleembed, f)
                            user_pool[train_list[user_count]['user_id']].totalembed = get_embedding_with_retry(
                                ollama_client,
                                list2sentence(user_pool[train_list[user_count]['user_id']].profile),
                                max_retries=5,
                            )
                            # while True:
                            #     try:
                            #         user_pool[train_list[user_count]['user_id']].totalembed = get_embedding(ollama_client, list2sentence(user_pool[train_list[user_count]['user_id']].profile))
                            #         # user_pool[train_list[user_count]['user_id']].totalembed2 = get_embedding2(list2sentence(user_pool[train_list[user_count]['user_id']].profile))
                            #         break
                            #     except:
                            #         pass
                            with open("./logs/{}/checkpoints/user_{}_{}_totalembed.pkl".format(runtime, train_list[user_count]['user_id'], user_count), "wb") as f:
                                pickle.dump(user_pool[train_list[user_count]['user_id']].totalembed, f)
                            # with open("./logs/{}/checkpoints/user_{}_{}_totalembed2.pkl".format(runtime, train_list[user_count]['user_id'], user_count), "wb") as f:
                            #     pickle.dump(user_pool[train_list[user_count]['user_id']].totalembed2, f)
                    # if user_count % 50 == 0:
                    #     statusobj.append({"Status": str(pbar2)})
                    pbar2.update(batch_size)
                
            if error_num == 10:
                error_num = 0
                if profile_max_length < 100:
                    continue
            
            if "user" in updated_object:
                with open("./logs/{}/checkpoints/user_{}_final_profile.pkl".format(runtime, userid_list[i]), "wb") as f:
                    pickle.dump(user_pool[userid_list[i]].profile, f)
                with open("./logs/{}/checkpoints/user_{}_final_singleembed.pkl".format(runtime, userid_list[i]), "wb") as f:
                    pickle.dump(user_pool[userid_list[i]].singleembed, f)
                user_pool[userid_list[i]].totalembed = get_embedding_with_retry(
                    ollama_client,
                    list2sentence(user_pool[userid_list[i]].profile),
                    max_retries=5,
                )
                # while True:
                #     try:
                #         user_pool[userid_list[i]].totalembed = get_embedding(ollama_client, list2sentence(user_pool[userid_list[i]].profile))
                #         # user_pool[userid_list[i]].totalembed2 = get_embedding2(list2sentence(user_pool[userid_list[i]].profile))
                #         break
                #     except:
                #         pass
                with open("./logs/{}/checkpoints/user_{}_final_totalembed.pkl".format(runtime, userid_list[i]), "wb") as f:
                    pickle.dump(user_pool[userid_list[i]].totalembed, f)
                # with open("./logs/{}/checkpoints/user_{}_final_totalembed2.pkl".format(runtime, userid_list[i]), "wb") as f:
                #     pickle.dump(user_pool[userid_list[i]].totalembed2, f)

            # 清理当前 user 的大字段
            if hasattr(user_pool[userid_list[i]], "singleembed"):
                user_pool[userid_list[i]].singleembed = None
            if hasattr(user_pool[userid_list[i]], "totalembed"):
                user_pool[userid_list[i]].totalembed = None
            
            print("Learning user {} profile complete.".format(userid_list[i]))
            del df_train, df_test, train_list
            gc.collect()

            # statusobj.append({"Status": "Learning user {} profile complete.".format(userid_list[i])})
            if i % 5 == 0:
                statusobj.append({"Status": str(pbar1)})
            i += 1
            pbar1.update()
    

def separate_users(userid_list, dataset):
    separate_num = 5
    userid_list2 = copy.deepcopy(userid_list)
    random.seed(2024)
    random.shuffle(userid_list2)
    group_len = len(userid_list2) // separate_num
    grouped_list = []
    count = 0
    for i in range(separate_num):
        if i < separate_num - 1:
            grouped_list.append(userid_list2[group_len * i: group_len * (i + 1)])
        else:
            grouped_list.append(userid_list2[group_len * i: len(userid_list2)])
        count += len(grouped_list[-1])
    assert count == len(userid_list2)
    with open("./{}_full5.pkl".format(dataset), "wb") as f:
        pickle.dump(grouped_list, f)
    sys.exit(1)

def run(llm, dataset_tuple):
    runtime=getTime()
    updated_object=["user"]
    if batch_size == 5 or batch_size == 1:
        # checkpoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        checkpoints = [20, 40, 60, 80]
    elif batch_size == 3:
        # checkpoints = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78]
        checkpoints = [12, 24, 36, 48, 60, 72]
    elif batch_size == 7:
        checkpoints = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
    elif batch_size == 4:
        checkpoints = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
    elif batch_size == 10:
        checkpoints = [10, 20, 30, 40, 50, 60, 70]
    thread_num = 20
    # checkpoints = []

    # global item_pool
    user_pool, item_pool, interaction_train, interaction_test = load_meta_data(updated_object, runtime, llm, dataset_tuple)
    
    # print(len(item_pool.keys()))
    # print(item_pool['B000A7Q1P6'])
    # sys.exit(1)
    
    userid_list = []
    for index, row in interaction_test.iterrows():
        if row["user_id"] not in userid_list:
            userid_list.append(row["user_id"])
    print("total user number: {}".format(len(userid_list)))
    
    # with open("./{}_full5.pkl".format(dataset_tuple[1]), "rb") as f:
    #     userid_list = pickle.load(f)[2]
    
    # print(len(userid_list))
    # time.sleep(1000)
    # with open("./condense_length_results/amazon_1/status/fail_userid.pkl", "rb") as f:
    #     userid_list = pickle.load(f)
    # if dataset == 'ml-25m':
    #     for i in range(len(userid_list)):
    #         userid_list[i] = float(userid_list[i])
    # userid_list = [20913.0]
    
    # for group 5 only.
    # userid_list2 = []
    # for i in range(len(userid_list)):
    #     temp_train, _ = extract_same_user_interaction(interaction_train, interaction_test, user_id=userid_list[i])
    #     # if 65 < len(temp_train) <= 80:
    #     # if len(temp_train) <= 65:
    #     if True:
    #         userid_list2.append(userid_list[i])
    # userid_list = copy.deepcopy(userid_list2)
    # separate_users(userid_list, dataset_tuple[1])
    
    indexs = [len(userid_list) * (i + 1) // thread_num for i in range(thread_num)]
    indexs = [0] + indexs
    group_userid_list = [userid_list[indexs[i]:indexs[i + 1]] for i in range(thread_num)]
    for i in range(thread_num):
        print(len(group_userid_list[i]), end=" ")
    print("")
    # time.sleep(1000)
    
    # completed = [33, 36, 42, 28, 37, 33, 35, 38, 34, 41, 41, 37, 28, 30, 29, 42, 34, 39, 31, 34]
    # for i in range(thread_num):
    #     group_userid_list[i] = group_userid_list[i][completed[i]:]
    
    # for i in range(thread_num):
    #     if i == 9:
    #         group_userid_list[i] = group_userid_list[i][19:]
    #     else:
    #         group_userid_list[i] = []
    
    if not os.path.exists("./logs/{}/status".format(runtime)):
        os.mkdir("./logs/{}/status".format(runtime))
    
    # print(interaction_train)
    # print(userid_list)
    # print(len(userid_list))
    # time.sleep(10000)
    
    # ndcg_scores = np.zeros(6)
    # mrr_scores = np.zeros(6)
    # bpref_scores = np.zeros(6)
    # group_count = np.zeros(6)
    
    # single_userid = 14
    # single_index = userid_list.index(single_userid)
    history_list = [History.create_eval(file_name="./logs/{}/status/thread_{}.jsonl".format(runtime, i)) for i in range(thread_num)]
    thread_list = [threading.Thread(target=run_one_thread,
                                    args=(group_userid_list[i], interaction_train, interaction_test, user_pool, item_pool,
                                          updated_object, checkpoints, runtime, history_list[i]))
                   for i in range(thread_num)]
    
    for i in range(thread_num):
        thread_list[i].start()
    for i in range(thread_num):
        thread_list[i].join()
    
def group_eval(userinter_dict, file, eval_length):
    with open(file, "r", encoding='latin-1') as f:
        userid_list = f.read().splitlines()
    random.seed(2024)
    inter_sortdict, return_dict = {}, {}
    for i in range(1, len(userid_list)):
        userid = userid_list[i].split("|")[0]
        interaction = userinter_dict[userid]
        random.shuffle(interaction)
        interaction = sorted(interaction, key=lambda x:x[2], reverse=True)
        inter_sortdict[userid] = interaction
    
    for i in range(len(eval_length)):
        total_mrr, total_ndcg = 0.0, 0.0
        total_bpref = 0.0
        for key in inter_sortdict.keys():
            interaction = inter_sortdict[key][:eval_length[i]]
            cur_ndcg, cur_mrr = get_ndcg_mrr2(interaction)
            cur_bpref = get_bpref2(interaction)
            total_ndcg += cur_ndcg
            total_mrr += cur_mrr
            total_bpref += cur_bpref
        total_ndcg /= (len(userid_list) - 1)
        total_mrr /= (len(userid_list) - 1)
        total_bpref /= (len(userid_list) - 1)
        return_dict[eval_length[i]] = {'ndcg@{}'.format(eval_length[i]):total_ndcg, 'mrr@{}'.format(eval_length[i]):total_mrr, 'bpref@{}'.format(eval_length[i]):total_bpref}
    return return_dict

def manual_evaluate(dataset):
    userinter_dict = {}
    userinter_dict2 = {}
    eval_length = [1, 5, 10, 20]
    if dataset == "ml100k":
        path = "./ml100k/user_200_3/recbole/user_200"
    elif "amazon" in dataset:
        subset = dataset.split("-")[1]
        path = "./amazon/user_200_3/{}/recbole/user_200".format(subset)
        
    userembed_dict, itemembed_dict = {}, {}
    
    with open("{}/user_200_total/user_200_total.useremb".format(path), "r") as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        cur_userid, cur_embed = lines[i].split("|")
        cur_embed = cur_embed.split(", ")[:-1]
        cur_embed = [float(x) for x in cur_embed]
        userembed_dict[cur_userid] = cur_embed
    with open("{}/user_200_total/user_200_total.itememb".format(path), "r") as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        cur_itemid, cur_embed = lines[i].split("|")
        cur_embed = cur_embed.split(", ")[:-1]
        cur_embed = [float(x) for x in cur_embed]
        itemembed_dict[cur_itemid] = cur_embed
    
    with open("{}/user_200_total/user_200_total.test.inter".format(path), "r", encoding='latin-1') as f:
        lines = f.read().splitlines()
    for i in range(1, len(lines)):
        user_id, item_id, rating, timestamp = lines[i].split("|")
        if user_id not in userinter_dict.keys():
            userinter_dict[user_id] = {}
        if int(rating) >= 4:
            is_pos = 1
        elif 0 < int(rating) <= 3:
            is_pos = -1
        else:
            is_pos = 0
        cur_cosscore = 1 - spatial.distance.cosine(userembed_dict[user_id], itemembed_dict[item_id])
        userinter_dict[user_id][item_id] = [item_id, is_pos, cur_cosscore]
    
    for key in userinter_dict.keys():
        cur_inter = list(userinter_dict[key].items())
        userinter_dict2[key] = []
        for i in range(len(cur_inter)):
            # print(cur_inter[i])
            # time.sleep(1000)
            userinter_dict2[key].append(cur_inter[i][1])
    # print(userinter_dict2)
    # time.sleep(1000)
    
    result_list = []    
    # total_test_result = group_eval(userinter_dict2, "{}/user_200_total/user_200_total.userprof".format(path))
    total_test_result = group_eval(userinter_dict2, "{}/user_200_total/user_200_total.user".format(path), eval_length)
    print("total test result: {}".format(total_test_result))
    result_list.append(total_test_result)
    
    # only evaluate one group, please comment out the code below
    for i in range(5):
        # group_test_result = group_eval(userinter_dict2, "{}/user_200_{}/user_200_{}.userprof".format(path, i, i))
        group_test_result = group_eval(userinter_dict2, "{}/user_200_{}/user_200_{}.user".format(path, i, i), eval_length)
        print("test result of group {}: {}".format(i, group_test_result))
        result_list.append(group_test_result)
    sys.exit(1)
    return result_list

def examine_status(folder, called=False):
    path_parts = folder.split("/")
    dataset = path_parts[-2].split("_", maxsplit=1)[0]
    # dataset = folder.split("_", maxsplit=1)[0]
    fail_userid, runned_userid = [], []
    with tqdm(total=20) as pbar:
        for i in range(20):
            with open("{}/status/thread_{}.jsonl".format(folder, i), "r") as f:
                lines = f.read().splitlines()
            for j in range(len(lines)):
                if "profile complete" in lines[j]:
                    flag = True # True代表是fail的。
                    for k in range(j - 1, j - 11, -1):
                        if "Error info" not in lines[k]:
                            flag = False
                            break
                    temp = lines[j].split(" profile complete")[0]
                    temp = temp.split("Learning user ")[1]
                    if dataset == "ml-25m":
                        temp = str(int(float(temp)))
                    if flag:
                        fail_userid.append(temp)
                        # print(i)
                    runned_userid.append(temp)
            pbar.update()
    if dataset == "amazon":
        path = "./amazon/user_5k/CDs_and_Vinyl/recbole/user_5k/user_5k_4"
    elif dataset == "ml-25m":
        path = "./ml-25m/user_5k/recbole/user_5k/user_5k_4"
    with open("{}/user_5k_4.user".format(path), "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    userid_list = []
    for i in range(1, len(lines)):
        userid_list.append(lines[i].split("|")[0])
    # print(len(runned_userid), len(set(runned_userid)))
    # print(len(userid_list), len(set(userid_list)))

    failed_set = set(userid_list) - set(runned_userid)
    # print(len(failed_set))
    # print(len(fail_userid))
    failed_set = failed_set.union(set(fail_userid))
    fail_userid = list(failed_set)
    
    # print(len(fail_userid), len(runned_userid))
    
    if not called:
        with open("./{}/status/fail_userid.pkl".format(folder), "wb") as f:
            pickle.dump(fail_userid, f)
    return fail_userid, runned_userid

def merge_folders():
    def copy_files(folder, userid_list):
        # userid_list = set(userid_list)
        # path_parts = folder.split("/")
        
        user_folder = "{}/user".format(folder)
        checkpoints = "{}/checkpoints".format(folder)
        listdir_user, listdir_check = os.listdir(user_folder), os.listdir(checkpoints)
        with tqdm(total=len(listdir_user)) as pbar:
            for file in listdir_user:
                cur_userid = file.split(".")[0]
                if cur_userid in userid_list:
                    shutil.copy("{}/{}".format(user_folder, file), "{}/user/{}".format(merged_folder, file))
                pbar.update()
        with tqdm(total=len(listdir_check)) as pbar:
            for file in listdir_check:
                cur_userid = file.split("_")[1]
                try:
                    cur_userid = str(int(float(cur_userid)))
                except:
                    cur_userid = cur_userid
                if cur_userid in userid_list:
                    shutil.copy("{}/{}".format(checkpoints, file), "{}/checkpoints/{}".format(merged_folder, file))
                pbar.update()
    
    def copy_status(folder1, folder2):
        with tqdm(total=20) as pbar:
            for i in range(20):
                with open("{}/status/thread_{}.jsonl".format(folder1, i), "r") as f:
                    lines1 = f.read().splitlines()
                pos = len(lines1) - 1
                for j in range(len(lines1) - 1, -1, -1):
                    if "profile complete" in lines1[j]:
                        pos = j + 1
                        break
                lines1 = lines1[:pos + 1]
                with open("{}/status/thread_{}.jsonl".format(folder2, i), "r") as f:
                    lines2 = f.read().splitlines()
                lines1 += lines2
                with open("{}/status/thread_{}.jsonl".format(merged_folder, i), "w") as f:
                    for j in range(len(lines1)):
                        f.write(lines1[j])
                        f.write("\n")
                pbar.update()
    
    folder1 = "./ml-25m_results/raw_results/ml-25m_5_32_newprompt/merged_13"
    folder2 = "./ml-25m_results/raw_results/ml-25m_5_32_newprompt/merged_2"
    merged_folder = "{}/merged".format(folder1.rsplit("/", maxsplit=1)[0])
    print(merged_folder)
    # merged_folder = "./history_length_results/merged"
    if not os.path.exists(merged_folder):
        os.mkdir(merged_folder)
        for folder in ["checkpoints", "item", "user", "status"]:
            os.mkdir("{}/{}".format(merged_folder, folder))
    copy_status(folder1, folder2)
    fail_userid1, userid_list1 = examine_status(folder1, True)
    fail_userid2, userid_list2 = examine_status(folder2, True)
    have_userid1 = set(userid_list1) - set(fail_userid1)
    have_userid2 = set(userid_list2) - set(fail_userid2)
    # print(len(have_userid1), len(have_userid2))
    # time.sleep(1000)
    
    copy_files(folder1, userid_list1)
    copy_files(folder2, userid_list2)
    
def check_condense():
    pattern = "***[User condense]***"
    folder = "./condense_length_results/amazon_5_40/user"
    count = 0
    files = os.listdir(folder)
    for file in files:
        with open("{}/{}".format(folder, file), "r") as f:
            lines = f.read().splitlines()
        for i in range(-10, 0):
            if pattern in lines[i]:
                count += 1
                break
    print("{}/{}".format(count, len(files)))
    sys.exit(1)
    
    """
    ml-25m_5_16: last 20 - 953/990, last 10 - 910/990
    ml-25m_5_32: last 20 - 708/990, last 10 - 484/990
    ml-25m_5_40: last 20 - 446/900, last 10 - 274/990
    ml-25m_5_48: last 20 - 609/990, last 10 - 439/990
    ml-25m_5_64: last 20 - 76/990, last 10 - 38/990
    
    amazon_5_16: last 20 - 987/996, last 10 - 955/996
    amazon_5_32: last 20 - 764/996, last 10 - 514/996
    amazon_5_40: last 20 - 544/996, last 10 - 357/996
    amazon_5_48: last 20 - 608/996, last 10 - 363/996
    
    amazon_3_16: last 20 - 993/996, last 10 - 981/996
    amazon_3_32: last 20 - 915/996, last 10 - 675/996
    amazon_3_48: last 20 - 720/996, last 10 - 434/996
    amazon_3_64: last 20 - 392/996, last 10 - 218/996
    """

def count_complete_usernum():
    path = "./logs/2024-09-21_18:11:29/checkpoints"
    users = {}
    for file in os.listdir(path):
        if file.endswith("final_profile.pkl"):
            cur_userid = file.split("_")[1]
            users[cur_userid] = True
    print(len(users))
    sys.exit(1)

def check_groups_pkl():
    with open("./amazon-CDs_and_Vinyl.pkl", "rb") as f:
        groups = pickle.load(f)
    for i in range(len(groups)):
        print(len(groups[i]))
    sys.exit(1)

if __name__== "__main__":
    """
    ml-25m: 1-0, 3-0, 5-1, 10-0, 20-139
    amazon: 1-0, 3-68, 5-620, 10-565
    """
    # check_groups_pkl()
    # count_complete_usernum()
    # check_condense()
    # examine_status("ml-25m_5")
    # examine_status("amazon_1_p2")
    # merge_folders()
    # sys.exit(1)
    
    # USER_ID=21
    # Support: LLaMA-7B, LLaMA-13B, LLaMA2-7B, LLaMA2-13B, ChatGPT
    # llm='LLaMA-13B'
    # manual_evaluate("amazon-CDs_and_Vinyl")
    
    # with open("./logs/2024-05-13_17:20:15/checkpoints/user_2_final_totalembed2.pkl", "rb") as f:
    #     embed = pickle.load(f)
    # print(len(embed), embed[0])
    # sys.exit(1)
    
    # connect_ssh()
    # sys.exit(1)
    
    llm = "llama3"
    # dataset = "ml100k"
    # dataset = "amazon-CDs_and_Vinyl"
    # dataset = "amazon-Office_Products"
    
    # profile_max_length = int(sys.argv[1])
    length_list = [32]
    # profile_max_length = 32
    # test()
    # llm_evaluate(dataset)
    # condense_desciption()
    # verify_description()
    # manual_eval(dataset)
    # user_count = 0
    # item_pool, train_list = None, None
    
    ollama_client = None
    # ollama_client = ollama.Client(host="http://127.0.0.1:11435")
    batchsize_list = [5]
    # batch_size = 7
    
    # connect_obj = ConnectSSH()
    
    # port start with 11434
    # if llm is 'llama3': Please run "OLLAMA_NUM_PARALLEL=20 OLLAMA_HOST=127.0.0.1:11441 CUDA_VISIBLE_DEVICES=7 nohup ./ollama-linux-amd64 serve" in the terminal first.
    # if login as other users: OLLAMA_MODELS=~/.ollama/models OLLAMA_NUM_PARALLEL=20 OLLAMA_HOST=127.0.0.1:11441 CUDA_VISIBLE_DEVICES=7 nohup ./ollama-linux-amd64 serve
    # for llm in ['llama3']:
    for batch_size in batchsize_list:
        for profile_max_length in length_list:
            for index, dataset in enumerate(['ml-25m']): # 'ml-25m', 
            # for index, dataset in enumerate(['amazon-CDs_and_Vinyl']):
            # if dataset == 'amazon-CDs_and_Vinyl':
                # if dataset in ['ml-25m', 'amazon-CDs_and_Vinyl']:
                # if dataset in ['ml-25m']:
                # if dataset in ['amazon-CDs_and_Vinyl']:
                    # if not (dataset == 'ml100k' and llm == 'ChatGPT'):
                if dataset == 'amazon-CDs_and_Vinyl':
                    with open("./log/XSimGCL/XSimGCL-user_5k_total_amazon_5_32_full-May-14-2026_14-59-29-5562bd/cosmatrix.pkl", "rb") as f:
                        temp = pickle.load(f)
                        cosmatrix, user_mapping, item_mapping = temp['cosmatrix'], temp['user_mapping'], temp['item_mapping']
                elif dataset == 'ml-25m':
                    with open("./ml-25m/cosmatrix.pkl", "rb") as f:
                        temp = pickle.load(f)
                        cosmatrix, user_mapping, item_mapping = temp['cosmatrix'], temp['user_mapping'], temp['item_mapping']

                if dataset == 'amazon-CDs_and_Vinyl':
                    dataset_idx = 1
                elif dataset == 'ml-25m':
                    dataset_idx = 0

                run(llm, (dataset_idx, dataset))
