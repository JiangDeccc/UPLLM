# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import time
import pickle
import random
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
# import ollama
from tqdm import tqdm

from prompt import Prompt

from openai import OpenAI

vllm_client = OpenAI(
    api_key=os.getenv("UPLLM_API_KEY", "EMPTY"),
    base_url="http://localhost:8002/v1"
)
MODEL_NAME = "./models/Meta-Llama-3.1-8B-Instruct"

# 初始化embedding client(和vllm_client并列放在__main__或全局)
embed_client = OpenAI(
    api_key=os.getenv("UPLLM_API_KEY", "EMPTY"),
    base_url="http://localhost:8001/v1"
)
EMBED_MODEL_NAME = "./models/Meta-Llama-3.1-8B-Instruct"

# =========================
# LLM IO Log 配置
# =========================

LLM_LOG_DIR = "./llm_io_logs"
ensure_log_lock = threading.Lock()
llm_log_lock = threading.Lock()

MODEL_LIST = ["BPR"]


def get_llm_log_file():
    """
    每次运行按日期时间生成一个 jsonl 文件。
    jsonl 每一行是一条 LLM 调用记录。
    """
    os.makedirs(LLM_LOG_DIR, exist_ok=True)
    date_str = time.strftime("%Y-%m-%d %H")
    return os.path.join(LLM_LOG_DIR, f"llm_io_{date_str}.jsonl")


def save_llm_io_log(
    stage: str,
    messages,
    response_text: str = None,
    error: str = None,
    extra: dict = None,
):
    """
    线程安全地保存 LLM 输入输出。
    """
    log_record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "model": MODEL_NAME,
        "messages": messages,
        "response": response_text,
        "error": error,
    }

    if extra is not None:
        log_record["extra"] = extra

    log_file = get_llm_log_file()

    with llm_log_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")


# =========================
# 1. 基础工具函数
# =========================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



def get_generate_llama(
    client,
    instruction=None,
    input=[{"role": "user", "content": "hello?"}],
    max_tokens=None,
    stage: str = "unknown",
    extra: dict = None,
):
    if instruction is None:
        finalinput = input
    else:
        finalinput = instruction + input

    kwargs = dict(
        model=MODEL_NAME,
        messages=finalinput,
        temperature=0.6,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = vllm_client.chat.completions.create(**kwargs)
        ret = response.choices[0].message.content

        save_llm_io_log(
            stage=stage,
            messages=finalinput,
            response_text=ret,
            error=None,
            extra=extra,
        )

        return ret

    except Exception as e:
        save_llm_io_log(
            stage=stage,
            messages=finalinput,
            response_text=None,
            error=repr(e),
            extra=extra,
        )
        raise

# def get_generate_llama(client, instruction=None, input=[{"role": "user", "content": "hello?"}], max_tokens=None):
#     if instruction is None:
#         finalinput = input
#     else:
#         finalinput = instruction + input

#     kwargs = dict(
#         model=MODEL_NAME,
#         messages=finalinput,
#         temperature=0.6,
#     )
#     if max_tokens is not None:
#         kwargs["max_tokens"] = max_tokens

#     response = vllm_client.chat.completions.create(**kwargs)
#     return response.choices[0].message.content


def get_embedding(client, sentence):
    response = embed_client.embeddings.create(
        model=EMBED_MODEL_NAME,
        input=sentence
    )
    return response.data[0].embedding


def list2sentence(profile_list):
    """
    把 profile list 拼成一句话。
    兼容 tuple / Ellipsis。
    """
    if isinstance(profile_list, tuple):
        new_profile = []
        for part in profile_list:
            new_profile += list(part)
        profile_list = new_profile

    profile_list = list(profile_list)

    for i in range(len(profile_list) - 1, -1, -1):
        if profile_list[i] is Ellipsis:
            profile_list.pop(i)

    sentence = ""
    for i, profile in enumerate(profile_list):
        sentence += str(profile)
        if i != len(profile_list) - 1:
            sentence += ", "
        else:
            sentence += "."
    return sentence


def extract_potential_interest(string: str) -> str:
    string = string.replace("\n", "")
    string = re.sub(r"\s+", " ", string)

    patterns = [
        r"(?<=Potential interest:).*",
        r"(?<=Potential interest: ).*",
        r"(?<=Potential Interest:).*",
        r"(?<=Potential Interest: ).*",
    ]

    result = None
    for pattern in patterns:
        match = re.search(pattern, string)
        if match:
            result = match.group(0).strip()

    if result is None:
        print(f"[Error] Failed to extract potential interest from LLM output: {string}")
        raise ValueError("Cannot decode potential interest from LLM output.")

    return result


def extract_new_profile(string: str) -> List[str]:
    string = string.replace("\n", " ")
    string = re.sub(r"\s+", " ", string)

    pattern_list = [
        r"(?<=New user profile:)\s*\[.*?\]",
        r"(?<=New user profile: )\s*\[.*?\]",
        r"(?<=New User Profile:)\s*\[.*?\]",
        r"(?<=New User Profile: )\s*\[.*?\]",
    ]

    raw_list_str = None
    for pattern in pattern_list:
        match_res = re.search(pattern, string, re.IGNORECASE)
        if match_res:
            raw_list_str = match_res.group(0).strip()
            break

    if raw_list_str is None:
        raise ValueError("Cannot decode new user profile from LLM output.")

    try:
        return ast.literal_eval(raw_list_str)
    except Exception:
        pass

    inner = raw_list_str.strip()
    if inner.startswith("["):
        inner = inner[1:]
    if inner.endswith("]"):
        inner = inner[:-1]

    items = _split_profile_items(inner)
    result = [_clean_item(item) for item in items if _clean_item(item)]

    if not result:
        raise ValueError(f"Extracted empty profile list from: {raw_list_str}")

    return result


def _split_profile_items(inner: str) -> List[str]:
    items = []
    current = []
    in_quote = None

    for char in inner:
        if in_quote is None:
            if char in ('"', "'"):
                in_quote = char
                current.append(char)
            elif char == ',':
                items.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        else:
            if char == in_quote:
                in_quote = None
                current.append(char)
            else:
                current.append(char)

    last = "".join(current).strip()
    if last:
        items.append(last)

    return items


def _clean_item(item: str) -> str:
    item = item.strip()
    if not item:
        return ""

    if item[0] in ('"', "'"):
        item = item[1:]
    if item and item[-1] in ('"', "'"):
        item = item[:-1]

    return item.strip()


# =========================
# 2. 路径配置
# =========================

def get_dataset_config(dataset: str):
    if dataset == "ml-25m":
        return {
            "dataset_short": "ml-25m",
            "origin_path": "./ml-25m/user_5k",
            "sortdict_path": "./sortdict",
            "base_folder": "./ml-25m_results/ml-25m_cfscore_3level",
            "profile_userid_float": True,
            "potential_cut_len": 30,
        }

    elif dataset == "amazon-CDs_and_Vinyl":
        return {
            "dataset_short": "amazon",
            "origin_path": "./amazon/user_5k/CDs_and_Vinyl",
            "sortdict_path": "./sortdict",
            # "base_folder": "./logs/2026-05-12_19:35:51",
            "base_folder": "./logs/2026-05-13_17:01:48",
            # "base_folder": "./logs/2026-05-16_15:05:09",
            "profile_userid_float": False,
            "potential_cut_len": 20,
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# =========================
# 3. 找 potential items (CPU/IO 轻量,保持单线程)
# =========================

def load_item_profiles(origin_path: str) -> Dict[str, str]:
    item_dict = {}

    item_file = os.path.join(origin_path, "u.item")
    with open(item_file, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        item_id, item_profile = line.split("|", maxsplit=1)
        item_dict[item_id] = item_profile

    return item_dict


def generate_potential_items_single(
    dataset: str,
    model_list=None,
    cut_size: int = 30,
):
    """
    这一步主要是文件 IO + 排序,瓶颈不在 LLM,保持单线程即可。
    """
    cfg = get_dataset_config(dataset)
    dataset_short = cfg["dataset_short"]
    sortdict_path = cfg["sortdict_path"]
    origin_path = cfg["origin_path"]

    if model_list is None:
        model_list = MODEL_LIST
        # model_list = ["LightGCL", "LightGCN", "SimpleX", "XSimGCL"]

    item_dict = load_item_profiles(origin_path)
    potential_dict = {}

    for model in model_list:
        sortdict_file = os.path.join(sortdict_path, f"{dataset_short}_{model}.pkl")

        with open(sortdict_file, "rb") as f:
            cur_sortdict = pickle.load(f)

        with tqdm(total=len(cur_sortdict), desc=f"Loading {model}") as pbar:
            for user_id, ranked_items in cur_sortdict.items():
                if user_id not in potential_dict:
                    potential_dict[user_id] = {}

                for cur_tuple in ranked_items:
                    item_id = cur_tuple[0]
                    item_info = cur_tuple[1][0]
                    item_score = cur_tuple[1][1]

                    if item_id not in potential_dict[user_id]:
                        potential_dict[user_id][item_id] = [item_info, 0.0]

                    potential_dict[user_id][item_id][-1] += item_score / len(model_list)

                pbar.update()

    for user_id in potential_dict:
        potential_dict[user_id] = sorted(
            potential_dict[user_id].items(),
            key=lambda x: x[1][-1],
            reverse=True
        )

    output_file = os.path.join(
        sortdict_path,
        f"{dataset_short}_potential_{cut_size}_{str(model_list)}.txt"
    )

    with open(output_file, "w") as f:
        with tqdm(total=len(potential_dict), desc=f"Saving top-{cut_size} potential items") as pbar:
            for user_id, items in potential_dict.items():
                top_items = items[:cut_size]

                for item_id, value in top_items:
                    item_info = value[0]
                    avg_score = value[1]
                    item_profile = item_dict[item_id]

                    f.write(
                        f"{user_id}|{item_id}|{item_info}|{avg_score}|{item_profile}\n"
                    )

                pbar.update()

    print(f"[Done] Potential items saved to: {output_file}")
    return output_file


# =========================
# 4. LLM 总结 potential interest (单线程 + 多线程)
# =========================

def load_potential_items_for_prompt(potential_file: str) -> Dict[str, List[str]]:
    generate_dict = {}

    with open(potential_file, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        cur_list = line.split("|")
        user_id = cur_list[0]

        if len(cur_list) >= 7:
            title = cur_list[-3]
            category = cur_list[-2]
            description = cur_list[-1]
            cur_itemprof = f"Title: {title} | Category: {category} | Description: {description}\n"
        else:
            cur_itemprof = "Item Profile: " + "|".join(cur_list[4:]) + "\n"

        if user_id not in generate_dict:
            generate_dict[user_id] = []

        generate_dict[user_id].append(cur_itemprof)

    return generate_dict


# ---------- 原单线程版本(保留) ----------

def generate_user_potential_interest_single(
    dataset: str,
    potential_file: str,
    port: int = 11442,
    max_retry: int = 20,
):
    cfg = get_dataset_config(dataset)
    sortdict_path = cfg["sortdict_path"]
    cut_len = cfg["potential_cut_len"]

    prompt_obj = Prompt()
    potential_prefix = prompt_obj.summarize_potential[0]
    # potential_prefix, potential_postfix = potential_prefix[:-1], potential_prefix[-1:]

    generate_dict = load_potential_items_for_prompt(potential_file)

    client = None

    output_file = os.path.join(
        sortdict_path,
        f"{dataset}_userpotential_{cut_len}.txt"
    )

    start_list = [
        "Based on", "The user", "This user", "It seems", "Given", "It appears",
        "**Based on", "**The user", "**This user", "**It seems", "**Given", "**It appears"
    ]

    error_users = []

    with open(output_file, "w") as f:
        with tqdm(total=len(generate_dict), desc="Generating potential interest") as pbar:
            for user_id, item_prompts in generate_dict.items():
                success = False
                for retry in range(max_retry):
                    input_msg = ""
                    for i, item_prompt in enumerate(item_prompts):
                        input_msg += f"Potential Interested Item {i + 1}. {item_prompt}\n"

                    # input_dict = [{"role": "user", "content": input_msg}]
                    # input_dict += potential_postfix
                    user_input = potential_prefix[1]['content'].format(items=input_msg)
                    input_dict = potential_prefix[0] + user_input
                    print(input_dict)

                    try:
                        ret = get_generate_llama(client, potential_prefix, input_dict)
                        potential_interest = extract_potential_interest(ret)

                        flag = False
                        for start in start_list:
                            if potential_interest.startswith(start) or potential_interest.startswith(start, 1):
                                flag = True
                                break

                        if not flag:
                            raise ValueError(f"Bad potential interest format: {potential_interest}")

                        f.write(f"{user_id}|{potential_interest}\n")
                        success = True
                        break

                    except Exception:
                        traceback.print_exc()
                        time.sleep(1)

                if not success:
                    error_users.append(user_id)

                pbar.update()

    error_file = os.path.join(
        sortdict_path,
        f"{dataset}_potential_interest_error_users.pkl"
    )
    with open(error_file, "wb") as f:
        pickle.dump(error_users, f)

    print(f"[Done] User potential interests saved to: {output_file}")
    print(f"[Info] Error users saved to: {error_file}, count = {len(error_users)}")

    return output_file


# ---------- 新多线程版本 ----------

def _task_generate_potential_interest(
    user_id: str,
    item_prompts: List[str],
    potential_prefix,
    start_list: List[str],
    max_retry: int,
) -> Tuple[str, str, bool]:
    """
    单个用户任务:返回 (user_id, potential_interest_or_None, success)
    在 worker 线程内执行,共享全局 vllm_client(线程安全)。
    """
    input_msg = ""
    for i, item_prompt in enumerate(item_prompts):
        input_msg += f"Potential Item {i + 1}. {item_prompt}\n"

    for retry in range(max_retry):
        user_input = potential_prefix[1]['content'].format(items=input_msg)
        input_dict = [potential_prefix[0]] + [{"role": "user", "content": user_input}]

        try:
            # ret = get_generate_llama(None, None, input_dict)
            ret = get_generate_llama(
                None,
                None,
                input_dict,
                stage="generate_potential_interest",
                extra={
                    "user_id": user_id,
                    "retry": retry,
                },
            )
            potential_interest = extract_potential_interest(ret)

            flag = False
            for start in start_list:
                if potential_interest.startswith(start) or potential_interest.startswith(start, 1):
                    flag = True
                    break

            if not flag:
                raise ValueError(f"Bad potential interest format: {potential_interest}")

            return (user_id, potential_interest, True)

        except Exception:
            # 线程内只打印简短信息,避免日志混乱
            # 完整堆栈写到 stderr 也行,这里保守一点
            time.sleep(1)
            continue

    return (user_id, None, False)


def generate_user_potential_interest_mt(
    dataset: str,
    potential_file: str,
    port: int = 11442,
    max_retry: int = 20,
    num_workers: int = 16,
):
    """
    多线程版本:并发调用 LLM 总结 potential interest。
    写文件用锁保护,顺序不保证(但 user_id 在每行第一列,后续步骤按 user_id 索引)。
    """
    cfg = get_dataset_config(dataset)
    sortdict_path = cfg["sortdict_path"]
    cut_len = cfg["potential_cut_len"]

    prompt_obj = Prompt()
    potential_prefix = prompt_obj.summarize_potential[0]
    # potential_prefix, potential_postfix = potential_prefix[:-1], potential_prefix[-1:]

    generate_dict = load_potential_items_for_prompt(potential_file)

    output_file = os.path.join(
        sortdict_path,
        f"{dataset}_userpotential_{cut_len}_{str(MODEL_LIST)}.txt"
    )

    start_list = [
        "Based on", "The user", "This user", "It seems", "Given", "It appears",
        "**Based on", "**The user", "**This user", "**It seems", "**Given", "**It appears"
    ]

    error_users = []
    write_lock = threading.Lock()
    pbar_lock = threading.Lock()

    print(f"[MT] Generating potential interest with {num_workers} workers, total users: {len(generate_dict)}")

    with open(output_file, "w") as fout:
        with tqdm(total=len(generate_dict), desc="Generating potential interest (MT)") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_uid = {
                    executor.submit(
                        _task_generate_potential_interest,
                        user_id,
                        item_prompts,
                        potential_prefix,
                        start_list,
                        max_retry,
                    ): user_id
                    for user_id, item_prompts in generate_dict.items()
                }

                for future in as_completed(future_to_uid):
                    try:
                        user_id, potential_interest, success = future.result()
                    except Exception:
                        traceback.print_exc()
                        uid = future_to_uid[future]
                        with write_lock:
                            error_users.append(uid)
                        with pbar_lock:
                            pbar.update()
                        continue

                    if success:
                        with write_lock:
                            fout.write(f"{user_id}|{potential_interest}\n")
                            fout.flush()
                    else:
                        with write_lock:
                            error_users.append(user_id)

                    with pbar_lock:
                        pbar.update()

    error_file = os.path.join(
        sortdict_path,
        f"{dataset}_potential_interest_error_users.pkl"
    )
    with open(error_file, "wb") as f:
        pickle.dump(error_users, f)

    print(f"[Done] User potential interests saved to: {output_file}")
    print(f"[Info] Error users saved to: {error_file}, count = {len(error_users)}")

    return output_file


# =========================
# 5. 把 potential interest 加回 user profile (单线程 + 多线程)
# =========================

def load_userpotential(userpotential_file: str) -> Dict[str, str]:
    potential_dict = {}

    with open(userpotential_file, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        if not line.strip():
            continue
        user_id, potential = line.split("|", maxsplit=1)
        potential_dict[user_id] = potential

    return potential_dict


def load_original_user_profile(
    base_folder: str,
    user_id: str,
    profile_userid_float: bool = False,
):
    if profile_userid_float:
        file_user_id = str(float(int(user_id)))
    else:
        file_user_id = user_id

    profile_file = os.path.join(
        base_folder,
        "checkpoints",
        f"user_{file_user_id}_final_profile.pkl"
    )

    with open(profile_file, "rb") as f:
        cur_profile = pickle.load(f)

    return cur_profile


# ---------- 原单线程版本(保留) ----------

def modify_user_profile_single(
    dataset: str,
    userpotential_file: str,
    port: int = 11442,
    max_retry: int = 20,
):
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]
    profile_userid_float = cfg["profile_userid_float"]

    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[0]

    potential_dict = load_userpotential(userpotential_file)

    output_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")
    ensure_dir(output_folder)

    client = None
    error_users = []

    with tqdm(total=len(potential_dict), desc="Modifying user profiles") as pbar:
        for user_id, cur_potential in potential_dict.items():
            try:
                cur_final_profile = load_original_user_profile(
                    base_folder=base_folder,
                    user_id=user_id,
                    profile_userid_float=profile_userid_float
                )
            except Exception:
                traceback.print_exc()
                error_users.append(user_id)
                pbar.update()
                continue

            success = False
            for retry in range(max_retry):
                input_msg = (
                    f"User Profile: {cur_final_profile}.\n"
                    f"Potential Preference Analyses: {cur_potential}.\n"
                )
                input_dict = [{"role": "user", "content": input_msg}]

                try:
                    ret = get_generate_llama(client, modify_prefix, input_dict)
                    new_profile = extract_new_profile(ret)

                    output_file = os.path.join(
                        output_folder,
                        f"user_{user_id}_final_profile.pkl"
                    )
                    with open(output_file, "wb") as f:
                        pickle.dump(new_profile, f)

                    success = True
                    break

                except Exception:
                    traceback.print_exc()
                    time.sleep(1)

            if not success:
                error_users.append(user_id)

            pbar.update()

    error_file = os.path.join(
        base_folder,
        f"new_profile_error_users_{cut_len}.pkl"
    )
    with open(error_file, "wb") as f:
        pickle.dump(error_users, f)

    print(f"[Done] New user profiles saved to: {output_folder}")
    print(f"[Info] Error users saved to: {error_file}, count = {len(error_users)}")

    return output_folder


# ---------- 新多线程版本 ----------

def _task_modify_user_profile(
    user_id: str,
    cur_potential: str,
    base_folder: str,
    profile_userid_float: bool,
    modify_prefix,
    output_folder: str,
    max_retry: int,
) -> Tuple[str, bool, str]:
    """
    单个用户任务:返回 (user_id, success, reason)
    每个 worker 线程内:
      1. 读原始 profile
      2. 调 LLM 生成 new profile
      3. 写 pkl 文件 (每个用户独立文件,不需要锁)
    """
    try:
        cur_final_profile = load_original_user_profile(
            base_folder=base_folder,
            user_id=user_id,
            profile_userid_float=profile_userid_float
        )
    except Exception:
        return (user_id, False, "load_original_profile_failed")

    # input_msg = (
    #     f"User Profile: {cur_final_profile}.\n"
    #     f"Potential Preference Analyses: {cur_potential}.\n"
    # )
    input_msg = modify_prefix[1]['content'].format(user_profile=str(cur_final_profile), potential_preference_analyses=str(cur_potential))

    for retry in range(max_retry):
        # input_dict = [{"role": "user", "content": input_msg}]
        input_dict = [modify_prefix[0]] + [{"role": "user", "content": input_msg}]
        try:
            # ret = get_generate_llama(None, None, input_dict)
            ret = get_generate_llama(
                None,
                None,
                input_dict,
                stage="modify_user_profile",
                extra={
                    "user_id": user_id,
                    "retry": retry,
                    "cur_potential": cur_potential,
                },
            )
            new_profile = extract_new_profile(ret)
            new_profile_all = cur_final_profile + new_profile


            output_file = os.path.join(
                output_folder,
                f"user_{user_id}_final_profile.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(new_profile_all, f)

            return (user_id, True, "ok")
        except Exception:
            time.sleep(1)
            continue

    return (user_id, False, "llm_max_retry_exceeded")


def modify_user_profile_mt(
    dataset: str,
    userpotential_file: str,
    port: int = 11442,
    max_retry: int = 20,
    num_workers: int = 16,
):
    """
    多线程版本:并发改写 user profile。
    每个用户写独立 pkl 文件,天然无冲突,只需要锁错误列表 + 进度条。
    """
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]
    profile_userid_float = cfg["profile_userid_float"]

    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[-2]

    potential_dict = load_userpotential(userpotential_file)

    output_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")
    ensure_dir(output_folder)

    error_users = []
    error_lock = threading.Lock()
    pbar_lock = threading.Lock()

    print(f"[MT] Modifying user profile with {num_workers} workers, total users: {len(potential_dict)}")

    with tqdm(total=len(potential_dict), desc="Modifying user profiles (MT)") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_uid = {
                executor.submit(
                    _task_modify_user_profile,
                    user_id,
                    cur_potential,
                    base_folder,
                    profile_userid_float,
                    modify_prefix,
                    output_folder,
                    max_retry,
                ): user_id
                for user_id, cur_potential in potential_dict.items()
            }

            for future in as_completed(future_to_uid):
                try:
                    user_id, success, reason = future.result()
                except Exception:
                    traceback.print_exc()
                    uid = future_to_uid[future]
                    with error_lock:
                        error_users.append(uid)
                    with pbar_lock:
                        pbar.update()
                    continue

                if not success:
                    with error_lock:
                        error_users.append(user_id)

                with pbar_lock:
                    pbar.update()

    error_file = os.path.join(
        base_folder,
        f"new_profile_error_users_{cut_len}.pkl"
    )
    with open(error_file, "wb") as f:
        pickle.dump(error_users, f)

    print(f"[Done] New user profiles saved to: {output_folder}")
    print(f"[Info] Error users saved to: {error_file}, count = {len(error_users)}")

    return output_folder


# =========================
# 6. 重新生成 user embedding (单线程 + 多线程)
# =========================

# ---------- 原单线程版本(保留) ----------

def generate_useremb_from_new_profile_single(
    dataset: str,
    new_profile_folder: str,
    port: int = 11442,
    output_name: str = None,
):
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]

    client = None

    userembed_dict = {}

    files = [
        x for x in os.listdir(new_profile_folder)
        if x.endswith("_final_profile.pkl")
    ]

    with tqdm(total=len(files), desc="Generating user embeddings") as pbar:
        for file in files:
            user_id = file.split("_")[1]

            profile_file = os.path.join(new_profile_folder, file)
            with open(profile_file, "rb") as f:
                cur_profile = pickle.load(f)

            sentence = list2sentence(cur_profile)
            cur_embed = get_embedding(client, sentence)

            userembed_dict[user_id] = cur_embed
            pbar.update()

    userembeds = sorted(userembed_dict.items(), key=lambda x: x[0])

    if output_name is None:
        output_name = f"{dataset}_potential_profile_{cut_len}.useremb"

    output_file = os.path.join(base_folder, output_name)

    with open(output_file, "w") as f:
        f.write("uid:token|user_emb:float_seq\n")

        with tqdm(total=len(userembeds), desc="Saving .useremb") as pbar:
            for user_id, emb in userembeds:
                f.write(f"{user_id}|")
                for value in emb:
                    f.write(f"{value}, ")
                f.write("\n")
                pbar.update()

    print(f"[Done] RecBole user embedding saved to: {output_file}")
    return output_file


# ---------- 新多线程版本 ----------

def _task_get_embedding(
    user_id: str,
    profile_file: str,
) -> Tuple[str, list, bool]:
    """
    单个用户 embedding 任务:返回 (user_id, embedding_or_None, success)
    """
    try:
        with open(profile_file, "rb") as pf:
            cur_profile = pickle.load(pf)

        sentence = list2sentence(cur_profile)
        cur_embed = get_embedding(None, sentence)
        return (user_id, cur_embed, True)
    except Exception:
        return (user_id, None, False)


def generate_useremb_from_new_profile_mt(
    dataset: str,
    new_profile_folder: str,
    port: int = 11442,
    output_name: str = None,
    num_workers: int = 16,
):
    """
    多线程版本:并发生成 embedding。
    embedding 收集到 dict 后统一按 user_id 排序写文件,保证文件内容可复现。
    """
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]

    files = [
        x for x in os.listdir(new_profile_folder)
        if x.endswith("_final_profile.pkl")
    ]

    userembed_dict = {}
    failed_users = []
    dict_lock = threading.Lock()
    pbar_lock = threading.Lock()

    print(f"[MT] Generating user embeddings with {num_workers} workers, total profiles: {len(files)}")

    with tqdm(total=len(files), desc="Generating user embeddings (MT)") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_uid = {}
            for file in files:
                user_id = file.split("_")[1]
                profile_file = os.path.join(new_profile_folder, file)
                fut = executor.submit(_task_get_embedding, user_id, profile_file)
                future_to_uid[fut] = user_id

            for future in as_completed(future_to_uid):
                try:
                    user_id, emb, success = future.result()
                except Exception:
                    traceback.print_exc()
                    uid = future_to_uid[future]
                    with dict_lock:
                        failed_users.append(uid)
                    with pbar_lock:
                        pbar.update()
                    continue

                if success:
                    with dict_lock:
                        userembed_dict[user_id] = emb
                else:
                    with dict_lock:
                        failed_users.append(user_id)

                with pbar_lock:
                    pbar.update()

    if failed_users:
        print(f"[Warn] {len(failed_users)} users failed to embed: {failed_users[:10]}...")

    userembeds = sorted(userembed_dict.items(), key=lambda x: x[0])

    if output_name is None:
        output_name = f"{dataset}_potential_profile_{cut_len}.useremb"

    output_file = os.path.join(base_folder, output_name)

    with open(output_file, "w") as f:
        f.write("uid:token|user_emb:float_seq\n")
        with tqdm(total=len(userembeds), desc="Saving .useremb") as pbar:
            for user_id, emb in userembeds:
                f.write(f"{user_id}|")
                for value in emb:
                    f.write(f"{value}, ")
                f.write("\n")
                pbar.update()

    print(f"[Done] RecBole user embedding saved to: {output_file}")
    return output_file


# =========================
# 6.5 重试 error users (单线程 + 多线程)
# =========================

# ---------- 原单线程版本(保留) ----------

def retry_error_users_single(
    dataset: str,
    userpotential_file: str,
    port: int = 11442,
    max_retry: int = 20,
):
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]
    profile_userid_float = cfg["profile_userid_float"]

    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[0]

    error_file = os.path.join(base_folder, f"new_profile_error_users_{cut_len}.pkl")
    with open(error_file, "rb") as f:
        error_users = pickle.load(f)

    print(f"[Info] Found {len(error_users)} error users to retry.")

    potential_dict = load_userpotential(userpotential_file)
    output_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")
    ensure_dir(output_folder)

    client = None
    still_error_users = []
    success_users = []

    with tqdm(total=len(error_users), desc="Retrying error users (modify profile)") as pbar:
        for user_id in error_users:
            if user_id not in potential_dict:
                print(f"[Warn] user_id {user_id} not found in potential_dict, skipping.")
                still_error_users.append(user_id)
                pbar.update()
                continue

            cur_potential = potential_dict[user_id]

            try:
                cur_final_profile = load_original_user_profile(
                    base_folder=base_folder,
                    user_id=user_id,
                    profile_userid_float=profile_userid_float
                )
            except Exception:
                traceback.print_exc()
                still_error_users.append(user_id)
                pbar.update()
                continue

            success = False
            for retry in range(max_retry):
                input_msg = (
                    f"User Profile: {cur_final_profile}.\n"
                    f"Potential Preference Analyses: {cur_potential}.\n"
                )
                input_dict = [{"role": "user", "content": input_msg}]

                try:
                    ret = get_generate_llama(client, modify_prefix, input_dict)
                    new_profile = extract_new_profile(ret)

                    output_file = os.path.join(
                        output_folder,
                        f"user_{user_id}_final_profile.pkl"
                    )
                    with open(output_file, "wb") as f:
                        pickle.dump(new_profile, f)

                    success = True
                    success_users.append(user_id)
                    break

                except Exception:
                    traceback.print_exc()
                    time.sleep(1)

            if not success:
                still_error_users.append(user_id)

            pbar.update()

    with open(error_file, "wb") as f:
        pickle.dump(still_error_users, f)

    print(f"[Done] Retry modify profile: {len(success_users)} succeeded, {len(still_error_users)} still failed.")
    return success_users, still_error_users


def append_useremb_for_users(
    dataset: str,
    user_ids: List[str],
    useremb_file: str,
    port: int = 11442,
):
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]

    new_profile_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")
    client = None

    append_count = 0
    skip_users = []

    with open(useremb_file, "a") as f:
        with tqdm(total=len(user_ids), desc="Appending embeddings for retried users") as pbar:
            for user_id in user_ids:
                profile_file = os.path.join(
                    new_profile_folder,
                    f"user_{user_id}_final_profile.pkl"
                )

                if not os.path.exists(profile_file):
                    print(f"[Warn] Profile file not found for user {user_id}, skipping.")
                    skip_users.append(user_id)
                    pbar.update()
                    continue

                try:
                    with open(profile_file, "rb") as pf:
                        cur_profile = pickle.load(pf)

                    sentence = list2sentence(cur_profile)
                    cur_embed = get_embedding(client, sentence)

                    f.write(f"{user_id}|")
                    for value in cur_embed:
                        f.write(f"{value}, ")
                    f.write("\n")

                    append_count += 1
                except Exception:
                    traceback.print_exc()
                    skip_users.append(user_id)

                pbar.update()

    print(f"[Done] Appended {append_count} user embeddings to: {useremb_file}")
    if skip_users:
        print(f"[Warn] Skipped users (profile missing or embed failed): {skip_users}")

    return append_count, skip_users


# ---------- 新多线程版本 ----------

def retry_error_users_mt(
    dataset: str,
    userpotential_file: str,
    port: int = 11442,
    max_retry: int = 20,
    num_workers: int = 8,
):
    """
    多线程版本:重新处理 modify_user_profile 阶段的 error users。
    error users 通常数量不多,worker 数可以小一些。
    """
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]
    profile_userid_float = cfg["profile_userid_float"]

    prompt_obj = Prompt()
    modify_prefix = prompt_obj.add_potential[-2]

    error_file = os.path.join(base_folder, f"new_profile_error_users_{cut_len}.pkl")
    with open(error_file, "rb") as f:
        error_users = pickle.load(f)

    print(f"[MT] Found {len(error_users)} error users to retry with {num_workers} workers.")

    potential_dict = load_userpotential(userpotential_file)
    output_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")
    ensure_dir(output_folder)

    still_error_users = []
    success_users = []
    result_lock = threading.Lock()
    pbar_lock = threading.Lock()

    # 提前筛掉 potential_dict 里没有的 user
    valid_tasks = []
    for user_id in error_users:
        if user_id not in potential_dict:
            print(f"[Warn] user_id {user_id} not found in potential_dict, skipping.")
            still_error_users.append(user_id)
            continue
        valid_tasks.append((user_id, potential_dict[user_id]))

    with tqdm(total=len(valid_tasks), desc="Retrying error users (MT)") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_uid = {
                executor.submit(
                    _task_modify_user_profile,
                    user_id,
                    cur_potential,
                    base_folder,
                    profile_userid_float,
                    modify_prefix,
                    output_folder,
                    max_retry,
                ): user_id
                for user_id, cur_potential in valid_tasks
            }

            for future in as_completed(future_to_uid):
                try:
                    user_id, success, reason = future.result()
                except Exception:
                    traceback.print_exc()
                    uid = future_to_uid[future]
                    with result_lock:
                        still_error_users.append(uid)
                    with pbar_lock:
                        pbar.update()
                    continue

                with result_lock:
                    if success:
                        success_users.append(user_id)
                    else:
                        still_error_users.append(user_id)

                with pbar_lock:
                    pbar.update()

    with open(error_file, "wb") as f:
        pickle.dump(still_error_users, f)

    print(f"[Done] Retry modify profile: {len(success_users)} succeeded, {len(still_error_users)} still failed.")
    return success_users, still_error_users


def append_useremb_for_users_mt(
    dataset: str,
    user_ids: List[str],
    useremb_file: str,
    port: int = 11442,
    num_workers: int = 16,
):
    """
    多线程版本:并发生成 embedding,然后串行追加到 useremb 文件末尾。
    """
    cfg = get_dataset_config(dataset)
    base_folder = cfg["base_folder"]
    cut_len = cfg["potential_cut_len"]

    new_profile_folder = os.path.join(base_folder, f"checkpoints_new_{cut_len}")

    tasks = []
    skip_users = []
    for user_id in user_ids:
        profile_file = os.path.join(
            new_profile_folder,
            f"user_{user_id}_final_profile.pkl"
        )
        if not os.path.exists(profile_file):
            print(f"[Warn] Profile file not found for user {user_id}, skipping.")
            skip_users.append(user_id)
            continue
        tasks.append((user_id, profile_file))

    print(f"[MT] Appending embeddings with {num_workers} workers, total: {len(tasks)}")

    results = {}  # user_id -> emb
    result_lock = threading.Lock()
    pbar_lock = threading.Lock()

    with tqdm(total=len(tasks), desc="Embedding retried users (MT)") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_uid = {
                executor.submit(_task_get_embedding, user_id, profile_file): user_id
                for user_id, profile_file in tasks
            }

            for future in as_completed(future_to_uid):
                try:
                    user_id, emb, success = future.result()
                except Exception:
                    traceback.print_exc()
                    uid = future_to_uid[future]
                    with result_lock:
                        skip_users.append(uid)
                    with pbar_lock:
                        pbar.update()
                    continue

                if success:
                    with result_lock:
                        results[user_id] = emb
                else:
                    with result_lock:
                        skip_users.append(user_id)

                with pbar_lock:
                    pbar.update()

    # 串行追加到文件
    append_count = 0
    with open(useremb_file, "a") as f:
        for user_id, emb in results.items():
            f.write(f"{user_id}|")
            for value in emb:
                f.write(f"{value}, ")
            f.write("\n")
            append_count += 1

    print(f"[Done] Appended {append_count} user embeddings to: {useremb_file}")
    if skip_users:
        print(f"[Warn] Skipped users (profile missing or embed failed): {skip_users}")

    return append_count, skip_users


def retry_and_append_pipeline_mt(
    dataset: str,
    userpotential_file: str,
    useremb_file: str,
    port: int = 11442,
    max_retry: int = 20,
    retry_workers: int = 8,
    embed_workers: int = 16,
):
    """
    多线程版本:一键重试 error users 完整流程。
    """
    print("=" * 80)
    print(f"[Retry Pipeline MT] Dataset: {dataset}")
    print(f"[Retry Pipeline MT] Target useremb file: {useremb_file}")
    print(f"[Retry Pipeline MT] retry_workers={retry_workers}, embed_workers={embed_workers}")
    print("=" * 80)

    success_users, still_error_users = retry_error_users_mt(
        dataset=dataset,
        userpotential_file=userpotential_file,
        port=port,
        max_retry=max_retry,
        num_workers=retry_workers,
    )

    if not success_users:
        print("[Info] No users recovered, nothing to append.")
        return

    append_count, skip_users = append_useremb_for_users_mt(
        dataset=dataset,
        user_ids=success_users,
        useremb_file=useremb_file,
        port=port,
        num_workers=embed_workers,
    )

    print("=" * 80)
    print(f"[Retry Pipeline MT Done] Appended {append_count} users.")
    if still_error_users:
        print(f"[Retry Pipeline MT] Still failing users: {still_error_users}")
    print("=" * 80)


# =========================
# 7. 一键运行完整 pipeline (单线程 + 多线程)
# =========================

def run_single_thread_pipeline(
    dataset: str = "ml-25m",
    port: int = 11442,
    cut_size: int = None,
):
    """
    原单线程完整流程(保留)。
    """
    cfg = get_dataset_config(dataset)
    if cut_size is None:
        cut_size = cfg["potential_cut_len"]

    print("=" * 80)
    print(f"[Single Thread] Dataset: {dataset}, cut_size: {cut_size}")
    print("=" * 80)

    potential_file = generate_potential_items_single(dataset=dataset, cut_size=cut_size)
    userpotential_file = generate_user_potential_interest_single(
        dataset=dataset, potential_file=potential_file, port=port
    )
    new_profile_folder = modify_user_profile_single(
        dataset=dataset, userpotential_file=userpotential_file, port=port
    )
    useremb_file = generate_useremb_from_new_profile_single(
        dataset=dataset, new_profile_folder=new_profile_folder, port=port
    )

    print("=" * 80)
    print("[All Done]")
    print(f"Final useremb file: {useremb_file}")
    print("=" * 80)

def resolve_existing_potential_file(
    dataset: str,
    potential_file: str = None,
    cut_size: int = None,
):
    """
    读取已有 potential items 文件。
    默认会读取:
        ./sortdict/{dataset_short}_potential_{cut_size}.txt

    例如 amazon-CDs_and_Vinyl + cut_size=20:
        ./sortdict/amazon_potential_20.txt
    """
    cfg = get_dataset_config(dataset)
    dataset_short = cfg["dataset_short"]
    sortdict_path = cfg["sortdict_path"]

    if cut_size is None:
        cut_size = cfg["potential_cut_len"]

    if potential_file is None:
        potential_file = os.path.join(
            sortdict_path,
            f"{dataset_short}_potential_{cut_size}.txt"
        )

    if not os.path.exists(potential_file):
        raise FileNotFoundError(
            f"Potential file not found: {potential_file}\n"
            f"Please check whether amazon_potential_20.txt exists."
        )

    print(f"[Info] Using existing potential file: {potential_file}")
    return potential_file


def run_multi_thread_pipeline_from_existing_potential(
    dataset: str = "amazon-CDs_and_Vinyl",
    potential_file: str = None,
    port: int = 11442,
    cut_size: int = None,
    potential_interest_workers: int = 16,
    modify_profile_workers: int = 16,
    embed_workers: int = 16,
):
    """
    从已有 potential items 文件开始运行完整流程:
      1. 读取已有 amazon_potential_20.txt
      2. LLM 总结 potential interest
      3. 把 potential interest 加回 user profile
      4. 重新生成 user embedding

    不再重新调用 generate_potential_items_single()。
    """
    cfg = get_dataset_config(dataset)
    if cut_size is None:
        cut_size = cfg["potential_cut_len"]

    print("=" * 80)
    print(f"[MT From Existing Potential] Dataset: {dataset}, cut_size: {cut_size}")
    print(f"  potential_interest_workers = {potential_interest_workers}")
    print(f"  modify_profile_workers     = {modify_profile_workers}")
    print(f"  embed_workers              = {embed_workers}")
    print("=" * 80)

    # 1. 直接读取已有 potential items 文件
    potential_file = resolve_existing_potential_file(
        dataset=dataset,
        potential_file=potential_file,
        cut_size=cut_size,
    )

    # 2. LLM 总结 potential interest
    userpotential_file = generate_user_potential_interest_mt(
        dataset=dataset,
        potential_file=potential_file,
        port=port,
        num_workers=potential_interest_workers,
    )

    # 3. potential interest 加回 user profile
    new_profile_folder = modify_user_profile_mt(
        dataset=dataset,
        userpotential_file=userpotential_file,
        port=port,
        num_workers=modify_profile_workers,
    )

    # 4. new profile 重新生成 embedding
    useremb_file = generate_useremb_from_new_profile_mt(
        dataset=dataset,
        new_profile_folder=new_profile_folder,
        port=port,
        num_workers=embed_workers,
    )

    print("=" * 80)
    print("[All Done - MT From Existing Potential]")
    print(f"Potential file: {potential_file}")
    print(f"User potential file: {userpotential_file}")
    print(f"Final useremb file: {useremb_file}")
    print("=" * 80)



def run_multi_thread_pipeline(
    dataset: str = "ml-25m",
    port: int = 11442,
    cut_size: int = None,
    # 各阶段独立的 worker 数,方便针对 vLLM / embedding 服务分别调优
    potential_interest_workers: int = 16,
    modify_profile_workers: int = 16,
    embed_workers: int = 16,
):
    """
    多线程完整流程。各阶段 worker 数独立可调:
      - potential_interest_workers: 调 vLLM chat completion (压力大,看 GPU 利用率)
      - modify_profile_workers:     调 vLLM chat completion
      - embed_workers:              调 embedding 服务 (通常更轻,可设大一些)
    """
    cfg = get_dataset_config(dataset)
    if cut_size is None:
        cut_size = cfg["potential_cut_len"]

    print("=" * 80)
    print(f"[Multi Thread] Dataset: {dataset}, cut_size: {cut_size}")
    print(f"  potential_interest_workers = {potential_interest_workers}")
    print(f"  modify_profile_workers     = {modify_profile_workers}")
    print(f"  embed_workers              = {embed_workers}")
    print("=" * 80)

    # 1. 找 potential items (无 LLM,单线程即可)
    potential_file = generate_potential_items_single(dataset=dataset, cut_size=cut_size)

    # 2. LLM 总结 potential interest (多线程)
    userpotential_file = generate_user_potential_interest_mt(
        dataset=dataset,
        potential_file=potential_file,
        port=port,
        num_workers=potential_interest_workers,
    )

    # 3. potential interest 加回 user profile (多线程)
    new_profile_folder = modify_user_profile_mt(
        dataset=dataset,
        userpotential_file=userpotential_file,
        port=port,
        num_workers=modify_profile_workers,
    )

    # 4. new profile 重新生成 embedding (多线程)
    useremb_file = generate_useremb_from_new_profile_mt(
        dataset=dataset,
        new_profile_folder=new_profile_folder,
        port=port,
        num_workers=embed_workers,
    )

    print("=" * 80)
    print("[All Done - MT]")
    print(f"Final useremb file: {useremb_file}")
    print("=" * 80)


if __name__ == "__main__":
    # ============ 多线程版本(推荐) ============
    run_multi_thread_pipeline(
        dataset="amazon-CDs_and_Vinyl",
        port=11442,
        cut_size=20,
        potential_interest_workers=20,  # 在这里调
        modify_profile_workers=20,      # 在这里调
        embed_workers=20,               # 在这里调
    )
    # run_multi_thread_pipeline_from_existing_potential(
    #     dataset="amazon-CDs_and_Vinyl",
    #     # potential_file="./sortdict/amazon-CDs_and_Vinyl_userpotential_20.txt",
    #     potential_file="./sortdict/amazon_potential_20.txt",
    #     port=11442,
    #     cut_size=20,
    #     potential_interest_workers=20,
    #     modify_profile_workers=20,
    #     embed_workers=20,
    # )

    # ============ 原单线程版本(保留) ============
    # run_single_thread_pipeline(
    #     dataset="amazon-CDs_and_Vinyl",
    #     port=11442,
    #     cut_size=20
    # )

    # ============ 多线程版本重试 error users ============
    # retry_and_append_pipeline_mt(
    #     dataset="amazon-CDs_and_Vinyl",
    #     userpotential_file="./sortdict/amazon-CDs_and_Vinyl_userpotential_20.txt",
    #     useremb_file="./logs/2026-05-12_19:35:51/amazon-CDs_and_Vinyl_potential_profile_20.useremb",
    #     port=11442,
    #     max_retry=20,
    #     retry_workers=8,
    #     embed_workers=16,
    # )