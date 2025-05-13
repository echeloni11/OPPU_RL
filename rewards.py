from typing import List

import torch
import torch.nn as nn
# import bitsandbytes as bnb # Keep if you were using quantization
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from transformers import pipeline, BitsAndBytesConfig # Keep if needed
import argparse
from rank_bm25 import BM25Okapi
# --- RL Imports ---
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
# --- End RL Imports ---
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM # No longer needed for SFT
import transformers # Still needed for utility data collator potentially, or args if reused
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training # Keep prepare_model_for_kbit_training if using quantization

# def calculate_reward(generated_texts, reference_texts):
#     """Calculates ROUGE-L F1 score as reward."""
#     rewards = []
#     for gen, ref in zip(generated_texts, reference_texts):
#         # Ensure ref is not empty, handle potential extraction errors
#         if not ref:
#              print("Warning: Empty reference text encountered.")
#              rewards.append(torch.tensor(0.0)) # Assign zero reward for empty reference
#              continue
#         try:
#             scores = scorer.score(ref, gen)
#             # Using F-measure of ROUGE-L
#             reward = scores['rougeL'].fmeasure
#             rewards.append(torch.tensor(reward))
#         except Exception as e:
#             print(f"Error calculating reward for: gen='{gen}', ref='{ref}'. Error: {e}")
#             rewards.append(torch.tensor(0.0)) # Assign zero reward on error
#     return rewards

# Implement the reward function for different tasks

FORMAT_PENALTY = -10
LENGTH_PENALTY = -1

def get_reward_func_for_task(task_name):
    if task_name == "citation":
        return calculate_citation_reward
    elif task_name == "movie_tagging":
        return calculate_movie_tagging_reward
    elif task_name == "news_categorize":
        return calculate_news_categorize_reward
    elif task_name == "news_headline":
        return calculate_news_headline_reward
    elif task_name == "product_rating":
        return calculate_product_rating_reward
    elif task_name == "scholarly_title":
        return calculate_scholarly_title_reward
    elif task_name == "tweet_paraphrase":
        return calculate_tweet_paraphrase_reward
    else:
        raise ValueError(f"Invalid task name: {task_name}")

# reward function should look like this:

# def reward_func(
#     completions: List[str],
#     *,
#     reference: List[str] | None = None,
#     **kwargs,
# ):
#     """Simple reward: +1 if completion ≈ reference (string match).
#     Return NaN for samples where reference is missing so they are ignored.
#     """
#     out: List[float] = []
#     for pred, ref in zip(completions, reference):
#         if ref is None:
#             out.append(float("nan"))  # excluded from loss
#         else:
#             out.append(float(pred.strip() == ref.strip()))
#     return out

def calculate_citation_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    # choices: '[1]' or '[2]'

    rewards = []
    for completion, ref in zip(completions, reference):
        # if completion == ref:
        #     reward = 1
        # else:
        #     reward = 0
        # # penalize the length
        # # remove original part if exists
        # if completion.startswith('[1]') or completion.startswith('[2]'):
        #     completion = completion[len('[1]'):]
        # # penalize the length
        # reward -= len(completion)
        # rewards.append(reward)

        # 1. legal output
        if completion.startswith('[1]') or completion.startswith('[2]'):
            useful_answer = completion[:len('[1]')]
            other_answer = completion[len('[1]'):]
        else:
            useful_answer = None
            other_answer = completion
        
        # 2. correct answer
        if useful_answer == ref:
            reward = 1
        elif useful_answer is not None:
            reward = 0
        else:
            reward = FORMAT_PENALTY

        # 3. penalize the length
        reward += len(other_answer) * LENGTH_PENALTY
        rewards.append(reward)

    return rewards

def calculate_movie_tagging_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    # choices: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]

    rewards = []
    choices = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]
    # for completion, ref in zip(completions, reference):
    #     if completion == ref:
    #         reward = 1
    #     else:
    #         reward = 0
    #     # penalize the length
    #     reward -= len(completion)
    #     for choice in choices:
    #         if completion.startswith(choice):
    #             completion = completion[len(choice):]
    #             break
    #     reward -= len(completion)
    #     rewards.append(reward)

    for completion, ref in zip(completions, reference):
        # 1. legal output
        useful_answer = None
        other_answer = completion
        for choice in choices:
            if completion.startswith(choice):
                useful_answer = choice
                other_answer = completion[len(choice):]
                break
        
        # 2. correct answer
        if useful_answer == ref:
            reward = 1
        elif useful_answer is not None:
            reward = 0
        else:
            reward = FORMAT_PENALTY

        # 3. penalize the length
        reward += len(other_answer) * LENGTH_PENALTY
        rewards.append(reward)

    return rewards

def calculate_news_categorize_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    # choices: [travel, education, parents, style & beauty, entertainment, food & drink, science & technology, business, sports, healthy living, women, politics, crime, culture & arts, religion]

    rewards = []
    choices = ["travel", "education", "parents", "style & beauty", "entertainment", "food & drink", "science & technology", "business", "sports", "healthy living", "women", "politics", "crime", "culture & arts", "religion"]
    for completion, ref in zip(completions, reference):
        # 1. legal output
        useful_answer = None
        other_answer = completion
        for choice in choices:
            if completion.startswith(choice):
                useful_answer = choice
                other_answer = completion[len(choice):]
                break
        
        # 2. correct answer
        if useful_answer == ref:
            reward = 1
        elif useful_answer is not None:
            reward = 0
        else:
            reward = FORMAT_PENALTY

        # 3. penalize the length
        reward += len(other_answer) * LENGTH_PENALTY
        rewards.append(reward)

    return rewards
        

def calculate_product_rating_reward(
    prompts: str,
    completions: List[str],
    reference: List[str] | None = None,
    **kwargs,
):
    # TODO: Implement product rating reward
    # num_generations = 4
    # each is score from 1 to 5
    # completions: something like ['5','5','5','4','5','5','4','5']
    # reference: something like ['5','5','5','5','4','4','4','4']
    # larger the difference, the worse the reward
    # reward = -abs(completion - reference)
    import math
    rewards = []
    for completion, ref in zip(completions, reference):
        # format check
        # words = completion.split()
        # has_digit = False
        # for word in words:
        #     if word.isdigit() and int(word) >= 1 and int(word) <= 5:
        #         has_digit = True
        #         digit = int(word)
        #         break
        # if not has_digit:
        #     reward = -10
        # else:
        #     reward = -abs(float(digit) - float(ref))

        # remove the part before the first digit

        # 1. legal output
        useful_answer = None
        other_answer = completion
        for digit in ['1', '2', '3', '4', '5']:
            if completion.startswith(digit):
                useful_answer = digit
                other_answer = completion[len(digit):]
                break
            
        # 2. correct answer
        if useful_answer is not None:
            reward = 1 - abs(float(useful_answer) - float(ref))
        else:
            reward = FORMAT_PENALTY

        # 3. penalize the length
        reward += len(other_answer) * LENGTH_PENALTY
        
        rewards.append(reward)
    return rewards

# def calculate_scholarly_title_reward(
#     prompts: str,
#     completions: List[str],
#     *,
#     reference: List[str] | None = None,
#     **kwargs,
# ):
#     # TODO: Implement scholarly title reward
#     raise NotImplementedError("Scholarly title reward not implemented")

from evaluate import load

# 预加载指标，避免在每个 batch 里重复初始化（BERTScore 会跑模型）
_bertscore = load("bertscore")             # 默认 roberta-large_L17 F1
_rouge     = load("rouge")                 # 自带 rouge1/2/L

def _single_score(preds, refs):
    # # 先裁剪：只取首行 & 至多 60 字符
    # preds = [p.split('\n', 1)[0][:60] for p in preds]

    # BERTScore Precision
    bert_p = _bertscore.compute(predictions=preds, references=refs,
                                lang="en", rescale_with_baseline=True)["precision"]

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL_f1 = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]

    rewards = []
    for bp, rp, pred in zip(bert_p, rougeL_f1, preds):
        score = 0.5*bp + 0.5*rp
        len_penalty = max(0, len(pred.split()) - len(refs[0].split())) / len(refs[0].split())
        rewards.append(float(score - len_penalty))
    return rewards

# def _single_score(preds: List[str], refs: List[str]) -> List[float]:
#     """
#     计算 (BERTScore_F1 + ROUGE-L_F1) / 2 作为 reward，范围大致 [0, 1]。
#     """
#     # ① BERTScore
#     bert_out = _bertscore.compute(
#         predictions=preds,
#         references=refs,
#         lang="en",  # 如果全是英文标题
#         rescale_with_baseline=True,  # 让分数≈0.0~1.0，更稳定
#     )
#     bert_f1 = bert_out["f1"]               # List[float]

#     # ② ROUGE-L
#     rouge_out = _rouge.compute(
#         predictions=preds,
#         references=refs,
#         rouge_types=["rougeL"]
#     )
#     # evaluate 的 rouge 返回字典，值是平均值；为了按样本取分，需要直接用 rouge-score
#     # 这里偷个懒：把整体 average 拆回每条（假设相同分布）；若要精确，可改成逐对计算。
#     rougeL_f1 = rouge_out["rougeL"]  # scalar

#     # 若想逐对精确，可改：
#     # from rouge_score import rouge_scorer
#     # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     # rougeL_f1 = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]

#     # ③ 汇总
#     rewards = []
#     for b, r in zip(bert_f1, [rougeL_f1]*len(preds)):
#         reward = 0.5 * b + 0.5 * r
#         rewards.append(float(reward))      # 转成 python float 方便 JSON‑serialise
#     return rewards


def calculate_scholarly_title_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    """
    Args:
        prompts:      原始 prompt；这里通常不用，但保留签名便于 GRPO 统一调用
        completions:  生成的标题列表，长度 = batch_size * num_generations
        reference:    标准标题，与 completions 一一对应
    Returns:
        List[float]   奖励标量；数值越大越好
    """
    if reference is None:
        raise ValueError("`reference` 不能为空，需提供真标题")

    if len(completions) != len(reference):
        raise ValueError("`completions` 与 `reference` 长度不一致")

    # === 核心评分 ===
    rewards = _single_score(completions, reference)

    return rewards

def calculate_tweet_paraphrase_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    # same as scholarly title reward
    if reference is None:
        raise ValueError("`reference` 不能为空，需提供真标题")

    if len(completions) != len(reference):
        raise ValueError("`completions` 与 `reference` 长度不一致")

    # === 核心评分 ===
    rewards = _single_score(completions, reference)

    return rewards

def calculate_news_headline_reward(
    prompts: str,
    completions: List[str],
    *,
    reference: List[str] | None = None,
    **kwargs,
):
    # same as scholarly title reward
    if reference is None:
        raise ValueError("`reference` 不能为空，需提供真标题")

    if len(completions) != len(reference):
        raise ValueError("`completions` 与 `reference` 长度不一致")

    # === 核心评分 ===
    rewards = _single_score(completions, reference)

    return rewards