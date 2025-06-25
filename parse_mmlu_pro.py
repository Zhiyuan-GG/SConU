import argparse
import json
import random

import numpy as np
import tqdm
import re
import os

import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import pandas as pd
import datasets
from datasets import Dataset

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--generate-model',
                    default=r'D:\projects\cache_model\vicuna-7b-v1.5',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--test-data-path',
                    default=r'../row_data/MMLU-Pro/test-00000-of-00001.parquet',
                    help='local path of row dataset')
parser.add_argument('--validation-data-path',
                    default=r'../row_data/MMLU-Pro/validation-00000-of-00001.parquet',
                    help='local path of row dataset')
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--few-shot-num',
                    default=3,
                    help='for few-shot prompt')
parser.add_argument('--max-num-for-each-category',
                    default=500,
                    help='for save')
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
# model_name for path of saved parsed dataset
model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
print('Generative LLM: ', model_name)
# ----------------------------------------------------------------------------------------------------------------------
# Set seed for recurrence
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# ----------------------------------------------------------------------------------------------------------------------
# for input_ids length (allowed by llm)
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.generate_model,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.generate_model,
                                                      local_files_only=True,
                                                      torch_dtype=torch.float16,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
max_input_ids_length = generative_llm.config.max_position_embeddings
print('LLM max input ids length: ', max_input_ids_length)
# ----------------------------------------------------------------------------------------------------------------------
def form_options(options: list, answer: str):
    option_str = '{'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        if opt[-1] == '.':
            temp = opt[:-1]
        else:
            temp = opt
        if o == answer:
            reference_answer = f'{o}: {temp}'
        option_str += f'{o}: {temp}; '
    option_str = option_str[:-2] + '}'
    return option_str, reference_answer

def load_dataset(path):
    df = pd.read_parquet(path)
    dict_list = df.to_dict(orient='records')
    return dict_list

if __name__ == "__main__":
    for_save_json = []
    # load
    validation_data_li = load_dataset(args.validation_data_path)  # 5 x 14 = 70, 0-69
    test_data_li = load_dataset(args.test_data_path)  # 70-12256
    # category list
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']
    # system prompt
    prefix = '### System:\nAnswer the following multiple-choice question by giving the most appropriate response. ' \
             'Answer should be one among [A, B, C, D, E, F, G, H, I, J].\n\n'
    # few shot prompt
    few_shot_prompts = {c: '' for c in categories}
    prompt_num = {c: 0 for c in categories}
    for d in validation_data_li:
        if prompt_num[d['category']] == args.few_shot_num:
            continue
        # form question
        query = d['question']
        query = query.replace('\n', ' ')
        query = re.sub(r"\s+", " ", query.strip())
        # A, B, C, ...
        reference = d['answer']
        # form few_shot_prompt
        few_shot_prompts[d['category']] += '### User:\n' + query + '\n' + form_options(d['options'], reference)[0] + \
                                           '\n### Assistant:\n' + reference + '\n\n'
        prompt_num[d['category']] += 1

    # for parse test data
    save_num = {c: 0 for c in categories}
    dataset = {}
    dataset['prompt'] = []
    dataset['question'] = []
    dataset['options'] = []
    dataset['answer'] = []
    dataset['category'] = []
    dataset['id'] = []
    for sample_idx, sample in enumerate(tqdm.tqdm(test_data_li)):
        question_id = sample['question_id']
        category = sample['category']
        question = sample['question']
        # form question
        question = question.replace('\n', ' ')
        question = re.sub(r"\s+", " ", question.strip())
        options = sample['options']
        answer = sample['answer']
        # form prompt
        few_shot_prompt = few_shot_prompts[sample['category']]
        prompt = prefix + few_shot_prompt + '### User:\n' + question + '\n' + form_options(options, answer)[0] + '\n### Assistant:\n'
        # form answer
        formed_answer = form_options(options, answer)[1]  # A: xxx
        # form option
        opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        formed_option = []
        for opt, o in zip(options, opts):
            if opt[-1] == '.':
                temp = opt[:-1]
            else:
                temp = opt
            formed_option.append(f'{o}: {temp}')
        # encode prompt
        input_ids = generative_tokenizer.encode(prompt)
        if len(input_ids) < max_input_ids_length:
            if save_num[category] < args.max_num_for_each_category:
                assert formed_answer in prompt
                assert formed_answer in formed_option

                dataset['prompt'].append(prompt)
                dataset['question'].append(question)
                dataset['options'].append(formed_option)
                dataset['answer'].append(answer)
                dataset['category'].append(category)
                dataset['id'].append(str(sample_idx) + '_' + str(question_id))
                # save num
                save_num[category] += 1
                # save_json
                for_save_json.append({})
                for_save_json[-1]['prompt'] = prompt
                for_save_json[-1]['question'] = question
                for_save_json[-1]['options'] = formed_option
                for_save_json[-1]['answer'] = answer
                for_save_json[-1]['category'] = category
                for_save_json[-1]['id'] = str(sample_idx) + '_' + str(question_id)

    print('Saved question-answer pairs: ', save_num)
    print('Total num: ', sum([save_num[c] for c in categories]))
    # ------------------------------------------------------------------------------------------------------------------
    # save
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(dataset_df)
    dataset.save_to_disk(f'{args.data_dir}/mmlu_pro_{model_name}')

    with open(r'../row_data/MMLU-Pro/mmlu_pro.json', 'w') as json_file:
        json.dump(for_save_json, json_file, indent=4)
