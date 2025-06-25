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
                    default=r'D:\projects\cache_model\Llama-3.2-3B-Instruct',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--row-data-path',
                    default=r'../row_data/MedMCQA',
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
parser.add_argument('--max-num',
                    default=2000,
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
    opts = ['A', 'B', 'C', 'D']
    for opt, o in zip(options, opts):
        if opt[-1] == '.':
            temp = opt[:-1]
        else:
            temp = opt
        temp = temp.strip()
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
    applied_qa = 0
    for_save_json = []
    # load validation data
    validation_name = 'validation-00000-of-00001.parquet'
    validation_path = os.path.join(args.row_data_path, validation_name)
    validation_dataset = load_dataset(validation_path)

    # system prompt
    prefix = '### System:\nAnswer the following multiple-choice question by giving the most appropriate response. ' \
             'Answer should be one among [A, B, C, D].\n\n'\

    few_shot_prompt = ''

    # few shot prompt
    idx_for_few_shot_prompt = random.sample(range(0, len(validation_dataset)), args.few_shot_num)
    for idx, validation_data in enumerate(validation_dataset):
        if idx in idx_for_few_shot_prompt:
            question = validation_data['question']
            options = [validation_data['opa'], validation_data['opb'], validation_data['opc'], validation_data['opd']]
            answer = ['A', 'B', 'C', 'D'][validation_data['cop']]

            formed_options = form_options(options, answer)[0]

            few_shot_prompt += '### User:\n' + question + '\n' + formed_options + '\n### Assistant:\n' + answer + '\n\n'

    few_shot_prompt = prefix + few_shot_prompt

    dataset = {}
    dataset['prompt'] = []
    dataset['question'] = []
    dataset['options'] = []
    dataset['answer'] = []
    dataset['id'] = []
    # parse
    for idx, validation_data in enumerate(tqdm.tqdm(validation_dataset)):
        if idx not in idx_for_few_shot_prompt:
            id = validation_data['id']
            question = validation_data['question']
            options = [validation_data['opa'], validation_data['opb'], validation_data['opc'], validation_data['opd']]
            answer = ['A', 'B', 'C', 'D'][validation_data['cop']]

            formed_options = form_options(options, answer)[0]

            prompt = few_shot_prompt + '### User:\n' + question + '\n' + formed_options + '\n### Assistant:\n'
            input_ids = generative_tokenizer.encode(prompt)
            if prompt.isascii() and len(input_ids) < max_input_ids_length:
                applied_qa += 1

                # for save options
                opts = ['A', 'B', 'C', 'D']
                saved_option = []
                for opt, o in zip(options, opts):
                    if opt[-1] == '.':
                        temp = opt[:-1]
                    else:
                        temp = opt
                    temp = temp.strip()
                    saved_option.append(f'{o}: {temp}')
                # print(prompt)
                # exit()
                dataset['prompt'].append(prompt)
                dataset['question'].append(question)
                dataset['options'].append(saved_option)
                dataset['answer'].append(answer)
                dataset['id'].append(id)

                # for check
                for_save_json.append({})
                for_save_json[-1]['prompt'] = prompt
                for_save_json[-1]['question'] = question
                for_save_json[-1]['options'] = saved_option
                for_save_json[-1]['answer'] = answer
                for_save_json[-1]['id'] = str(id)
        # if applied_qa == args.max_num:
        #     break
    # ------------------------------------------------------------------------------------------------------------------
    # save
    print('Applied number: ', applied_qa)
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(dataset_df)
    dataset.save_to_disk(f'{args.data_dir}/medmcqa_{model_name}')

    with open(r'../row_data/MedMCQA/medmcqa.json', 'w') as json_file:
        json.dump(for_save_json, json_file, indent=4)




