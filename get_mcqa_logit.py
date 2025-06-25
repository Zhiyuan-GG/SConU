import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
# ----------------------------------------------------------------------------------------------------------------------
# for run name
parser.add_argument('--record-dir',
                    default='../records',
                    help='save experimental records')
parser.add_argument('--generate-model',
                    default=r'D:\projects\cache_model\Qwen2.5-3B-Instruct',
                    help='local path of generative llm')
parser.add_argument('--dataset', default='medmcqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')  ##
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# run_name for saving experimental record
model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
if args.dataset in ['mmlu_pro', 'mmlu', 'medmcqa', 'medqa']:
    args.max_length_of_generation = 1
if args.sample:
    run_name = os.path.join(args.record_dir,
                            args.dataset,
                            model_name,
                            'num_generations-' + str(args.num_generations_per_prompt),
                            'temperature-' + str(args.temperature),
                            'max_len_of_generation-' + str(args.max_length_of_generation))
else:
    run_name = os.path.join(args.record_dir,
                            args.dataset,
                            model_name,
                            'num_beams-' + str(args.num_beams),
                            'max_len_of_generation-' + str(args.max_length_of_generation))
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
# cache path for hf_datasets
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# ----------------------------------------------------------------------------------------------------------------------
# load LLM, tokenizer
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
vocab_size = generative_llm.config.vocab_size
print('Vocab Size: ', vocab_size)
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

id_to_logits = dict()

for generation in tqdm.tqdm(generations):

    id = generation['id']
    prompt = generation['prompt']
    input_ids = generative_tokenizer(prompt, return_tensors="pt")['input_ids'].cuda()

    option_letters = [temp[0] for temp in generation['options']]
    choice_ids = [generative_tokenizer.convert_tokens_to_ids(c) for c in option_letters]

    # 获取模型输出
    with torch.no_grad():
        outputs = generative_llm(input_ids).logits[0, -1, :]
        choice_logits = outputs[choice_ids]
        probs = F.softmax(choice_logits, dim=0).detach().cpu()

        id_to_logits[id] = probs.tolist()
        print(id_to_logits[id])

# save
with open(f'{run_name}/mcqa_logit.pkl', 'wb') as record_file:
    pickle.dump(id_to_logits, record_file)
print('Record saved to ', f'{run_name}/mcqa_logit.pkl')

