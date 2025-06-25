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
if args.dataset in ['mmlu_pro', 'mmlu', 'medmcqa']:
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
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

# special tokens
eos_token_id = generative_tokenizer('. ')['input_ids'][1]
bad_words = ['### User:', '### Assistant:', '### System:', 'User:', 'Assistant:', 'System:', '###', '\n']
bad_words_ids = [generative_tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]

# options
if args.dataset in ['mmlu_pro']:
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
elif args.dataset in ['mmlu', 'medmcqa']:
    options = ['A', 'B', 'C', 'D']
elif args.dataset in ['medqa']:
    options = ['A', 'B', 'C', 'D', 'E']

# clean
results = []
for generation in tqdm.tqdm(generations):
    sampled_generations = generation['sampled_generated_texts']
    cleaned_generations = []
    error_generation = 0
    for sample_generation in sampled_generations:
        if sample_generation[0] in options:
            cleaned_generations.append(sample_generation[0])
        else:
            print('Error: ', [sample_generation])
            error_generation += 1

    if error_generation >= (0.6*args.num_generations_per_prompt):
            continue
    # re-generate
    if error_generation != 0:
        print(f"{error_generation} error generations.")
        # encode
        prompt = generation['prompt']
        encode = generative_tokenizer(prompt)
        input_ids = torch.LongTensor(encode['input_ids']).cuda()
        attention_mask = torch.LongTensor(encode['attention_mask']).cuda()
        input_length = len(input_ids)
        input_ids = torch.reshape(input_ids, (-1, input_length))
        attention_mask = torch.reshape(attention_mask, (-1, input_length))

        while True:
            gen = generative_llm.generate(input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=True,
                                          num_return_sequences=1,  # <= num_beams
                                          num_beams=1,  # greedy search
                                          max_length=input_length + args.max_length_of_generation,
                                          eos_token_id=eos_token_id,
                                          bad_words_ids=bad_words_ids,
                                          temperature=args.temperature,
                                          top_p=args.top_p)
            gen_text = generative_tokenizer.decode(gen[0][input_length:], skip_special_tokens=True)
            if gen_text[0] in options:
                cleaned_generations.append(gen_text[0])
                error_generation -= 1
                print(f"{error_generation} error generations.")

            if error_generation == 0:
                break
    # re-check
    assert len(cleaned_generations) == args.num_generations_per_prompt
    for cleaned_generation in cleaned_generations:
        assert cleaned_generation in options

    generation['sampled_generated_texts'] = cleaned_generations
    results.append(generation)

# save
with open(f'{run_name}/cleaned_generations.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/cleaned_generations.pkl')
