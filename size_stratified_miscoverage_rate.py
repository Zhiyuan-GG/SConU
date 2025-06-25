import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

from collections import Counter

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import heapq
from scipy import stats


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
                    default=r'D:\projects\cache_model\vicuna-7b-v1.5',
                    help='local path of generative llm')
parser.add_argument('--dataset', default='mmlu')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')  ##
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for definition of nonconformity score
parser.add_argument('--uncertainty-metric', type=list,
                    default=['logit', 'frequency'],
                    help='for nonconformity score')
parser.add_argument('--uncertainty-weight', type=list,
                    default=[0.5, 0.5],
                    help='for nonconformity score, sum=1.0')
# for mmlu and mmlu_pro
parser.add_argument('--category', type=str, default='clinical_knowledge')
# mmlu: ['computer_security', 'high_school_computer_science', 'college_computer_science', 'machine_learning',
#        'formal_logic', 'high_school_biology', 'anatomy', 'clinical_knowledge', 'college_medicine',
#        'professional_medicine', 'college_chemistry', 'marketing', 'public_relations', 'management',
#        'business_ethics', 'professional_accounting']
# mmlu_pro: ['business', 'law', 'psychology', 'biology', 'chemistry', 'history', 'other', 'health', 'economics', 'math',
#            'physics', 'computer science', 'philosophy', 'engineering']
parser.add_argument('--split-ratio', type=float, default=0.5)
parser.add_argument('--multi-split', type=int, default=100)
# for selective conformal uncertainty
parser.add_argument('--significance-level', type=float, default=0.3, help='delta')
parser.add_argument('--alpha', type=float, default=0.3, help='upper bound of error rate')
parser.add_argument('--multiple-test', type=bool, default=False, help='multiple hypothesis testing')
parser.add_argument('--statistical-metric', type=list,
                    default=['predictive_entropy_black', 'num_semantics', 'top_2_confidence_black'],
                    help='for p-value')
# ['predictive_entropy_white', 'predictive_entropy_black', 'num_semantics', 'top_2_confidence_white', 'top_2_confidence_black']
parser.add_argument('--sampling-size', type=int, default=20, help='upper bound')
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
# random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)
# cache path for hf_datasets
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load mcqa logit
with open(f'{run_name}/mcqa_logit.pkl', 'rb') as record_file:
    mcqa_logit = pickle.load(record_file)
# ----------------------------------------------------------------------------------------------------------------------
if args.dataset in ['medmcqa', 'mmlu']:
    max_set_size = 4
elif args.dataset in ['mmlu_pro']:
    max_set_size = 10

# load data: applied_generations
if args.dataset in ['mmlu_pro', 'mmlu']:
    applied_generations = []
    for generation in tqdm.tqdm(generations):
        category = generation['category']
        if category == args.category:
            applied_generations.append(generation)
    print('Category: ', args.category)
else:
    applied_generations = generations
print('Total num: ', len(applied_generations))
# ----------------------------------------------------------------------------------------------------------------------
# for save multi-test results
miscoverage_rate_result = []
set_num_result = []
# for black-box min error rate
min_alpha_list = []
for i in range(args.multi_split):
    random.seed(i)
    # split
    calibration_set = []
    test_set = []
    num_calibration = int(len(applied_generations) * args.split_ratio)
    num_test = len(applied_generations) - num_calibration
    # random seed not fixed
    idx_for_calibration = random.sample(list(range(len(applied_generations))), num_calibration)
    idx_for_test = [idx for idx in list(range(len(applied_generations))) if idx not in idx_for_calibration]
    print(f'----------------------- Split-{i} -----------------------')
    print('Calibration data num: ', len(idx_for_calibration))  # N
    print('Test data num: ', len(idx_for_test))
    for idx, generation in enumerate(applied_generations):
        if idx in idx_for_calibration:
            calibration_set.append(generation)
        else:
            test_set.append(generation)
    # ------------------------------------------------------------------------------------------------------------------
    # minimum alpha on the calibration set (only for black-box)
    miscoverage = 0
    for calibration_data in calibration_set:
        gt_answer = calibration_data['answer']
        sampled_responses = calibration_data['sampled_generated_texts']
        if gt_answer not in sampled_responses:
            miscoverage += 1
    # L(1)
    miscoverage_rate = miscoverage / len(calibration_set)
    # N * L(1) / (N + 1)
    min_alpha = len(calibration_set) * miscoverage_rate / (len(calibration_set) + 1)
    print('Minimum alpha (black-box): ', min_alpha)
    min_alpha_list.append(min_alpha)
    # ------------------------------------------------------------------------------------------------------------------
    # nonconformity scores on calibration set
    nonconformity_score_list = []
    for calibration_data in tqdm.tqdm(calibration_set):
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts'][:args.sampling_size]
        # --------------------------------------------------------------------------------------------------------------
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_responses)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses) for option in
                                   option_letters}
        # probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(cal_logit, option_letters)}

        uncertainty_metric = {'logit': response_probability_dict,
                               'frequency': response_frequency_dict}
        # --------------------------------------------------------------------------------------------------------------
        # uncertainty score for each option: 1 - confidence
        uncertainty_dict = {
            option: 1 - args.uncertainty_weight[0] * uncertainty_metric[args.uncertainty_metric[0]][option] \
                    - args.uncertainty_weight[1] * uncertainty_metric[args.uncertainty_metric[1]][option] \
            for option in option_letters}
        # nonconformity score
        nonconformity_score = uncertainty_dict[gt_answer]
        nonconformity_score_list.append(nonconformity_score)
    # ------------------------------------------------------------------------------------------------------------------
    # q-hat
    q_level = np.ceil((len(nonconformity_score_list) + 1) * (1 - args.alpha)) / len(nonconformity_score_list)
    q_hat = np.quantile(nonconformity_score_list, q_level, method='higher')
    # ------------------------------------------------------------------------------------------------------------------
    # test set
    size_stratified_miscoverage_num_dict = {}  # number for miscoverage on various set size
    size_stratified_total_num_dict = {}  # total number for various set size
    size_stratified_miscoverage_rate_dict = {}
    for test_data in tqdm.tqdm(test_set):
        test_id = test_data['id']
        test_logit = mcqa_logit[test_id]

        gt_answer = test_data['answer']
        sampled_responses = test_data['sampled_generated_texts'][:args.sampling_size]
        option_letters = [temp[0] for temp in test_data['options']]
        # --------------------------------------------------------------------------------------------------------------
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_responses)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses) for
                                   option in option_letters}
        # probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(test_logit, option_letters)}
        uncertainty_metric = {'logit': response_probability_dict,
                              'frequency': response_frequency_dict}
        # --------------------------------------------------------------------------------------------------------------
        # uncertainty score for each option: 1 - confidence
        uncertainty_dict = {
            option: 1 - args.uncertainty_weight[0] * uncertainty_metric[args.uncertainty_metric[0]][option] \
                      - args.uncertainty_weight[1] * uncertainty_metric[args.uncertainty_metric[1]][option] \
            for option in option_letters}
        # prediction set
        prediction_set = []
        for option in option_letters:
            if uncertainty_dict[option] <= q_hat:
                prediction_set.append(option)
        # set size
        set_size = len(prediction_set)
        # count total num
        if set_size not in size_stratified_total_num_dict:
            size_stratified_total_num_dict[set_size] = 1
        else:
            size_stratified_total_num_dict[set_size] += 1
        # count miscoverage num
        if gt_answer not in prediction_set:
            if set_size not in size_stratified_miscoverage_num_dict:
                size_stratified_miscoverage_num_dict[set_size] = 1
            else:
                size_stratified_miscoverage_num_dict[set_size] += 1
    for set_size in size_stratified_total_num_dict:
        if size_stratified_total_num_dict[set_size] != 0 and set_size in size_stratified_miscoverage_num_dict:
            size_stratified_miscoverage_rate_dict[set_size] = size_stratified_miscoverage_num_dict[set_size] / size_stratified_total_num_dict[set_size]
        else:
            size_stratified_miscoverage_rate_dict[set_size] = 0
    for set_size in range(max_set_size + 1):
        if set_size not in size_stratified_miscoverage_rate_dict:
            assert set_size not in size_stratified_total_num_dict
            size_stratified_miscoverage_rate_dict[set_size] = 0
            size_stratified_total_num_dict[set_size] = 0
    miscoverage_rate_result.append(size_stratified_miscoverage_rate_dict)
    set_num_result.append(size_stratified_total_num_dict)

# numpy
miscoverage_rate_numpy = np.zeros(shape=(max_set_size + 1, args.multi_split))  # (set size, epoch)
for i, size_stratified_miscoverage_rate_dict in enumerate(miscoverage_rate_result):
    # i epoch 0-99
    for set_size in size_stratified_miscoverage_rate_dict:
        # set_size 0-9/3
        miscoverage_rate_numpy[set_size][i] = size_stratified_miscoverage_rate_dict[set_size]

print(max(min_alpha_list))
mean = np.mean(miscoverage_rate_numpy, axis=1)
std = np.std(miscoverage_rate_numpy, axis=1)
print(mean)
print(std)

set_size_list = list(range(max_set_size + 1))  # 1, 2, 3, 4, ...
print(set_size_list)

# 折线图
plt.figure(figsize=(8, 6))
plt.axhline(y=args.alpha, color='r', linestyle='--', label='Upper Bound')
plt.plot(set_size_list, mean, '-', color='darkgreen', marker='o', label='Size-stratified EMR')
plt.fill_between(set_size_list, np.maximum(mean - std, 0), np.minimum(mean + std, 1.0), color='darkgreen', alpha=0.2)

plt.xlabel('Set Size', fontsize=15)
plt.ylabel('Empirical Miscoverage Rate (EMR)', fontsize=15)
plt.xticks(set_size_list, fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

# 箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=miscoverage_rate_numpy.T)  # 转置数据，使每列表示一个size
plt.axhline(y=args.alpha, color='r', linestyle='--', label='Upper Bound')
plt.xlabel('Set Size')
plt.ylabel('Empirical Miscoverage Rate (EMR)')
plt.xticks(set_size_list)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# for save multi-test results
miscoverage_rate_result = []
set_num_result = []
for i in range(args.multi_split):
    random.seed(i)
    # split
    calibration_set = []
    test_set = []
    num_calibration = int(len(applied_generations) * args.split_ratio)
    num_test = len(applied_generations) - num_calibration
    # random seed not fixed
    idx_for_calibration = random.sample(list(range(len(applied_generations))), num_calibration)
    idx_for_test = [idx for idx in list(range(len(applied_generations))) if idx not in idx_for_calibration]
    print(f'----------------------- Split-{i} -----------------------')
    print('Calibration data num: ', len(idx_for_calibration))  # N
    print('Test data num: ', len(idx_for_test))
    for idx, generation in enumerate(applied_generations):
        if idx in idx_for_calibration:
            calibration_set.append(generation)
        else:
            test_set.append(generation)
    # ------------------------------------------------------------------------------------------------------------------
    # nonconformity scores on calibration set
    nonconformity_score_list = []
    # statistical scores for p-value
    statistical_score_list = []
    for calibration_data in tqdm.tqdm(calibration_set):
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts'][:args.sampling_size]
        # --------------------------------------------------------------------------------------------------------------
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_responses)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses) for option in
                                   option_letters}
        # probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(cal_logit, option_letters)}

        uncertainty_metric = {'logit': response_probability_dict,
                               'frequency': response_frequency_dict}
        # --------------------------------------------------------------------------------------------------------------
        # uncertainty score for each option: 1 - confidence
        uncertainty_dict = {
            option: 1 - args.uncertainty_weight[0] * uncertainty_metric[args.uncertainty_metric[0]][option] \
                    - args.uncertainty_weight[1] * uncertainty_metric[args.uncertainty_metric[1]][option] \
            for option in option_letters}
        # nonconformity score
        nonconformity_score = uncertainty_dict[gt_answer]
        nonconformity_score_list.append(nonconformity_score)

        # statistical metrics
        # 1. number of semantic cluster (black-box)
        num_semantics = sum(1 for option in option_letters if response_frequency_dict[option] > 0)
        # 2. predictive entropy w/b
        probabilities_white = np.array(list(response_probability_dict.values()))
        predictive_entropy_white = -np.sum(
            probabilities_white[probabilities_white > 0] * np.log(probabilities_white[probabilities_white > 0]))
        probabilities_black = np.array(list(response_frequency_dict.values()))
        predictive_entropy_black = -np.sum(
            probabilities_black[probabilities_black > 0] * np.log(probabilities_black[probabilities_black > 0]))
        # 3. top-2 option confidence w/b
        top_2_confidence_white = sorted(list(response_probability_dict.values()), reverse=True)[1]
        top_2_confidence_black = sorted(list(response_frequency_dict.values()), reverse=True)[1]
        statistical_score_list.append({'num_semantics': num_semantics,
                                       'predictive_entropy_white': predictive_entropy_white,
                                       'predictive_entropy_black': predictive_entropy_black,
                                       'top_2_confidence_white': top_2_confidence_white,
                                       'top_2_confidence_black': top_2_confidence_black})
    # ------------------------------------------------------------------------------------------------------------------
    # test calibration set for calibration data (exchangeability within calibration set)
    for_calibration_data_quality = []
    for idx, calibration_data in enumerate(calibration_set):
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts'][:args.sampling_size]
        # --------------------------------------------------------------------------------------------------------------
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_responses)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses) for option in
                                   option_letters}
        # probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(cal_logit, option_letters)}

        uncertainty_metric = {'logit': response_probability_dict,
                              'frequency': response_frequency_dict}
        # --------------------------------------------------------------------------------------------------------------
        # uncertainty score for each option: 1 - confidence
        uncertainty_dict = {
            option: 1 - args.uncertainty_weight[0] * uncertainty_metric[args.uncertainty_metric[0]][option] \
                    - args.uncertainty_weight[1] * uncertainty_metric[args.uncertainty_metric[1]][option] \
            for option in option_letters}
        if idx == 0:
            temp_nonconformity_score_list = nonconformity_score_list[idx + 1:]
        elif idx == len(nonconformity_score_list) - 1:
            temp_nonconformity_score_list = nonconformity_score_list[:idx]
        else:
            temp_nonconformity_score_list = nonconformity_score_list[:idx] + nonconformity_score_list[idx + 1:]
        assert len(temp_nonconformity_score_list) == len(nonconformity_score_list) - 1

        temp_q_level = np.ceil((len(temp_nonconformity_score_list) + 1) * (1 - args.alpha)) / len(
            temp_nonconformity_score_list)
        temp_q_hat = np.quantile(temp_nonconformity_score_list, temp_q_level, method='higher')
        for_calibration_data_quality.append(uncertainty_dict[gt_answer] <= temp_q_hat)
    # ------------------------------------------------------------------------------------------------------------------
    N = len(statistical_score_list)
    # q-hat
    q_level = np.ceil((len(nonconformity_score_list) + 1) * (1 - args.alpha)) / len(nonconformity_score_list)
    q_hat = np.quantile(nonconformity_score_list, q_level, method='higher')
    # ------------------------------------------------------------------------------------------------------------------
    # test set
    size_stratified_miscoverage_num_dict = {}  # number for miscoverage on various set size
    size_stratified_total_num_dict = {}  # total number for various set size
    size_stratified_miscoverage_rate_dict = {}
    for test_data in tqdm.tqdm(test_set):
        test_id = test_data['id']
        test_logit = mcqa_logit[test_id]

        gt_answer = test_data['answer']
        sampled_responses = test_data['sampled_generated_texts'][:args.sampling_size]
        option_letters = [temp[0] for temp in test_data['options']]
        # --------------------------------------------------------------------------------------------------------------
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_responses)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses) for
                                   option in option_letters}
        # probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(test_logit, option_letters)}
        uncertainty_metric = {'logit': response_probability_dict,
                              'frequency': response_frequency_dict}
        # --------------------------------------------------------------------------------------------------------------
        # uncertainty score for each option: 1 - confidence
        uncertainty_dict = {
            option: 1 - args.uncertainty_weight[0] * uncertainty_metric[args.uncertainty_metric[0]][option] \
                      - args.uncertainty_weight[1] * uncertainty_metric[args.uncertainty_metric[1]][option] \
            for option in option_letters}
        # ------------------------------------------------------------------------------------------------------
        # statistical metrics
        # 1. number of semantic cluster (black-box)
        num_semantics = sum(1 for option in option_letters if response_frequency_dict[option] > 0)
        # 2. predictive entropy w/b
        probabilities_white = np.array(list(response_probability_dict.values()))
        predictive_entropy_white = -np.sum(
            probabilities_white[probabilities_white > 0] * np.log(probabilities_white[probabilities_white > 0]))
        probabilities_black = np.array(list(response_frequency_dict.values()))
        predictive_entropy_black = -np.sum(
            probabilities_black[probabilities_black > 0] * np.log(probabilities_black[probabilities_black > 0]))
        # 3. top-2 option confidence w/b
        top_2_confidence_white = sorted(list(response_probability_dict.values()), reverse=True)[1]
        top_2_confidence_black = sorted(list(response_frequency_dict.values()), reverse=True)[1]

        statistical_metric_dict = {'num_semantics': num_semantics,
                                   'predictive_entropy_white': predictive_entropy_white,
                                   'predictive_entropy_black': predictive_entropy_black,
                                   'top_2_confidence_white': top_2_confidence_white,
                                   'top_2_confidence_black': top_2_confidence_black}
        # ------------------------------------------------------------------------------------------------------
        # p-value
        if not args.multiple_test:
            test_statistical_score = statistical_metric_dict[args.statistical_metric[0]]
            calibration_statistical_score_list = [temp[args.statistical_metric[0]] for temp in
                                                  statistical_score_list]
            p_value = (sum(
                1 for cal_idx, temp in enumerate(calibration_statistical_score_list)
                if temp >= test_statistical_score and for_calibration_data_quality[cal_idx]) + 1) / (
                              N + 1)
        else:
            p_value_list = []
            for metric in args.statistical_metric:
                test_statistical_score = statistical_metric_dict[metric]
                calibration_statistical_score_list = [temp[metric] for temp in statistical_score_list]
                p_value_list.append((sum(
                    1 for cal_idx, temp in enumerate(calibration_statistical_score_list)
                    if temp >= test_statistical_score and for_calibration_data_quality[cal_idx]) + 1) / (
                                            N + 1))
            p_value = min(p_value_list)
        # ------------------------------------------------------------------------------------------------------
        # selective prediction
        if p_value > args.significance_level:
            # prediction set
            prediction_set = []
            for option in option_letters:
                if uncertainty_dict[option] <= q_hat:
                    prediction_set.append(option)
            # set size
            set_size = len(prediction_set)
            # count total num
            if set_size not in size_stratified_total_num_dict:
                size_stratified_total_num_dict[set_size] = 1
            else:
                size_stratified_total_num_dict[set_size] += 1
            # count miscoverage num
            if gt_answer not in prediction_set:
                if set_size not in size_stratified_miscoverage_num_dict:
                    size_stratified_miscoverage_num_dict[set_size] = 1
                else:
                    size_stratified_miscoverage_num_dict[set_size] += 1
    for set_size in size_stratified_total_num_dict:
        if size_stratified_total_num_dict[set_size] != 0 and set_size in size_stratified_miscoverage_num_dict:
            size_stratified_miscoverage_rate_dict[set_size] = size_stratified_miscoverage_num_dict[set_size] / size_stratified_total_num_dict[set_size]
        else:
            size_stratified_miscoverage_rate_dict[set_size] = 0
    for set_size in range(max_set_size + 1):
        if set_size not in size_stratified_miscoverage_rate_dict:
            assert set_size not in size_stratified_total_num_dict
            size_stratified_miscoverage_rate_dict[set_size] = 0
            size_stratified_total_num_dict[set_size] = 0
    miscoverage_rate_result.append(size_stratified_miscoverage_rate_dict)
    set_num_result.append(size_stratified_total_num_dict)

# numpy
miscoverage_rate_numpy = np.zeros(shape=(max_set_size + 1, args.multi_split))  # (set size, epoch)
for i, size_stratified_miscoverage_rate_dict in enumerate(miscoverage_rate_result):
    # i epoch 0-99
    for set_size in size_stratified_miscoverage_rate_dict:
        # set_size 0-9/3
        miscoverage_rate_numpy[set_size][i] = size_stratified_miscoverage_rate_dict[set_size]

mean = np.mean(miscoverage_rate_numpy, axis=1)
std = np.std(miscoverage_rate_numpy, axis=1)
print(mean)
print(std)

set_size_list = list(range(max_set_size + 1))  # 1, 2, 3, 4, ...
print(set_size_list)

# 折线图
plt.figure(figsize=(8, 6))
plt.axhline(y=args.alpha, color='r', linestyle='--', label='Upper Bound')
plt.plot(set_size_list, mean, '-', color='darkgreen', marker='o', label='Size-stratified EMR')
plt.fill_between(set_size_list, np.maximum(mean - std, 0), np.minimum(mean + std, 1.0), color='darkgreen', alpha=0.2)

plt.xlabel('Set Size', fontsize=15)
plt.ylabel('Empirical Miscoverage Rate (EMR)', fontsize=15)
plt.xticks(set_size_list, fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

# 箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=miscoverage_rate_numpy.T)  # 转置数据，使每列表示一个size
plt.axhline(y=args.alpha, color='r', linestyle='--', label='Upper Bound')
plt.xlabel('Set Size')
plt.ylabel('Empirical Miscoverage Rate (EMR)')
plt.xticks(set_size_list)
plt.show()
