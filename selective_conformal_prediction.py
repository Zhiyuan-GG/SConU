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
from matplotlib.colors import LinearSegmentedColormap

import heapq

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
                    default=r'D:\projects\cache_model\Llama-3.1-8B-Instruct',
                    help='local path of generative llm')
parser.add_argument('--dataset', default='mmlu_pro')  # 0.2673
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')  ##
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=50, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for definition of nonconformity score
parser.add_argument('--uncertainty-metric', type=list,
                    default=['logit', 'frequency'],
                    help='for nonconformity score')
parser.add_argument('--uncertainty-weight', type=list,
                    default=[0.0, 1.0],
                    help='for nonconformity score, sum=1.0')
# for selective conformal prediction
parser.add_argument('--error-rate', type=float, default=0.28, help='alpha')
parser.add_argument('--significance-level', type=float, default=0.28, help='for outlier detection')
parser.add_argument('--multiple-test', type=bool, default=False, help='multiple hypothesis testing')
parser.add_argument('--statistical-metric', type=list,
                    default=['predictive_entropy_black', 'num_semantics',
                             'top_2_confidence_black'],
                    help='for p-value')
# ['predictive_entropy_white', 'predictive_entropy_black', 'num_semantics', 'top_2_confidence_white', 'top_2_confidence_black']
parser.add_argument('--filter', type=bool, default=False, help='for filtering outliers')
parser.add_argument('--filter-tolerance', type=float, default=0.5, help='filtering threshold')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# run_name for saving experimental record
model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
if args.dataset in ['mmlu_pro', 'mmlu']:
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
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load mcqa logit
with open(f'{run_name}/mcqa_logit.pkl', 'rb') as record_file:
    mcqa_logit = pickle.load(record_file)
# ----------------------------------------------------------------------------------------------------------------------
# applied generations
if args.filter:
    filtered_samples = 0
    applied_generations = []
    for generation in tqdm.tqdm(generations):
        gt_answer = generation['answer']
        sampled_generated_texts = generation['sampled_generated_texts']
        response_counter_dict = Counter(sampled_generated_texts)
        # filter hallucinations
        if (response_counter_dict[gt_answer] == 0) and (max(list(response_counter_dict.values())) >= (
                1 - args.filter_tolerance) * args.num_generations_per_prompt):
            filtered_samples += 1
        else:
            applied_generations.append(generation)
    print('Filtered samples: ', filtered_samples)
else:
    applied_generations = generations
# ----------------------------------------------------------------------------------------------------------------------
# applied category number
category_num_dict = {}
for generation in tqdm.tqdm(applied_generations):
    category = generation['category']
    # new category
    if category not in category_num_dict:
        category_num_dict[category] = 1
    # add
    else:
        category_num_dict[category] += 1
# plot data distribution of different categories
plt.figure(figsize=(12, 6))
bars = plt.barh(list(category_num_dict.keys()), list(category_num_dict.values()), color='steelblue')
plt.xlabel('Counts')
plt.grid(axis='x')
for bar in bars:
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             str(bar.get_width()), va='center')
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# category list
category_list = list(category_num_dict.keys())
#  applied generations for each category
category_to_generations = {category: [] for category in category_list}
for generation in tqdm.tqdm(applied_generations):
    category_to_generations[generation['category']].append(generation)
# check number
for category in category_list:
    assert category_num_dict[category] == len(category_to_generations[category])
# ----------------------------------------------------------------------------------------------------------------------
#################
# minimum alpha #
#################
min_alpha_dict = {}
for category in category_list:
    # calibration set
    calibration_set = category_to_generations[category]
    miscoverage = 0
    for calibration_data in calibration_set:
        gt_answer = calibration_data['answer']
        sampled_responses = calibration_data['sampled_generated_texts']
        if gt_answer not in sampled_responses:
            miscoverage += 1
    # L(1)
    miscoverage_rate = miscoverage / category_num_dict[category]
    # N * L(1) / (N + 1)
    min_alpha_dict[category] = category_num_dict[category] * miscoverage_rate / (category_num_dict[category] + 1)
print('Minimum alpha: ', max(list(min_alpha_dict.values())))
# plot
plt.figure(figsize=(12, 6))
bars = plt.barh(list(min_alpha_dict.keys()), list(min_alpha_dict.values()), color='steelblue')
plt.xlabel(r'Minimum $\alpha$')
plt.grid(axis='x')
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center')
plt.tight_layout()
plt.show()
# exit()
# ----------------------------------------------------------------------------------------------------------------------
###############################
# distribution shift coverage #
###############################
# results of empirical miscoverage rate
emr_results = {}
# results of average set size
apss_results = {}
for calibration_category in category_list:
    # calibration set
    calibration_set = category_to_generations[calibration_category]
    # calculate nonconformity scores
    nonconformity_score_list = []
    for calibration_data in calibration_set:
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts']
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
    # q-hat
    q_level = np.ceil((len(nonconformity_score_list) + 1) * (1 - args.error_rate)) / len(nonconformity_score_list)
    q_hat = np.quantile(nonconformity_score_list, q_level, method='higher')

    # for test set of different categories
    miscoverage_rate_dict = {c: 0 for c in category_list}
    average_set_size_dict = {c: 0 for c in category_list}
    for test_category in category_list:
        if test_category == calibration_category:
            emr = args.error_rate
            apss = 1
        else:
            miscoverage_num = 0
            set_size_list = []
            test_set = category_to_generations[test_category]
            for test_data in test_set:
                test_id = test_data['id']
                test_logit = mcqa_logit[test_id]

                gt_answer = test_data['answer']
                sampled_responses = test_data['sampled_generated_texts']
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
                set_size_list.append(set_size)
                # miscoverage
                if gt_answer not in prediction_set:
                    miscoverage_num += 1
            # miscoverage rate
            emr = miscoverage_num / len(test_set)
            # average set size
            apss = sum(set_size_list) / len(set_size_list)
        miscoverage_rate_dict[test_category] = emr
        average_set_size_dict[test_category] = apss
    emr_results[calibration_category] = miscoverage_rate_dict
    apss_results[calibration_category] = average_set_size_dict
# 将数据转换为 DataFrame
df = pd.DataFrame(emr_results)

# 自定义颜色映射
colors = [
    (85/255, 130/255, 150/255),  # 蓝色 (R:85, G:130, B:150)
    (242/255, 240/255, 239/255),  # 灰色/白色 (R:242, G:240, B:239)
    (182/255, 94/255, 68/255),    # 红色 (R:182, G:94, B:68)
]
cmap = LinearSegmentedColormap.from_list("blue_gray_red", colors)

# Blues, PuBu, vlag
# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14},center=args.error_rate)
# plt.title(rf'Original EMR ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('Original_EMR.pdf', format='pdf', bbox_inches='tight')
plt.show()
# exit()

df = pd.DataFrame(apss_results)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='vlag_r', fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14}, center=1)
# plt.title(rf'Original APSS ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('Original_APSS.pdf', format='pdf', bbox_inches='tight')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
############################
#   selective prediction   #
############################
# results of empirical miscoverage rate
emr_results = {}
# results of average set size
apss_results = {}
# selective test data
selected_test_data_num = {}
for calibration_category in category_list:
    # calibration set
    calibration_set = category_to_generations[calibration_category]
    # for save nonconformity scores
    nonconformity_score_list = []
    # for save statistical metrics
    statistical_score_list = []
    for calibration_data in calibration_set:
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts']
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

    N = len(statistical_score_list)
    # q-hat
    q_level = np.ceil((N + 1) * (1 - args.error_rate)) / N
    q_hat = np.quantile(nonconformity_score_list, q_level, method='higher')

    # for test set of different categories
    miscoverage_rate_dict = {c: 0 for c in category_list}
    average_set_size_dict = {c: 0 for c in category_list}
    test_data_num = {c: 0 for c in category_list}
    for test_category in category_list:
        if test_category == calibration_category:
            emr = args.error_rate
            apss = 1
            selected_sample = category_num_dict[test_category]
        else:
            # selected to prediction
            selected_sample = 0
            # miscoverage number in selected samples
            miscoverage_num = 0
            set_size_list = []
            test_set = category_to_generations[test_category]
            for test_data in test_set:
                test_id = test_data['id']
                test_logit = mcqa_logit[test_id]

                gt_answer = test_data['answer']
                sampled_responses = test_data['sampled_generated_texts']
                option_letters = [temp[0] for temp in test_data['options']]
                # ----------------------------------------------------------------------------------------------------------
                # for frequency (black-box)
                response_counter_dict = Counter(sampled_responses)
                response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses)
                                           for option in option_letters}
                # probability (white-box)
                response_probability_dict = {option: logit for logit, option in zip(test_logit, option_letters)}
                uncertainty_metric = {'logit': response_probability_dict,
                                      'frequency': response_frequency_dict}
                # ----------------------------------------------------------------------------------------------------------
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
                        1 for temp in calibration_statistical_score_list if temp >= test_statistical_score) + 1) / (
                                      N + 1)
                else:
                    p_value_list = []
                    for metric in args.statistical_metric:
                        test_statistical_score = statistical_metric_dict[metric]
                        calibration_statistical_score_list = [temp[metric] for temp in statistical_score_list]
                        p_value_list.append((sum(
                            1 for temp in calibration_statistical_score_list if temp >= test_statistical_score) + 1) / (
                                                        N + 1))
                    p_value = min(p_value_list)
                # ------------------------------------------------------------------------------------------------------
                # selective prediction
                if p_value > args.significance_level:
                    selected_sample += 1
                    # prediction set
                    prediction_set = []
                    for option in option_letters:
                        if uncertainty_dict[option] <= q_hat:
                            prediction_set.append(option)
                    # set size
                    set_size = len(prediction_set)
                    set_size_list.append(set_size)
                    # miscoverage
                    if gt_answer not in prediction_set:
                        miscoverage_num += 1
            if selected_sample == 0:
                emr = args.error_rate
                apss = 1
            else:
                # miscoverage rate
                emr = miscoverage_num / selected_sample
                # average set size
                apss = sum(set_size_list) / len(set_size_list)
        # selected_sample
        test_data_num[test_category] = selected_sample
        miscoverage_rate_dict[test_category] = emr
        average_set_size_dict[test_category] = apss
    emr_results[calibration_category] = miscoverage_rate_dict
    apss_results[calibration_category] = average_set_size_dict
    selected_test_data_num[calibration_category] = test_data_num
# 将数据转换为 DataFrame
df = pd.DataFrame(emr_results)

# Blues, PuBu, vlag
# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14})
# plt.title(rf'Calibrated EMR ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('Baseline_EMR.pdf', format='pdf', bbox_inches='tight')
plt.show()

df = pd.DataFrame(apss_results)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='vlag_r', fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14}, center=1.0)
# plt.title(rf'Calibrated APSS ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('Baseline_APSS.pdf', format='pdf', bbox_inches='tight')
plt.show()
# exit()
df = pd.DataFrame(selected_test_data_num)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='vlag_r', fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14})
plt.title(rf'Selected test data number ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=0, fontsize=14)  # test set
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
###################################
#   conformal outlier detection   #
###################################
# results of empirical miscoverage rate
emr_results = {}
# results of average set size
apss_results = {}
# selective test data
selected_test_data_num = {}
for calibration_category in category_list:
    # calibration set
    calibration_set = category_to_generations[calibration_category]
    # for save nonconformity scores
    nonconformity_score_list = []
    # for save statistical metrics
    statistical_score_list = []
    for calibration_data in calibration_set:
        cal_id = calibration_data['id']
        cal_logit = mcqa_logit[cal_id]

        gt_answer = calibration_data['answer']
        option_letters = [temp[0] for temp in calibration_data['options']]
        sampled_responses = calibration_data['sampled_generated_texts']
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
        sampled_responses = calibration_data['sampled_generated_texts']
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

        temp_q_level = np.ceil((len(temp_nonconformity_score_list) + 1) * (1 - args.error_rate)) / len(
            temp_nonconformity_score_list)
        temp_q_hat = np.quantile(temp_nonconformity_score_list, temp_q_level, method='higher')
        for_calibration_data_quality.append(uncertainty_dict[gt_answer] <= temp_q_hat)
    # ------------------------------------------------------------------------------------------------------------------
    N = len(statistical_score_list)
    # q-hat
    q_level = np.ceil((N + 1) * (1 - args.error_rate)) / N
    q_hat = np.quantile(nonconformity_score_list, q_level, method='higher')

    # for test set of different categories
    miscoverage_rate_dict = {c: 0 for c in category_list}
    average_set_size_dict = {c: 0 for c in category_list}
    test_data_num = {c: 0 for c in category_list}
    for test_category in category_list:
        if test_category == calibration_category:
            emr = args.error_rate
            apss = 1
            selected_sample = category_num_dict[test_category]
        else:
            # selected to prediction
            selected_sample = 0
            # miscoverage number in selected samples
            miscoverage_num = 0
            set_size_list = []
            test_set = category_to_generations[test_category]
            for test_data in test_set:
                test_id = test_data['id']
                test_logit = mcqa_logit[test_id]

                gt_answer = test_data['answer']
                sampled_responses = test_data['sampled_generated_texts']
                option_letters = [temp[0] for temp in test_data['options']]
                # ----------------------------------------------------------------------------------------------------------
                # for frequency (black-box)
                response_counter_dict = Counter(sampled_responses)
                response_frequency_dict = {option: response_counter_dict[option] / len(sampled_responses)
                                           for option in option_letters}
                # probability (white-box)
                response_probability_dict = {option: logit for logit, option in zip(test_logit, option_letters)}
                uncertainty_metric = {'logit': response_probability_dict,
                                      'frequency': response_frequency_dict}
                # ----------------------------------------------------------------------------------------------------------
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
                    selected_sample += 1
                    # prediction set
                    prediction_set = []
                    for option in option_letters:
                        if uncertainty_dict[option] <= q_hat:
                            prediction_set.append(option)
                    # set size
                    set_size = len(prediction_set)
                    set_size_list.append(set_size)
                    # miscoverage
                    if gt_answer not in prediction_set:
                        miscoverage_num += 1
            if selected_sample == 0:
                emr = args.error_rate
                apss = 1
            else:
                # miscoverage rate
                emr = miscoverage_num / selected_sample
                # average set size
                apss = sum(set_size_list) / len(set_size_list)
        # selected_sample
        test_data_num[test_category] = selected_sample
        miscoverage_rate_dict[test_category] = emr
        average_set_size_dict[test_category] = apss
    emr_results[calibration_category] = miscoverage_rate_dict
    apss_results[calibration_category] = average_set_size_dict
    selected_test_data_num[calibration_category] = test_data_num
# 将数据转换为 DataFrame
df = pd.DataFrame(emr_results)

# Blues, PuBu, vlag
# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14})
# plt.title(rf'Calibrated EMR by SConU ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('SConU_EMR.pdf', format='pdf', bbox_inches='tight')
plt.show()

df = pd.DataFrame(apss_results)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='vlag_r', fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14}, center=1.0)
# plt.title(rf'Calibrated APSS by SConU  ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=25, fontsize=14)  # test set
plt.savefig('SConU_APSS.pdf', format='pdf', bbox_inches='tight')
plt.show()

df = pd.DataFrame(selected_test_data_num)

# 创建热力图
plt.figure(figsize=(16, 10))
sns.heatmap(df, annot=True, cmap='vlag_r', fmt=".2f", cbar=True, square=False, linewidths=0.0, annot_kws={"size": 14})
plt.title(rf'Selected test data number by SConU  ($\alpha$={args.error_rate})', fontsize=16)
# 设置标签旋转
plt.xticks(rotation=20, ha='right', fontsize=14)  # calibration set
plt.yticks(rotation=0, fontsize=14)  # test set
plt.show()