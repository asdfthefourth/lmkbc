import itertools
import json
import os

import torch


def permut():
    # Your list of numbers
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Calculate all permutations of length 3
    permutations = itertools.permutations(numbers, 3)
    # Print each permutation
    for permutation in permutations:
        print(permutation)


def test():
    from itertools import combinations, chain

    # list to find permutations of
    my_list = [0,1,2,3,4,5,6,7,8,9,10]

    # generate all permutations with length from 0 to n
    all_combinations = chain(*[combinations(my_list, i) for i in range(4)])


    # convert each combination to a list and print it
    for combination in all_combinations:
        print(list(combination))


def cmp_f1_scores():
    consens = "data/results/ensemble_size_11_val_1/summary/consens.json"
    ensemble = "data/results/ensemble_size_11_val_1/summary/ensemble.json"

    with open(consens) as file:
        consens_dict = json.load(file)
    with open(ensemble) as file:
        ensemble_dict = json.load(file)

    f1 = 0
    for key in consens_dict.keys():
        con_f1 = consens_dict[key][0]
        ens_f1 = ensemble_dict[key][0]
        if ens_f1 > con_f1:
            f1 += ens_f1
        else:
            f1 += con_f1
    print(f1 / 21)

def combine_set():
    consens = "data/results/ensemble_perfomance_size_11_train_1/summary/exclusion.json"
    result = []
    with open(consens) as file:
        consens_dict = json.load(file)
    for key in consens_dict.keys():
        if key == 'f1':
            continue
        with open(consens_dict[key]['file']) as file:
            for line in file:
                result.append(json.loads(line))
    with open("data/results/ensemble_perfomance_size_11_train_1/merged_optimal.jsonl", 'w') as file:
        for line in result:
            file.write(json.dumps(line) + '\n')

def combine_new_set():
    input_dir = "data/results/eval_big_few_shot_10_test/"
    files = os.listdir(input_dir)
    res = []

    for file in files:
        if 'transfer' in file:
            with open(input_dir + file) as file:
                res.extend([json.loads(line) for line in file])

    with open(f"{input_dir}predictions.jsonl", 'w') as file:
        for line in res:
            file.write(json.dumps(line) + '\n')

combine_new_set()
