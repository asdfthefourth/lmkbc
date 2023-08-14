import json
from collections import Counter

from src.evaluate import split_ensemble_files_by_relation


def convert_to_dict(record):
    return {int(item['id']): {'unformat': item['unformat'], 'missing': item['missing'],
                              'double': item['double'], 'percentage': item['percentage']} for item in record}


def compare_analysis():
    file1 = "data/results/eval_small_few_10_val"
    file2 = "data/results/eval_small_few_10_val_1"

    with open(file1 + "/summary/analyze.json") as file:
        dict1 = json.load(file)
    with open(file2 + "/summary/analyze.json") as file:
        dict2 = json.load(file)

    result = {}
    for key in dict1:
        if key in dict2:
            converted1 = convert_to_dict(dict1[key])
            converted2 = convert_to_dict(dict2[key])
            result[key] = []
            for id in converted1:
                if id in converted2:
                    score_diff = {
                        'id': id,
                        'unformat_diff': converted1[id]['unformat'] - converted2[id]['unformat'],
                        'missing_diff': converted1[id]['missing'] - converted2[id]['missing'],
                        'double_diff': converted1[id]['double'] - converted2[id]['double'],
                        'percentage_diff': converted1[id]['percentage'] - converted2[id]['percentage']
                    }
                    result[key].append(score_diff)

    with open('data/results/ad.json', 'w') as file:
        json.dump(result, file)


def generate_analysis():
    dataset_dir = "data/results/eval_big_few_shot_10_train"
    rel_dict = split_ensemble_files_by_relation(dataset_dir)
    res_dict = {}
    for relation in rel_dict.keys():
        res_dict[relation] = []
        for file in rel_dict[relation]:
            u, m, d, p, e = analyze_dataset(file)
            prompt_id = file.split('.jsonl')[0].split('_')[-1]
            res_dict[relation].append({'id': prompt_id, 'unformat': u, 'missing': m, 'double': d, 'percentage': p, 'empty': e})
    with open(dataset_dir + "/summary/analyze.json", 'w') as file:
        json.dump(res_dict, file)


def analyze_dataset(file):
    unformatted_object_ids = 0.
    missing_answer_part = 0.
    double_object_ids = 0.
    highest_answer_percentage = 0.
    false_empty = 0.
    with open(file) as fp:
        df = [json.loads(line) for line in fp]

    relation = file.split('/')[-1].split('_')[1]
    prompt_id = file.split('.jsonl')[0].split('_')[-1]

    for idx, line in enumerate(df):
        new_u = count_unformatted_object_ids(line['ObjectEntitiesID'])
        unformatted_object_ids += new_u
        new_m = is_answer_missing_part(line['OGAnswer'])
        missing_answer_part += new_m
        new_d = is_objectid_double(line['ObjectEntitiesID'])
        double_object_ids += new_d
        new_answer_percentage = answer_percentage(line['OGAnswer'])
        new_e = check_empty(line['ObjectEntitiesID'])
        false_empty += new_e


        if new_u > 0:
            print(f"Unformat Line {idx}: For {relation} in prompt {prompt_id} {line['ObjectEntitiesID']}")
        if new_m > 0:
            print(f"Missing Line {idx}: For {relation} in prompt {prompt_id} {line['OGAnswer']}")
        if new_d > 0:
            print(f"Duplicate Line {idx}: For {relation} in prompt {prompt_id} {line['ObjectEntitiesID']}")
        if new_e > 0:
            print(f"False Empty Line {idx}: For {relation} in prompt {prompt_id} {line['ObjectEntitiesID']}")


        if new_answer_percentage > highest_answer_percentage:
            highest_answer_percentage = new_answer_percentage

        if new_m > 0 and new_d == 0:
            print(f"Tokensize not enough for {relation}")

    unformatted_object_ids /= len(df)
    missing_answer_part /= len(df)
    double_object_ids /= len(df)
    false_empty /= len(df)

    return unformatted_object_ids, missing_answer_part, double_object_ids, highest_answer_percentage, false_empty


def try_int_cast(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def count_unformatted_object_ids(object_ids):
    counter = 0
    for obj_id in object_ids:
        if obj_id != "" and not try_int_cast(obj_id) and obj_id[0] != "Q":
            counter += 1
    return counter


def is_answer_missing_part(answer):
    return "]" not in answer


def is_objectid_double(object_ids):
    counter = Counter(object_ids)
    return sum(val - 1 for val in counter.values() if val > 1)


def answer_percentage(answer):
    pos = answer.find(']\n')
    if pos == -1:
        return 1
    else:
        return pos / len(answer)


def check_empty(object_ids):
    if "" in object_ids and len(object_ids) > 1:
        return True
    else:
        return False


generate_analysis()
