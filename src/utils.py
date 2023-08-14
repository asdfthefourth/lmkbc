import random
import json
import csv
from multiprocessing import Pool
from pathlib import Path

import requests
import pandas as pd
import re

from typing import List, Union

from tqdm import tqdm


def create_prompt(subject_entity: str, prompt_template: str, instantiated_templates: List[str],
                  few_shot: int = 0) -> str:
    task_explanation = "Please answer the question. Beforehand there a few examples."
    if few_shot > 0:
        random_examples = random.sample(instantiated_templates, min(few_shot, len(instantiated_templates)))
    else:
        random_examples = []
    few_shot_examples = "\n".join(random_examples)
    if few_shot > 0:
        prompt = f"{task_explanation}\n{few_shot_examples}\nQuestion: {prompt_template.format(subject_entity=subject_entity)}"
    else:
        prompt = f"{task_explanation}\nQuestion: {prompt_template.format(subject_entity=subject_entity)}"
    return prompt


def create_gpt_prompt(subject_entity: str, prompt_template: str, instantiated_templates: List[str],
                      few_shot: int = 0) -> str:
    task_explanation = ("Please answer the question. Beforehand there a few examples. The output format shoudl be a "
                        "list of possible answers prefaced by 'Answer: ', also if there is no answer write Answer: ['']")
    if few_shot > 0:
        random_examples = random.sample(instantiated_templates, min(few_shot, len(instantiated_templates)))
    else:
        random_examples = []
    few_shot_examples = "\n".join(random_examples)
    if few_shot > 0:
        system = f"{task_explanation}\n{few_shot_examples}"
        user = f"Question: {prompt_template.format(subject_entity=subject_entity)}"
    else:
        system = f"{task_explanation}"
        user = f"Question: {prompt_template.format(subject_entity=subject_entity)}"
    res = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return res


def create_fact_prompt(subject_entity: str, object_entity: str, prompt_template: str, instantiated_templates: List[str],
                       few_shot: int = 0) -> str:
    # prompt_template = prompt_templates[relation]
    task_explanation = "Please answer the question with yes or no. Beforehand there a few examples."
    few_shot_examples = "\n".join(instantiated_templates)
    if few_shot > 0:
        prompt = f"{task_explanation}\n{few_shot_examples}\nQuestion: {prompt_template.format(subject_entity=subject_entity, object_entity=object_entity)}"
    else:
        prompt = f"{task_explanation}\nQuestion: {prompt_template.format(subject_entity=subject_entity, object_entity=object_entity)}"
    return prompt

def create_gpt_fact_prompt(subject_entity: str, object_entity: str, prompt_template: str, instantiated_templates: List[str],
                       few_shot: int = 0) -> str:
    # prompt_template = prompt_templates[relation]
    task_explanation = "Please answer the question with yes or no. Beforehand there a few examples."
    few_shot_examples = "\n".join(instantiated_templates)
    if few_shot > 0:
        prompt = f"{task_explanation}\n{few_shot_examples}\nQuestion: {prompt_template.format(subject_entity=subject_entity, object_entity=object_entity)}"
    else:
        prompt = f"{task_explanation}\nQuestion: {prompt_template.format(subject_entity=subject_entity, object_entity=object_entity)}"
    return prompt


def read_lm_kbc_jsonl(file_path: Union[str, Path]):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df


def _disambiguate_files(cache_disambiguate_func, file_list, relation):
    for file in tqdm(file_list, total= len(file_list)):
        ambiguous_content = []
        with open(file) as fp:
            ambiguous_content.extend([json.loads(line) for line in fp])

        with Pool(10) as p:
            res = p.starmap(process_disambi, tqdm(
                zip([cache_disambiguate_func] * len(ambiguous_content), ambiguous_content,
                    [relation] * len(ambiguous_content)), total=len(ambiguous_content)))
        with open(file, 'w') as fp:
            for line in res:
                fp.write(json.dumps(line) + "\n")


def process_disambi(cache_disambiguate_func, line, relation):
    line["ObjectEntitiesID"] = []
    for object_entity in line["QAEntities"]:
        if relation not in ["PersonHasNumberOfChildren", "SeriesHasNumberOfEpisodes"]:
            object_id = cache_disambiguate_func(object_entity)
            if object_id and type(object_id) == str:
                line["ObjectEntitiesID"].append(object_id)
            else:
                line.get("FailedConversion", []).append(object_id)
        else:
            line["ObjectEntitiesID"].append(str(object_entity))
    line["ObjectEntitiesID"] = list(set(line["ObjectEntitiesID"]))
    if len(line["ObjectEntitiesID"]) == 0:
        line["ObjectEntitiesID"] = [""]
    return line


def _ambiguate_files(cache_disambiguate_func, file_list, relation):
    for file in tqdm(file_list, total= len(file_list)):
        ambiguous_content = []
        with open(file) as fp:
            ambiguous_content.extend([json.loads(line) for line in fp])
        with Pool(10) as p:
            res = p.starmap(process_ambi, tqdm(zip([cache_disambiguate_func] * len(ambiguous_content), ambiguous_content, [relation] * len(ambiguous_content)), total=len(ambiguous_content)))
        with open(file, 'w') as fp:
            for line in res:
                fp.write(json.dumps(line) + "\n")


def process_ambi(cache_disambiguate_func, line, relation):
    line["ObjectEntities"] = []
    for object_entity in line["ObjectEntitiesID"]:
        if relation not in ["PersonHasNumberOfChildren", "SeriesHasNumberOfEpisodes"]:
            object_id = cache_disambiguate_func(object_entity)
            if object_id and type(object_id) == str:
                line["ObjectEntities"].append(object_id)
            else:
                line.get("FailedConversion", []).append(object_id)
        else:
            line["ObjectEntities"].append(str(object_entity))
    line["ObjectEntities"] = list(set(line["ObjectEntities"]))
    if len(line["ObjectEntities"]) == 0:
        line["ObjectEntities"] = [""]
    return line


# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item


def ambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['label']
        except:
            return item


class DisambiguationImproved():

    def __init__(self, cached_func):
        self.func = cached_func

    def __call__(self, *args, **kwargs):
        item = args[0]
        res = self.func(item)
        if item == res and res != '' and res[0] != 'Q':
            return None
        else:
            return res


def disambiguation_improved(cached_func, item):
    res = cached_func(item)
    if item == res and res[0] != 'Q':
        return None
    else:
        return res


# Read prompt templates from a CSV file
def read_prompt_templates_from_csv(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates


def read_train_data_from_file(file_path: str, relation=None):
    """
    This function reads a given CSV file and converts each line into a dictionary using json.loads.

    If a 'relation' argument is provided, only records that match the provided relation are appended to the train_data list.
    If 'relation' is None (or not provided), this function appends all records to the train_data list.

    Args:
        file_path (str): The path to the input CSV file.
        relation (str, optional): The relation to filter records by. If None, all records are included.

    Returns:
        train_data (List[Dict]): A list of dictionaries representing the training data in the input CSV file.
    """
    train_data = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if relation is None or data.get("Relation") == relation:
                train_data.append(data)
    return train_data


def generate_ensemble_files():
    path = "../data/prompts/ensemble/"
    prompts = process_input()
    for relation in read_lm_kbc_jsonl_to_df(Path("../data/base_dataset/train.jsonl")).Relation.unique():
        with open(f"{path}{relation}.jsonl", 'w') as file:
            for prompt in prompts[relation]:
                file.write(prompt + "\n")


def process_input():
    full_text_path = "data/prompts/ensemble/gpt_prompts.txt"
    res_dict = {}
    with open(full_text_path) as file:
        full_text = "\n".join([row for row in file])
    splits = re.split("\s\d{1,2}.\s", full_text)[1:]
    for split in splits:
        idx = 0
        relation = split.split(":")[0]
        prompt_block = split.split(":")[1].split("\n")
        for prompt in prompt_block:
            if not prompt:
                continue
            cleaned_prompt = prompt.strip()
            prompt_dict = {'id': idx, 'prompt': cleaned_prompt, 'base_prompt': False}
            if idx == 0:
                prompt_dict["base_prompt"] = True
            res_dict.setdefault(relation, []).append(prompt_dict)
            idx += 1
    return res_dict


def transfer_config(test_dir, val_dir):
    with open(f'{test_dir}/summary/full_config.json') as file:
        test_config = json.load(file)
