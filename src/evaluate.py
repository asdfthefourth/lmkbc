import json
import os

import string
from typing import List, Dict, Union

import pandas as pd
import numpy as np


def calc_scores(prediction_files: list, ground_truth_file: str, relation: str):
    with open(ground_truth_file, 'r') as gt:
        # ground_truths = {json.loads(line)['SubjectEntity']: set(json.loads(line)['ObjectEntitiesID']) for line in gt
        #                  if json.loads(line)['Relation'] == relation}
        ground_truths = [json.loads(line) for line in gt if json.loads(line)['Relation'] == relation]

    best_f1 = [0, -1, -1]
    all_f1 = {}
    for pred_file in prediction_files:
        with open(pred_file, 'r') as pf:
            # predictions = {json.loads(line)['SubjectEntity']: set(json.loads(line)['ObjectEntitiesID']) for line in
            #                pf}
            predictions = [json.loads(line) for line in pf]
        id = pred_file.split('.jsonl')[0].split('_')[-1]
        # res = calc_scores_with_dict(predictions, ground_truths)
        res = calc_new_score_with_list(predictions, ground_truths)[relation]
        all_f1[id] = res
        if res['f1'] > best_f1[0]:
            best_f1 = [res['f1'], res['recall'], res['precision'], pred_file,
                       pred_file.split('.jsonl')[0].split('_')[-1]]
        elif res['f1'] == best_f1[0]:
            best_f1[1] = res['recall']
            best_f1[2] = res['precision']
            best_f1.append(pred_file.split('.jsonl')[0].split('_')[-1])
        print(f'For file {pred_file}, Precision: {res["precision"]}, Recall: {res["recall"]}, F1-score: {res["f1"]}')
    return best_f1, all_f1


def convert_jsonl_to_dict(convert_jsonl):
    convert_dict = {line['SubjectEntity']: set(line['ObjectEntitiesID']) for line in convert_jsonl}
    return convert_dict


def calc_scores_with_dict(prediction_dict: dict, ground_truth_dict: dict):
    ground_truths = ground_truth_dict
    predictions = prediction_dict

    TP = sum([len(p & ground_truths[s]) for s, p in predictions.items() if s in ground_truths])
    FP = sum([len(p - ground_truths[s]) for s, p in predictions.items() if s in ground_truths])
    FN = sum([len(ground_truths[s] - p) for s, p in predictions.items() if s in ground_truths])

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall}


def split_ensemble_files_by_relation(directory: str, file_type: str = 'ensemble') -> dict:
    files = os.listdir(directory)
    file_dict = {}

    for file in files:
        if file_type not in file:
            continue
        name_parts = file.split('_')
        if len(name_parts) > 1:
            key = name_parts[1]  # using the second part
            file_dict.setdefault(key, []).append(directory + "/" + file)

    return file_dict


def true_positives(preds: List, gts: List) -> int:
    tp = 0
    for pred in preds:
        if (pred in gts):
            tp += 1

    return tp


def precision(preds: List[str], gts: List[str]) -> float:
    # when nothing is predicted, precision 1 irrespective of the ground truth value
    try:
        if len(preds) == 0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1 even if there are predictions (edge case)
        if len(gts) == 0 or gts == [""]:
            return 1.0
        # When the ground truth is not empty
        return true_positives(preds, gts) / len(gts)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntitiesID"] for r in rows}


def evaluate_per_sr_pair(pred_rows, gt_rows) -> List[Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        # get the ground truth objects
        gts = gt_dict[(subj, rel)]

        # get the predictions
        preds = pred_dict[(subj, rel)]

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "precision": p,
            "recall": r,
            "f1": f1
        })

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        })

    for rel in scores:
        scores[rel] = {
            "precision": sum([x["precision"] for x in scores[rel]]) / len(scores[rel]),
            "recall": sum([x["recall"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores


def calc_new_score_with_list(pred_rows, gt_rows):
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "precision": sum([x["precision"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "recall": sum([x["recall"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(scores_per_relation),
    }

    # print(pd.DataFrame(scores_per_relation).transpose().round(3))
    return scores_per_relation

def show_bad_good_prompts(scores):
    result = {}
    for relation, inner_dict in scores.items():
        # Get inner keys sorted by value
        sorted_keys = sorted(inner_dict.items(), key=lambda item: item[1].get("f1"))

        # The 2 worst and the 2 best for each outer key
        result[relation] = {
            '2_worst': sorted_keys[:2],
            '2_best': sorted_keys[-2:][::-1]  # Reversed for descending order
        }
    with open(ensemble_dir + f"/summary/prompt_comparison.json", 'w') as file:
        json.dump(result, file)


if __name__ == "__main__":
    val_file = "data/base_dataset/val.jsonl"
    ensemble_dir = "data/results/gpt_test_1"
    for eval_type in ["transfer", "ensemble", "exclusion", "facte"]:
        ensemble_dict = split_ensemble_files_by_relation(ensemble_dir, eval_type)
        best_f1_score = 0
        best_recall_score = 0
        best_precision_score = 0
        rel_dict = {}
        comp_dict = {}
        for relation in ensemble_dict.keys():

            score, all_score = calc_scores(prediction_files=ensemble_dict[relation], ground_truth_file=val_file, relation=relation)
            if type(score[2]) == str:
                pass
            comp_dict[relation] = all_score
            best_f1_score += score[0]
            best_recall_score += score[1]
            best_precision_score += score[2]
            rel_dict[relation] = {'f1': score[0], 'recall': score[1], 'precision': score[2], 'file': score[3],
                                  'thresholds': score[4:]}
        show_bad_good_prompts(comp_dict)
        if len(ensemble_dict.keys()) > 0:
            print(f'Best Avg f1: {best_f1_score / len(ensemble_dict.keys())}')
            rel_dict['f1'] = best_f1_score / len(ensemble_dict.keys())
            rel_dict['precision'] = best_precision_score / len(ensemble_dict.keys())
            rel_dict['recall'] = best_recall_score / len(ensemble_dict.keys())
        print(rel_dict)
        with open(ensemble_dir + f"/summary/{eval_type}.json", 'w') as file:
            json.dump(rel_dict, file, sort_keys=True)
            pass
