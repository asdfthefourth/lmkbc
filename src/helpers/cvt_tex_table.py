import json

with open("data/results/eval_big_few_shot_10_val/summary/prompt_comparison.json") as file:
    cvt = json.load(file)
sorted_data = {k: cvt[k] for k in sorted(cvt)}
with open("data/prompts/ensemble_prompts.json") as file:
    ensemble_dict = json.load(file)
base = """Relation & Prompt & F1-Score \\\\ \hline \hline\n"""
bad_relation_template = "{relation} & *\\color*red**{prompt} **& {f1_score} \\\\\hline\n"
best_relation_template = "{relation} & *\\color*green**{prompt} **& {f1_score} \\\\\n"
res = ""
res += base
for relation in sorted_data:
    top_2 = sorted_data[relation]["2_best"][:1]
    worst_2 = sorted_data[relation]["2_worst"][:1]
    for top in top_2:
        res += best_relation_template.format(relation=relation, prompt=[ensemble['prompt'] for ensemble in
                                                                   ensemble_dict[relation] if ensemble['id'] == int(top[0])][0],
                                        f1_score=top[1]['f1'])
    for worst in worst_2:
        res += bad_relation_template.format(relation=relation, prompt=[ensemble['prompt'] for ensemble in
                                                                   ensemble_dict[relation] if ensemble['id'] == int(worst[0])][0],
                                        f1_score=worst[1]['f1'])
print(res.replace("_", "\_").replace("**", "}").replace("*", "{"))
