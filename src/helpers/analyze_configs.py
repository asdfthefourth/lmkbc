import json

def compare_subsets(dict1, dict2, keys):
    for key in keys:
        if dict1.get(key) != dict2.get(key):
            return False
    return True

def compare_configs():
    file1 = "data/results/ensemble_bigmodel_size_11_val"
    file2 = "data/results/ensemble_bigmodel_size_11_train_1"

    with open(file1 + "/summary/full_config.json") as file:
        dict1 = json.load(file)
    with open(file2 + "/summary/full_config.json") as file:
        dict2 = json.load(file)

    res = {}

    for rel in dict1:
        if rel in dict2:
            res[rel] = {'configs': []}
            for config in dict1[rel]['configs']:
                for conf2 in dict2[rel]['configs']:
                    if compare_subsets(config, conf2, "config"):
                        if config['threshold'] == conf2['threshold']:
                            new_treshold = config['threshold']
                        else:
                            new_treshold = config['threshold'] if config['threshold'] < conf2['threshold'] else conf2['threshold']
                        new_config = {'config': config['config'], 'threshold': new_treshold}
                        res[rel]['configs'].append(new_config)
                        break
    with open("data/results/ensemble_bigmodel_size_11_val/summary/full_config.json", 'w') as file:
        json.dump(res, file)
    print(res)

compare_configs()