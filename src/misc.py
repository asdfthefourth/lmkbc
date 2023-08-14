import json

from src.evaluate import split_ensemble_files_by_relation

change_rel = ['PersonHasNumberOfChildren', 'SeriesHasNumberOfEpisodes']
fix_dir = "data/results/ensemble_size_11_train_1"
file_dict = split_ensemble_files_by_relation(fix_dir)

for rel in change_rel:
    for file in file_dict[rel]:
        new_content = []
        new_file = fix_dir + file.split(fix_dir)[1]
        with open(file) as fp:
            for line in fp:
                new_content.append(json.loads(line))
        with open(new_file, 'w') as fp:
            for line in new_content:
                line["ObjectEntitiesID"] = [str(line["ObjectEntitiesID"][0])]
                fp.write(json.dumps(line) + "\n")
