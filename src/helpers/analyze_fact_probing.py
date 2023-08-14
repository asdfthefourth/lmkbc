import copy
import json
from src.evaluate import split_ensemble_files_by_relation


class FactProbe():
    def __init__(self):
        self.dir = "data/results/eval_big_few_shot_10_val"
        self.facts_dir = "data/results/eval_big_few_shot_10_val/facts"
        self.res_dir = "data/results/eval_big_few_shot_10_val/test_facts"
        self.use_facts()


    def process_facts_answers(self):
        pass

    def use_facts(self, pre_consens=True):
        if pre_consens:
            ensemble_dict = split_ensemble_files_by_relation(self.dir)
        else:
            ensemble_dict = split_ensemble_files_by_relation(self.dir)

        with open(self.facts_dir + '/facts.jsonl') as file:
            self.facts = [json.loads(line) for line in file]

        for relation in ensemble_dict:
            if relation == 'PersonHasPlaceOfDeath':
                continue
            if relation not in ["CompoundHasParts", "CountryHasStates", "SeriesHasNumberOfEpisodes", "FootballerPlaysPosition", "BandHasMember", "PersonHasEmployer"]:
                continue
            for file_name in ensemble_dict[relation]:
                prompt_id = file_name.split('.jsonl')[0].split('_')[-1]
                new_df = []
                filtered_facts = [d for d in self.facts if d['relation'] == relation and d['prompt_id'] == prompt_id]
                with open(file_name) as file:
                    df = [json.loads(line) for line in file]
                for line in df:
                    new_line = copy.deepcopy(line)
                    new_obj_entities = []
                    for obj_ent in line['ObjectEntities']:
                        try:
                            full_question = [d for d in filtered_facts if d['subject'] == obj_ent and d['object'] == line['SubjectEntity']]
                            part_question = full_question[0]['unprocessed_res'][0]['generated_text']
                            # answer = full_question.split('\n\n')[2].split('Answer: ')[1]
                            answer = part_question.split('\n')
                            answer = answer[-1]
                            answer = answer.split('Answer: ')
                            answer = answer[1]
                            if answer == 'Yes':
                                new_obj_entities.append(obj_ent)
                            else:
                                pass
                        except:
                            pass
                    new_line['ObjectEntities'] = new_obj_entities
                    new_df.append(new_line)
                with open(self.dir + "/" + f"facte_{relation}_{prompt_id}.jsonl", 'w+') as file:
                    for line in new_df:
                        file.write(json.dumps(line) + "\n")




if __name__ == '__main__':
    fact = FactProbe()