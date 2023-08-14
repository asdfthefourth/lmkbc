import argparse
import collections
import gc
import itertools
import json
import logging
import os
import time
from itertools import combinations, chain
from pathlib import Path

import openai
import tiktoken
import ast

from joblib import Memory

from optimum.bettertransformer import BetterTransformer
from multiprocessing import Process, Pool

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, \
    StoppingCriteriaList

from evaluate import convert_jsonl_to_dict, calc_scores_with_dict, calc_new_score_with_list
from helpers.stoppingcriteria import EndListCriteria
from utils import read_train_data_from_file, create_prompt, disambiguation_baseline, \
    read_lm_kbc_jsonl_to_df, disambiguation_improved, DisambiguationImproved, create_fact_prompt, create_gpt_prompt, \
    _disambiguate_files

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def gpt_response(q, max_new_tokens, model):
    no_response_yet = True
    while no_response_yet:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=q,
                max_tokens=max_new_tokens,
                frequency_penalty=0
            )
            return response
        except openai.error.RateLimitError as e:
            logger.info(f"Sleep {e.headers.get('x-ratelimit-reset-tokens')} seconds")
            try:
                time.sleep(e.headers.get("x-ratelimit-reset-tokens") + 5)
            except:
                time.sleep(50)
        except Exception as e:
            logger.error(e)
            time.sleep(10)


class PromptEnsemble:
    def __init__(self, prog_args: argparse.Namespace):
        self.args = prog_args
        self._parse_args()

    def _parse_args(self):
        # Dataset paths and Output paths
        self.task = "text-generation"
        self.train_data_path = self.args.train_data
        self.val_data_path = self.args.val_data
        self.output_folder = self.args.output
        self.prompt_dir = self.args.prompts
        self.model_cache_path = self.args.model_cache
        self.func_cache_path = self.args.func_cache

        self.func_cache_memory = Memory(self.func_cache_path, verbose=0)
        self.cached_disambiguation_baseline = DisambiguationImproved(
            self.func_cache_memory.cache(disambiguation_baseline))
        self.process_list = []

        self.name = self.args.name
        self.use_val_data = self.args.use_val
        self.multi = self.args.multi
        self.use_gpt = self.args.use_gpt

        if not self.args.relations:
            self.relations = read_lm_kbc_jsonl_to_df(Path(self.train_data_path)).Relation.unique()
        else:
            self.relations = self.args.relations
        # Model params
        self.model_name = self.args.model
        print(self.args.memory_management)
        self.memory_map = {
            int(key) if key != "cpu" else key: value
            for key, value in json.loads(self.args.memory_management).items()
        }
        if 'cpu' in self.memory_map:
            self.cpu = True
            torch.set_num_threads(64)
        else:
            self.cpu = False
        self.batch_size = self.args.batch_size
        self.few_shot = self.args.few_shot
        self.top_k = self.args.top_k
        with open("data/prompts/elicitation.json") as file:
            self.elicitation = json.load(file)
        # Api Tokens
        self.hf_token = self.args.hftoken  # Huggingface token for model download

        openai.api_key = self.args.openaikey

        if self.multi:
            self.multi_config_file_path = f'{self.output_folder}{self.name}/summary/multi_config.json'
            self.create_or_read_multi_config_file()

        if not self.args.consensus and not self.args.transfer_config_name and not self.use_gpt:
            logger.info("Setup pipeline and model")
            self._setup_pipeline()
            logger.info("Finished Setup")

        if self.args.transfer_config_name:
            self.transfer_config_name = self.args.transfer_config_name

        self.ensemble_files = {}
        self._setup_prompts()

        self.new_token_dict = {
            'BandHasMember': 100,
            'CityLocatedAtRiver': 60,
            'CompanyHasParentOrganisation': 30,
            'CompoundHasParts': 30,
            'CountryBordersCountry': 115,
            'CountryHasOfficialLanguage': 90,
            'CountryHasStates': 90,
            'FootballerPlaysPosition': 15,
            'PersonCauseOfDeath': 15,
            'PersonHasAutobiography': 20,
            'PersonHasEmployer': 45,
            'PersonHasNoblePrize': 25,
            'PersonHasNumberOfChildren': 6,
            'PersonHasPlaceOfDeath': 15,
            'PersonHasProfession': 60,
            'PersonHasSpouse': 20,
            'PersonPlaysInstrument': 60,
            'PersonSpeaksLanguage': 60,
            'RiverBasinsCountry': 65,
            'SeriesHasNumberOfEpisodes': 10,
            'StateBordersState': 115,
        }


    def _setup_prompts(self):
        with open(self.prompt_dir + "ensemble_prompts.json") as file:
            self.ensemble_prompt_templates = json.load(file)
        self.few_shot_prompts = {}
        for key, values in self.ensemble_prompt_templates.items():
            for value in values:
                if value["base_prompt"]:
                    self.few_shot_prompts.setdefault(key, value["prompt"])

    def create_multi_config_file(self):
        Path(f"{self.output_folder}{self.name}/summary").mkdir(parents=True, exist_ok=True)
        config_file = {
            'started_relations': [],
            'use_val': self.use_val_data,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'few_shot': self.few_shot,
            'model_name': self.model_name
        }
        with open(self.multi_config_file_path, 'w') as file:
            json.dump(config_file, file)

    def read_multi_config_file(self):
        with open(self.multi_config_file_path, 'r') as file:
            config_file = json.load(file)
        self.use_val_data = config_file['use_val']
        self.train_data_path = config_file['train_data_path']
        self.val_data_path = config_file['val_data_path']
        self.few_shot = config_file['few_shot']
        self.model_name = config_file['model_name']

    def check_relation_already_reserved(self, relation):
        with open(self.multi_config_file_path, 'r') as file:
            multi_config = json.load(file)
        if relation in multi_config['started_relations']:
            return True
        else:
            with open(self.multi_config_file_path, 'w') as file:
                multi_config['started_relations'].append(relation)
                json.dump(multi_config, file)
            return False

    def create_or_read_multi_config_file(self):
        if os.path.isfile(self.multi_config_file_path):
            self.read_multi_config_file()
        else:
            self.create_multi_config_file()

    def _setup_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        self.tokenizer.padding_side = 'left'

        nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16)

        if self.cpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model, cache_dir=self.model_cache_path,
                use_auth_token=self.args.hftoken, low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model, cache_dir=self.model_cache_path, device_map='sequential',
                use_auth_token=self.args.hftoken,
                quantization_config=nf4_config, max_memory=self.memory_map)
            self.stopping_crit_list = StoppingCriteriaList()
            stopping_list = self.tokenizer.encode("]\n")
            stopping_list = stopping_list[:2]
            # self.stopping_crit_list.append(EndListCriteria([stopping_list]))

        self.pipe = pipeline(task=self.task, model=self.model, tokenizer=self.tokenizer, top_k=args.top_k)
        self.tokenizer.pad_token_id = self.model.config.eos_token_id

    def reset_pipeline(self):
        del self.model
        del self.pipe
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self._setup_pipeline()

    def run(self):
        Path(self.output_folder + f"{self.name}/summary").mkdir(parents=True, exist_ok=True)
        Path(self.output_folder + f"{self.name}/facts").mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting Run for these Relations: {self.relations}")
        token_dict = {}
        for relation in self.relations:
            # Path(self.output_folder + f"{self.name}/checkpoint_{relation}/").mkdir(parents=True, exist_ok=True)
            if self.multi:
                if not self.check_relation_already_reserved(relation):
                    self._run_relation(relation)
                    # self.reset_pipeline()
                else:
                    logger.info(f"Skipping relation because its already done or running: {relation}")
            else:
                self._run_relation(relation)
                # self.reset_pipeline()

        for process in self.process_list:
            process.join()
        self.process_list = []
        self.model = ""
        self.pipe = ""
        self.tokenizer = ""
        self.calc_consens_configs()

    def transfer_config(self):
        with open(f'{self.output_folder}{self.transfer_config_name}/summary/full_config.json') as config_file:
            base_config = json.load(config_file)

        for relation in base_config.keys():
            rel_config = base_config[relation]
            for config in rel_config['configs'][-1:]:
                self._reach_consensus(relation=relation, threshold=config['threshold'], exclusion_list=config['config'],
                                      write=True, transfer=True)

    def calc_consens_configs(self):
        with Pool(12) as p:
            results = p.map(self._process_calc_consens_config_relation, list(self.ensemble_files.keys()))

        all_configs = {relation: best_config for relation, best_config in results}
        total_f1 = 0
        for relation, config in all_configs.items():
            total_f1 += config['f1']
            self._reach_consensus(relation, config['configs'][0]['threshold'], config['configs'][0]['config'],
                                  write=True)
        with open(f"{self.output_folder}{self.name}/summary/full_config.json", 'w') as file:
            json.dump(all_configs, file)
        logger.info(f"Best Collection of Permutations found with score: {total_f1 / len(all_configs.items())}")

    def _process_calc_consens_config_relation(self, relation):
        if self.use_val_data:
            use_file = self.val_data_path
        else:
            use_file = self.train_data_path
        with open(use_file) as file:
            gt_jsonl = [json.loads(line) for line in file if json.loads(line)["Relation"] == relation]
        logger.info(f"Consens search for relation: {relation} starts")
        prompt_ids = [line['id'] for line in self.ensemble_prompt_templates[relation]]

        permutations = chain(*[combinations(prompt_ids, i) for i in range(10)])

        # permutations = itertools.permutations(prompt_ids, 3)
        best_config = {'f1': 0, 'recall': 0, 'precision': 0, 'configs': []}
        for permutation in permutations:
            permutation = list(permutation)
            for threshold in range(0, 11, 1):
                threshold = threshold / 11
                consens_jsonl = self._reach_consensus(relation, threshold, permutation, write=False)
                score = calc_new_score_with_list(pred_rows=consens_jsonl, gt_rows=gt_jsonl)[relation]
                if score['f1'] > best_config['f1']:
                    logger.info(
                        f"Consens search for relation: {relation} found new config: {permutation}, threshold: {threshold}, f1: {score['f1']}")
                    best_config['f1'] = score['f1']
                    best_config['recall'] = score['recall']
                    best_config['precision'] = score['precision']
                    best_config['configs'] = [{'config': permutation, 'threshold': threshold}]
                elif score['f1'] == best_config['f1']:
                    best_config['configs'].append({'config': permutation, 'threshold': threshold})

        logger.info(
            f"Consens search for relation: {relation} found best config: {best_config['configs']}, f1: {best_config['f1']}")
        return relation, best_config

    def _run_relation(self, relation):
        logger.info(f"Starting Run for {relation}")

        instantiated_templates = []
        train_data = read_train_data_from_file(self.train_data_path, relation)
        logger.info(f"Data length: {len(train_data)}")
        for row in train_data:
            prompt_template = self.few_shot_prompts[relation]
            object_entities = row['ObjectEntities']
            answers = ' [' + ', '.join(object_entities) + ']'
            instantiated_example = "Question: " + prompt_template.format(
                subject_entity=row["SubjectEntity"]) + f" Answer: {answers}"
            instantiated_templates.append(instantiated_example)

        if self.use_val_data:
            val_data = read_train_data_from_file(self.val_data_path, relation)
            used_data = val_data
        else:
            used_data = train_data
        used_data = used_data
        df = self._generate_dataset(self.ensemble_prompt_templates[relation], instantiated_templates, used_data)
        logger.info(f"Full dataset size: {len(df)}")

        if self.use_gpt:
            logger.debug(f'{relation} tokensizes: {self._calc_token_size(df, relation)}')
            df = self._generate_gpt_dataset(self.ensemble_prompt_templates[relation], instantiated_templates, used_data, relation)
            output = self._execute_prompt_gpt(df, self.new_token_dict[relation])
            preprocessed_file = self._handle_output_gpt(used_data * len(self.ensemble_prompt_templates[relation]),
                                                        output, df,
                                                        relation,
                                                        len(used_data))
        else:
            df = self._generate_dataset(self.ensemble_prompt_templates[relation], instantiated_templates, used_data)
            # output = self._execute_prompt(df, self.new_token_dict[relation])
            output = self.test_execute_prompt(df, self.new_token_dict[relation])
            preprocessed_file = self._handle_output(used_data * len(self.ensemble_prompt_templates[relation]), output,
                                                    df,
                                                    relation,
                                                    len(used_data))
        process = Process(target=_disambiguate_files,
                          args=(self.cached_disambiguation_baseline, preprocessed_file, relation))
        process.start()
        self.process_list.append(process)

    def _generate_dataset(self, prompts, instantiated_templates, input_rows):
        df = []
        sorted_prompts = sorted(prompts, key=lambda item: item['id'])
        for prompt in sorted_prompts:
            df.extend([create_prompt(
                subject_entity=row["SubjectEntity"],
                prompt_template=prompt["prompt"],
                instantiated_templates=instantiated_templates,
                few_shot=self.few_shot
            ) for row in input_rows])
        return df

    def _generate_gpt_dataset(self, prompts, instantiated_templates, input_rows, relation):
        df = []
        sorted_prompts = sorted(prompts, key=lambda item: item['id'])
        for prompt in sorted_prompts:
            df.extend([create_gpt_prompt(
                subject_entity=self.elicitation[relation] + row["SubjectEntity"],
                prompt_template=prompt["prompt"],
                instantiated_templates=instantiated_templates,
                few_shot=self.few_shot
            ) for row in input_rows])
        return df

    def _calc_token_size(self, df, relation):
        output_length = self.new_token_dict[relation] * len(df)
        enc = tiktoken.encoding_for_model('gpt-4')
        input_tokens = enc.encode_batch(df)
        input_length = sum(len(sublist) for sublist in input_tokens)

        return input_length, output_length

    def _execute_prompt(self, prompt_df, new_token_length=40):
        logger.info("Starting pipeline")
        outputs = self.pipe(prompt_df, batch_size=args.batch_size, max_new_tokens=new_token_length)
        logger.info("Finished pipeline")
        return outputs

    def test_execute_prompt(self, prompt_df, new_token_length=40):
        generate_texts = []
        input_ids = self.tokenizer.batch_encode_plus(prompt_df, padding='longest',
                                                     # pads to the longest sequence in the batch
                                                     truncation=True,
                                                     max_length=512,  # max length to truncate/pad
                                                     return_tensors="pt")
        # batched_inputs = torch.split_with_sizes(input_ids['input_ids'], self.batch_size, dim=0)
        # batched_attentionmaks = torch.split_with_sizes(input_ids['attention_mask'], self.batch_size, dim=0)

        batches = {'input_ids': [input_ids['input_ids'][i:i + self.batch_size] for i in
                                 range(0, input_ids['input_ids'].size(0), self.batch_size)],
                   'attention_mask': [input_ids['attention_mask'][i:i + self.batch_size] for i in
                                      range(0, input_ids['attention_mask'].size(0), self.batch_size)]
                   }
        for i, a in tqdm(zip(batches['input_ids'], batches['attention_mask']), total=len(batches['input_ids'])):
            i = i.to('cuda')
            a = a.to('cuda')
            with torch.no_grad():
                outputs = self.model.generate(i, attention_mask=a, max_new_tokens=new_token_length,
                                              num_return_sequences=1)  # , stopping_criteria=self.stopping_crit_list)
            generate_texts.extend([self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
        outputs = [[{'generated_text': generate_text}] for generate_text in generate_texts]
        return outputs

    def _execute_prompt_gpt(self, prompt_df, new_token_length=40):
        with Pool(4) as p:
            res = p.starmap(gpt_response, tqdm(zip(prompt_df, [new_token_length] * len(prompt_df),
                        [self.model_name] * len(prompt_df)), total=len(prompt_df)))
        return res

    def _handle_output_gpt(self, inputs, outputs, prompts, relation, data_size):
        results = []
        logger.info("Start Answer process")
        for row, output, prompt in zip(inputs, outputs, prompts):
            object_entities_with_wikidata_id = []
            # Remove the original prompt from the generated text
            og_answer = output.choices[0].message['content'].split("\n")[0]
            qa_answer = og_answer.split("Answer: ")[1].replace("]", "").replace("[", "").replace('\'\'', "").replace(
                '\'', '')
            qa_entities = qa_answer.split(", ")
            # for entity in qa_entities:
            #     wikidata_id = self.cached_disambiguation_baseline(entity)
            #     if relation in ["PersonHasNumberOfChildren", "SeriesHasNumberOfEpisodes"]:
            #         wikidata_id = str(wikidata_id)
            #     object_entities_with_wikidata_id.append(wikidata_id)

            result_row = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "ObjectEntitiesID": [],
                "ObjectEntities": [],
                "OGAnswer": og_answer,
                "QAAnswer": qa_answer,
                "QAEntities": qa_entities,
                "Relation": row["Relation"],
                "FailedConversion": [],
                "Prompt": prompt
            }
            results.append(result_row)
        logger.info("Finished Answer Process")
        logger.info("Start Split output")
        result_dict = self._split_outputs(results, relation, data_size)
        logger.info("Finish Split output")
        files = []
        for key, item in result_dict.items():
            self.ensemble_files.setdefault(relation, []).append(
                self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl")
            files.append(
                self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl")
            with open(self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl",
                      'w') as file:
                for result in item:
                    file.write(json.dumps(result) + "\n")
        return files

    def _handle_output(self, inputs, outputs, prompts, relation, data_size):
        results = []
        logger.info("Start Answer process")
        for row, output, prompt in zip(inputs, outputs, prompts):
            object_entities_with_wikidata_id = []
            # Remove the original prompt from the generated text
            og_answer = output[0]['generated_text'].split(prompt)[-1].strip()
            qa_answer = og_answer.split('\n')[0].split('Answer:')[-1].replace('[', '').replace(']', '').strip()
            qa_entities = qa_answer.split(", ")
            # for entity in qa_entities:
            #     wikidata_id = self.cached_disambiguation_baseline(entity)
            #     if relation in ["PersonHasNumberOfChildren", "SeriesHasNumberOfEpisodes"]:
            #         wikidata_id = str(wikidata_id)
            #     object_entities_with_wikidata_id.append(wikidata_id)

            result_row = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "ObjectEntitiesID": [],
                "ObjectEntities": [],
                "OGAnswer": og_answer,
                "QAAnswer": qa_answer,
                "QAEntities": qa_entities,
                "Relation": row["Relation"],
                "FailedConversion": [],
                "Prompt": prompt
            }
            results.append(result_row)
        logger.info("Finished Answer Process")
        logger.info("Start Split output")
        result_dict = self._split_outputs(results, relation, data_size)
        logger.info("Finish Split output")
        files = []
        for key, item in result_dict.items():
            self.ensemble_files.setdefault(relation, []).append(
                self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl")
            files.append(
                self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl")
            with open(self.output_folder + f"{self.name}/ensemble_{relation}_{key}.jsonl",
                      'w') as file:
                for result in item:
                    file.write(json.dumps(result) + "\n")
        return files

    def _split_outputs(self, outputs, relation, data_size):
        output_dict = {}
        sorted_prompts = sorted(self.ensemble_prompt_templates[relation], key=lambda item: item['id'])
        for p_id, sorted_prompt in enumerate(sorted_prompts):
            output_dict[sorted_prompt['id']] = outputs[data_size * p_id: data_size * (p_id + 1)]
        return output_dict

    def _find_ensemble_files(self):
        files = os.listdir(self.output_folder + self.name)
        file_dict = {}

        for file in files:
            if "ensemble" not in file:
                continue
            name_parts = file.split('_')
            if len(name_parts) > 1:
                key = name_parts[1]  # using the second part
                file_dict.setdefault(key, []).append(self.output_folder + self.name + "/" + file)
        self.ensemble_files = file_dict

    def _reach_consensus(self, relation, threshold, exclusion_list=None, write=True, transfer=False):
        if exclusion_list:
            exclusion_list = [str(i) for i in exclusion_list]
        files = self.ensemble_files[relation]
        consensus_dict = collections.defaultdict(list)

        # Read each file and group ObjectEntitiesID by SubjectEntityID
        for file in files:
            if exclusion_list is not None and file.split('.')[0].split('_')[-1] in exclusion_list:
                continue
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    consensus_dict[data['SubjectEntity']].append(data['ObjectEntitiesID'])

        # calculate threshold count
        threshold_count = threshold * len(files)
        jsonl_result = []
        # Calculate predictions for each SubjectEntityID that exceed the threshold
        for subject_id, predictions in consensus_dict.items():
            object_count = collections.Counter(chain(*predictions))
            if relation in ["PersonHasNumberOfChildren", "SeriesHasNumberOfEpisodes"]:
                obj_entities_id = [obj if count >= threshold_count else '0' for obj, count in
                                   object_count.most_common(1)]
            elif relation in ["PersonHasPlaceOfDeath", "PersonHasNobelPrize", "PersonCauseOfDeath"]:
                obj_entities_id = [obj if count >= threshold_count else '' for obj, count in
                                   object_count.most_common(1)]
            else:
                obj_entities_id = [obj for obj, count in object_count.items() if count >= threshold_count]
            jsonl_result.append(
                {'SubjectEntity': subject_id, 'Relation': relation, 'ObjectEntitiesID': obj_entities_id})
        if write:
            if transfer:
                prefix = 'transfer' + '-'.join(exclusion_list)
            elif exclusion_list:
                prefix = 'exclusion' + '-'.join(exclusion_list)
            else:
                prefix = 'consens'
            with open(self.output_folder + f"{self.name}/{prefix}_{relation}_{threshold}.jsonl", 'w') as f:
                for line in jsonl_result:
                    f.write(json.dumps(line) + '\n')
        return jsonl_result

    def consensus(self):
        self._find_ensemble_files()
        for threshold in range(0, 11, 1):
            for relation in self.relations:
                self._reach_consensus(relation, threshold / 11)

    def fact_probing(self):
        for relation in self.ensemble_files:
            logger.info(f'Started fact probing for {relation}')
            dfs = []
            mappings = []
            fact_size = 0
            for prompt_file in self.ensemble_files[relation]:
                df, mapping = self.generate_fact_probing_df(prompt_file, relation)
                fact_size += len(df)
                dfs.extend(df)
                mappings.extend(mapping)
            logger.info(f'Fact size: {fact_size}')
            result = self.test_execute_prompt(dfs, new_token_length=3)
            self.save_fact_probing(result, mappings)
            logger.info(f'Finished fact probing for {relation}')

    def _run_fact_probing(self, fact_df):
        logger.info("Starting fact pipeline")
        outputs = self.pipe(fact_df, batch_size=128, max_new_tokens=5)
        logger.info("Finished fact pipeline")
        return outputs

    def _run_fact_probing_gpt(self, fact_df):
        pass

    def save_fact_probing(self, df, mappings):

        with open(f'{self.output_folder}{self.name}/facts/facts.jsonl', 'a+') as file:
            for line, mapping in zip(df, mappings):
                res_line = mapping
                res_line['unprocessed_res'] = line
                file.write(json.dumps(res_line) + '\n')

    def generate_fact_probing_df(self, probing_file, relation):
        with open("data/prompts/fact_checking_prompts.json") as file:
            fact_probing_prompts = json.load(file)
        with open("data/prompts/few_shot/fact_checking_examples.json") as file:
            fact_probing_examples_train = json.load(file)

        prompt_id = probing_file.split('.jsonl')[0].split('_')[-1]

        with open(probing_file) as file:
            df = [json.loads(line) for line in file]

        prob_df = [
            create_fact_prompt(subject_entity=subj, prompt_template=fact_probing_prompts[relation],
                               object_entity=row['SubjectEntity'],
                               instantiated_templates=fact_probing_examples_train["few_shots"], few_shot=self.few_shot)
            for row in df for subj in row['ObjectEntities']
        ]

        mapping_df = [
            {'relation': relation,
             'prompt_id': prompt_id,
             'subject': subj,
             'object': row['SubjectEntity']}
            for row in df for subj in row['ObjectEntities']
        ]

        return prob_df, mapping_df


def main(prog_args: argparse.Namespace):
    prompt_ensemble = PromptEnsemble(prog_args)
    if prog_args.consensus:
        # prompt_ensemble.consensus()
        prompt_ensemble._find_ensemble_files()
        prompt_ensemble.calc_consens_configs()
    elif prog_args.transfer_config_name:
        prompt_ensemble._find_ensemble_files()
        prompt_ensemble.transfer_config()
    elif prog_args.only_fact_probing:
        prompt_ensemble._find_ensemble_files()
        prompt_ensemble.fact_probing()

    else:
        prompt_ensemble.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model with Question and Fill-Mask Prompts")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="HuggingFace model name")
    parser.add_argument("--train_data", type=str, required=False,
                        help="CSV file containing train data for few-shot examples (required)",
                        default="data/base_dataset/train.jsonl")
    parser.add_argument("--val_data", type=str, required=False, help="Input test file (required)",
                        default="data/base_dataset/val.jsonl")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output Dir (required)",
                        default="data/results/")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Top k prompt outputs (default: 100)")
    parser.add_argument("-f", "--few_shot", type=int, default=2, help="Number of few-shot examples (default: 5)")
    parser.add_argument("--memory_management", type=str, required=True)
    parser.add_argument("--model_cache", type=str, required=False, default="../cache/",
                        help="Directory for caching the hf models")
    parser.add_argument("--func_cache", type=str, default="data/func_cache/")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model. (default:32)")
    parser.add_argument("--relations", type=str, nargs="+")
    parser.add_argument("--hftoken", type=str, default="")
    parser.add_argument("--openaikey", type=str)
    parser.add_argument("--name", type=str, required=True, help="Run name used for file naming etc")
    parser.add_argument("--prompts", type=str, required=False, default="data/prompts/")
    parser.add_argument("--consensus", action='store_true', default=False)
    parser.add_argument("--use_val", action='store_true', default=False)
    parser.add_argument("--transfer_config_name", type=str)
    parser.add_argument("--multi", action='store_true', default=False)
    parser.add_argument("--only_fact_probing", action='store_true', default=False)
    parser.add_argument("--use_gpt", action='store_true', default=False)

    args = parser.parse_args()
    main(args)
