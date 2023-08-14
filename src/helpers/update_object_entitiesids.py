from joblib import Memory

from src.evaluate import split_ensemble_files_by_relation
from src.utils import disambiguation_baseline, disambiguation_improved, DisambiguationImproved, _disambiguate_files, \
    _ambiguate_files, ambiguation_baseline


def main():
    update_dir = "data/results/gpt_test_1"
    use_new = True
    func_cache_memory = Memory("data/func_cache/", verbose=0)
    baseline_disam = func_cache_memory.cache(disambiguation_baseline)
    baseline_ambi = func_cache_memory.cache(ambiguation_baseline)
    if use_new:
        pass
    else:
        cached_disambiguation_func = func_cache_memory.cache(disambiguation_baseline)
        cached_ambiguation_func = func_cache_memory.cache(ambiguation_baseline)
    rel_dict = split_ensemble_files_by_relation(update_dir)
    for relation in rel_dict:
        print(f'Started {relation}')
        _disambiguate_files(cache_disambiguate_func=baseline_disam, file_list=rel_dict[relation], relation=relation)
        _ambiguate_files(cache_disambiguate_func=baseline_ambi, file_list=rel_dict[relation],
                            relation=relation)

if __name__ == '__main__':
    main()