import torch
import logging
from transformers import StoppingCriteria

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
class EndListCriteria(StoppingCriteria):

    def __init__(self, stopping_words: torch.LongTensor):
        self.stopping_ids = stopping_words

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        input_ids = input_ids.tolist()
        check_list = check_sequences(input_ids, self.stopping_ids)
        res = all(check_list)
        if res:
            logger.debug(f"Stopped Generation cause newline detected at size {len(input_ids[0])}")
        return res


def is_sublist(lst1, lst2):
    len_diff = len(lst1) - len(lst2)
    return any(lst1[i:i+len(lst2)] == lst2 for i in range(len_diff + 1))

# Main function to find if any sequence from list2 is in each sequence from list1
def check_sequences(list1, list2):
    result = []
    for lst in list1:
        result.append(any(is_sublist(lst, seq) for seq in list2))
    return result

