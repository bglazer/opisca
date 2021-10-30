import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
from copy import deepcopy
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)["input_ids"]
    return examples

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        if args.n_process == 1:
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        else:
            n_proc = args.n_process
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(lines)/n_proc)
            for i in range(1, n_proc+1):
                if i != n_proc:
                    indexes.append(len_slice*(i))
                else:
                    indexes.append(len(lines))
            results = []
            for i in range(n_proc):
                results.append(p.apply_async(convert_line_to_example,[tokenizer, lines[indexes[i]:indexes[i+1]], block_size,]))
                print(str(i) + " start")
            p.close() 
            p.join()

            self.examples = []
            for result in results:
                ids = result.get()
                self.examples.extend(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
