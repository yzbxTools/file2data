"""
test the fuzzy match for file names

args:
    --txt: for each line, with the format of: file_query file_match
"""

import os
import os.path as osp
import argparse
from thefuzz import fuzz as fuzz1
from rapidfuzz import fuzz as fuzz2
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, required=True)
    args = parser.parse_args()

    with open(args.txt, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        file_query, file_match = line.split()
        print(file_query, file_match)
        for module in [fuzz1, fuzz2]:
            for func in ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']:
                # file_score = getattr(module, func)(file_query, file_match)
                # name_score = getattr(module, func)(file_query, os.path.basename(file_match))
                
                file_score = getattr(module, func)(file_match, file_query)
                name_score = getattr(module, func)(os.path.basename(file_match), file_query)

                if file_score > name_score:
                    logger.info(f'file vs name {func}: {file_score} > {name_score}')
                else:
                    logger.warning(f'file vs name {func}: {file_score} < {name_score}')
                break

if __name__ == '__main__':
    main()