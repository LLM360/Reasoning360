# Code adapted from: https://github.com/keirp/hf-data-scrubber
import datasets
import argparse
from nltk.util import ngrams
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import nltk.util
import glob


# --- Test datasets
def math_dataset(split='test'):
    ds = datasets.load_dataset('competition_math')[split]
    id2example = {}
    for i in range(len(ds)):
        example = ds[i]
        id2example[i] = {
            'id': i,
            'input': example['problem'],
            'output': example['solution'],
            'meta': {
                'level': example['level'],
                'type': example['type']
            }
        }
    return id2example

def math_500(split='test'):
    ds = datasets.load_dataset('HuggingFaceH4/MATH-500', 'default')[split]
    id2example = {}
    for i in range(len(ds)):
        example = ds[i]
        id2example[i] = {
            'id': i,
            'input': example['problem'],
            'output': example['solution'],
        }
    return id2example

def gsm8k(split='test'):
    ds = datasets.load_dataset('gsm8k', 'main')[split]
    id2example = {}
    for i in range(len(ds)):
        example = ds[i]
        id2example[i] = {
            'id': i,
            'input': example['question'],
            'output': example['answer'],
            'meta': {
                'example': example
            }
        }
    return id2example


def model_generations(output_json):
    import json
    with open(output_json) as f:
        data = json.load(f)

    id2example = {}
    i = 0
    for task in data['cache']:
        task_cache = data['cache'][task]
        for item in task_cache:
            if ('selected_answer' not in item['metadata'] or
                    not isinstance(item['metadata']['unprocessed_answers'], str)):
                raise NotImplementedError(
                    'model_generations only supports checking a string stored in metadata["unprocessed_answers"]'
                )
            id2example[i] = {
                'id': i,
                'input': '',
                'output': item['metadata']['unprocessed_answers'],
                'meta': {}
            }

            i += 1
    return id2example

# ---


def load_test_dataset(dataset):
    if dataset == 'MATH':
        id2example = math_dataset()
    elif dataset == 'gsm8k':
        id2example = gsm8k()
    elif dataset == 'MATH-500':
        id2example = math_500()
    else:
        id2example = model_generations(dataset)
    return id2example


def make_test_index(id2example, key):
    # Compute:
    #   id -> {-1 : [full text string],
    #          30   : [30gram1_string, 30gram2_string, ...],
    #          20   : ...
    #          .. }
    index = defaultdict(lambda: defaultdict(list))
    for id_, obj in tqdm(id2example.items(), total=len(id2example)):
        text = obj[key]
        index[id_][-1].append(text)

        for n in [30, 20, 15, 10]:
            obj_ngrams = ngrams(text.split(), n)
            obj_ngrams = list(set([' '.join(ngram) for ngram in obj_ngrams]))
            index[id_][n] = obj_ngrams
    return index


def save_index(id2example, index, output_dir, dataset):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = dataset.replace('/', '_')
    with open(os.path.join(output_dir, '%s_index.json' % dataset), 'w') as f:
        json.dump({
            'id2example': id2example,
            'index': index
        }, f, indent=4)


def save_hits(hits_ds, output_dir, dataset, test_dataset, test_key, ngram_n):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = dataset.replace('/', '_')
    test_dataset = test_dataset.replace('/', '_')
    outfile = os.path.join(
        output_dir,
        '%s_%s_%s_hits_%d.json' % (dataset, test_dataset, test_key, ngram_n)
    )
    hits_ds.to_json(outfile)


def _check_full_string(text, test_ngrams, test_id, ngram_n):
    hits = []
    for ngram in test_ngrams:
        if ngram in text:
            hits.append({
                'test_id': test_id,
                'ngram_n': ngram_n,
                'ngram': ngram
            })
    return hits


def _check_intersection(text_ngrams_set, test_ngrams, test_id, ngram_n):
    hits = []
    intersection = text_ngrams_set.intersection(set(test_ngrams))
    for ngram in intersection:
        hits.append({
            'test_id': test_id,
            'ngram_n': ngram_n,
            'ngram': ngram
        })
    return hits


def get_hits(example, index, column, ngram_n):
    # Map an example to a list of hits. A hit means that some n-gram
    # (with n = ngram_n, or a full string when ngram_n=-1) from the test
    # set occurs in the text found in example[column]. The hit records
    # the matched text and the id of the test example.
    text = example[column]
    # HACK: Add a placeholder, otherwise `datasets` crashes with empty `hits`
    hits = [{
        'test_id': -123,
        'ngram_n': -123,
        "ngram": "[PLACEHOLDER]"
    }]

    if ngram_n != -1:
        text_ngrams = ngrams(text.split(), ngram_n)
        text_ngrams_set = set([' '.join(ngram) for ngram in text_ngrams])

    for test_id in index:
        n2ngrams = index[test_id]
        test_ngrams = n2ngrams[ngram_n]
        if ngram_n == -1:
            hits_ = _check_full_string(text, test_ngrams, test_id, ngram_n)
        else:
            hits_ = _check_intersection(text_ngrams_set, test_ngrams, test_id, ngram_n)
        hits.extend(hits_)
    return {
        'hits': hits
    }


def flatten(data, id2example):
    out = {}
    for k, v in data.items():
        if k != 'hits':
            out[k] = v
    for hit in data['hits']:
        if hit['test_id'] != -123:
            for k, v in hit.items():
                out[k] = v
            for k, v in id2example[hit['test_id']].items():
                out[k] = v
    return out


def main(args):
    # Load test dataset and make an index of n-grams
    test_ds = load_test_dataset(args.test_dataset)
    test_index = make_test_index(test_ds, key=args.test_key)
    save_index(test_ds, test_index, args.output_dir, args.test_dataset)

    # Load dataset and find n-gram matches
    ds = datasets.load_dataset(args.dataset, cache_dir=args.cache_dir, dataset_path=args.dataset_path)['train']

    for ngram_n in args.ngram_n:
        print("##### ngram_n %d" % ngram_n)
        hits_ds = ds.map(
            lambda x: get_hits(x, test_index, args.dataset_column, ngram_n),
            batched=False,
            num_proc=args.n_processes
        )

        hits_ds = hits_ds.filter(
            lambda x: len(x['hits']) > 1,
            batched=False,
            num_proc=args.n_processes
        )
        print("%d hits" % len(hits_ds))

        hits_ds = hits_ds.map(
            lambda x: flatten(x, test_ds),
            batched=False,
            num_proc=args.n_processes
        )
        save_hits(hits_ds, args.output_dir, args.dataset, args.test_dataset, args.test_key, ngram_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-dataset', type=str, default='MATH-500',
        help='dataset name (MATH or gsm8k or MATH-500) or lm-evaluation-harness output JSON path'
    )
    parser.add_argument(
        '--test-key', type=str, default='input', choices=['input', 'output'],
        help='Specifies whether to check overlap with test inputs or test outputs.'
    )
    parser.add_argument('--dataset', type=str, default='open-web-math/open-web-math')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--dataset-column', type=str, default='text')
    parser.add_argument('--n-processes', type=int, default=16)
    parser.add_argument(
        '--ngram-n', type=int, nargs='+',
        default=[20, 30],
        choices=[10, 20, 30, -1],
        help='Ngram overlap(s) to check. -1 for full match of `--test-key`.'
    )
    parser.add_argument('--cache-dir', type=str, default='~/.cache')
    parser.add_argument('--output-dir', type=str, default='output')

    args = parser.parse_args()
    main(args)
