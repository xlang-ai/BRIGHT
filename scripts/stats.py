import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def calc(tokenizer, task, long=False):
    data = load_dataset('xlangai/BRIGHT', 'examples')[task]

    if long:
        corpus = load_dataset('xlangai/bright', 'long_documents')[task]
    else:
        corpus = load_dataset('xlangai/BRIGHT', 'documents')[task]

    num_queries = len(data)
    num_docs = len(corpus)

    data = data.map(lambda x: tokenizer(x["query"]))
    corpus = corpus.map(lambda x: tokenizer(x["content"]))

    q_len = np.mean([len(x["input_ids"]) for x in data])
    d_len = np.mean([len(x["input_ids"]) for x in corpus])

    field = "gold_ids" if not long else "gold_ids_long"
    n_doc = np.mean([len(x[field]) for x in data])

    stats = [task] + [str(x) for x in [num_queries, num_docs, q_len, d_len, n_doc]]
    print(",".join(stats))
    return stats


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tasks = ["aops", "theoremqa_questions", "theoremqa_theorems"]

    stats = []
    for task in tasks:
        s = calc(tokenizer, task)
        stats.append(s)

    print("task,num_queries,num_docs,avg_query_len,avg_doc_len,avg_num_gold_docs")
    for s in stats:
        print(",".join(s))
