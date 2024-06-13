import copy
import os
import re
import time
import json
from tqdm import tqdm
import argparse
from anthropic import AnthropicVertex
from datasets import load_dataset

def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    return contents.replace(f"{lan}\n", '')

class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def rerank(self,docs,query,topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages:\n'
                  f'```json\n'
                  f'[...]\n'
                  f'```')
        message = self.model.messages.create(
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.version,
            temperature=0.8,
            top_p=0.8
        )
        response = json.loads(message.model_dump_json(indent=2))
        ranks = extract_program(response['content'][0]['text'],lan='json')
        return [indices_map[r] for r in ranks]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa',
                                 'stackoverflow','sustainable_living','math','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--llm', type=str, default=None)
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--rerank_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    args = parser.parse_args()
    with open(args.rerank_score_file,'w') as f:
        json.dump('hello', f, indent=2)
    raw_examples = load_dataset('xlangai/BRIGHT', 'examples')[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/BRIGHT', 'long_documents')[args.task]
    else:
        doc_pairs = load_dataset('xlangai/BRIGHT', 'documents')[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']
    with open(args.score_file) as f:
        all_scores = json.load(f)
    new_scores = copy.deepcopy(all_scores)
    for qid,scores in all_scores.items():
        docs = []
        sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
        for did,_ in enumerate(sorted_scores):
            docs.append([did,documents[did]])
        if 'claude' in args.llm:
            model = ClaudeModel(version=args.llm)
        else:
            raise ValueError(f'Model {args.llm} has not been implemented')
        exec_count = 0
        success = False
        while not success:
            if exec_count > 5:
                print('Fail after trying for 5 times')
                break
            exec_count += 1
            try:
                new_rank = model.rerank(docs=docs,query=examples[qid]['query'],topk=args.k)
                cur_score = {}
                for rank_id, r in enumerate(new_rank):
                    cur_score[r] = args.k - rank_id
                new_scores[qid] = cur_score
                success = True
            except:
                time.sleep(5)
    with open(args.rerank_score_file,'w') as f:
        json.dump(new_scores, f, indent=2)





