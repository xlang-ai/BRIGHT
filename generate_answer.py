import copy
import json
import os
from tqdm import tqdm
from anthropic import AnthropicVertex
import multiprocessing as mp
from datasets import load_dataset

retriever = 'qwen'
PROJECT_ID = ""
LOCATION = ""
claude_client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

def worker(arg):
    exec_count = 0
    documentation = ''
    for did,d in enumerate(arg['documents']):
        documentation += f'---------- Document {did+1} start ----------\n{d.strip()}\n---------- Document {did+1} end ----------\n'
    while exec_count<50:
        try:
            exec_count += 1
            message = claude_client.messages.create(
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"{documentation}\n"
                                   f"---------- post start ----------\n"
                                   f"{arg['query'].strip()}\n"
                                   f"---------- post end ----------\n\n"
                                   f"Let's think step by step to address the query and give a detailed answer.",
                    }
                ],
                model="claude-3-5-sonnet@20240620",
                temperature=0,
            )
            response = json.loads(message.model_dump_json(indent=2))
            completion = response['content'][0]['text']
            output = copy.deepcopy(arg)
            output['pred'] = completion
            with open(os.path.join(f'outputs/{retriever}_retrieval/{arg["task"]}', f"{arg['id']}.json"), 'w') as f:
                json.dump(output, f, indent=2)
            return
        except Exception as e:
            print(e)
            pass

for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
    data = []
    examples_hf = load_dataset('xlangai/BRIGHT', 'examples')[task]
    examples = {}
    for e in examples_hf:
        examples[e['id']] = e
    documents_hf = load_dataset('xlangai/BRIGHT', 'documents')[task]
    documents = {}
    for d in documents_hf:
        documents[d['id']] = d['content']
    with open(f"../0617/outputs/{task}_{retriever}_long_False/score.json") as f:
        scores = json.load(f)
    examples_hf = load_dataset('xlangai/BRIGHT', 'examples')[task]
    for e in examples_hf:
    # for task_file in os.listdir(f'qa_data/{task}'):
        # if not task_file.endswith('.json'):
        #     continue
        # with open(os.path.join(f'qa_data/{task}',task_file)) as f:
        #     e = json.load(f)
        eid = e['id']
        cur_scores = sorted(scores[eid].items(),key=lambda x:x[1],reverse=True)[:10]
        selected_ids = [doc_score[0] for doc_score in cur_scores]
        cur_documents = []
        for doc_id in selected_ids:
            cur_documents.append(documents[doc_id])
        # assert len(e["content"])>=10
        data.append({
            'id': e["id"],
            'query': e['query'],
            'gold': e["gold_answer"],
            'task':task,
            'documents': cur_documents
        })
    if not os.path.isdir(f'outputs/{retriever}_retrieval/{task}'):
        os.makedirs(f'outputs/{retriever}_retrieval/{task}')
    with mp.Pool(64) as pool, tqdm(total=len(data), desc=task) as pbar:
        for return_contents in pool.imap_unordered(worker, data):
            pbar.update()