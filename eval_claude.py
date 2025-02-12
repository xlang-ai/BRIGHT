import os
import json
from tqdm import tqdm
from anthropic import AnthropicVertex
import multiprocessing as mp

retriever = 'qwen'
PROJECT_ID = ""
LOCATION = ""
claude_client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)

criteria = '''0 - The student's answer is completely irrelevant or blank.
10 - The student's answer addresses about 10% of the reference content.
20 - The student's answer addresses about 20% of the reference content.
30 - The student's answer addresses about 30% of the reference content.
40 - The student's answer addresses about 40% of the reference content.
50 - The student's answer addresses about 50% of the reference content.
60 - The student's answer addresses about 60% of the reference content.
70 - The student's answer addresses about 70% of the reference content.
80 - The student's answer addresses about 80% of the reference content.
90 - The student's answer addresses about 90% of the reference content.
100 - The student's answer addresses about 100% of the reference content.'''


def worker(arg):
    exec_count = 0
    completion = None
    while exec_count<50:
        try:
            exec_count += 1
            message = claude_client.messages.create(
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"---------- PROBLEM START ----------\n"
                                   f"{arg['query'].strip()}\n"
                                   f"---------- PROBLEM END ----------\n"
                                   f"---------- STUDENT ANSWER START ----------\n"
                                   f"{arg['pred'].strip()}\n"
                                   f"---------- STUDENT ANSWER END ----------\n"
                                   f"---------- REFERENCE ANSWER START ----------\n"
                                   f"{arg['gold'].strip()}\n"
                                   f"---------- REFERENCE ANSWER END ----------\n\n"
                                   f"Criteria:\n{criteria}\n\n"
                                   f"Use the following format to give a score:\n"
                                   f"REASON:\n"
                                   f"Describe why you give a specific score\n"
                                   f"SCORE:\n"
                                   f"The score you give, e.g., 60\n"
                                   f"Do not say anything after the score.",
                    }
                ],
                system="You are a teacher to judge student's answer",
                model="claude-3-5-sonnet@20240620",
                temperature=0,
            )
            response = json.loads(message.model_dump_json(indent=2))
            completion = response['content'][0]['text']
            score = float(completion.split('SCORE')[-1].replace(':','').strip())
            return score
        except Exception as e:
            # print('completion:',completion)
            pass

for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
    data = []
    for output_file in os.listdir(f"outputs/{retriever}_retrieval/{task}"):
        if not output_file.endswith('.json'):
            continue
        with open(os.path.join(f"outputs/{retriever}_retrieval/{task}",output_file)) as f:
            r = json.load(f)
        data.append({
            'id': r['id'],
            'pred': r['pred'],
            'gold': r['gold'],
            'query': r['query']
        })
    scores = []
    with mp.Pool(64) as pool, tqdm(total=len(data), desc=task) as pbar:
        for return_score in pool.imap_unordered(worker, data):
            # print(return_score)
            scores.append(return_score)
            pbar.update()
    print(task,sum(scores)/len(scores))