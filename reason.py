import json
import argparse
import time
from datasets import load_dataset
from tqdm import tqdm


class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def generate(self, prompt):
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
        return response['content'][0]['text']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--llm', type=str, required=True)
    args = parser.parse_args()
    examples = load_dataset('xlangai/BRIGHT', 'examples')[args.task]

    with open(args.output_file, 'w') as f:
        json.dump('hello', f, indent=2)

    if 'claude' in args.llm:
        model = ClaudeModel(version=args.llm)
    else:
        raise ValueError(f'Model {args.llm} has not been implemented')

    rewritten_examples = []
    for e in tqdm(examples):
        cur_post = e["query"].replace('\n', ' ')
        prompt = (f'{cur_post}\n\n'
                  f'Instructions:\n'
                  f'1. Identify the essential problem.\n'
                  f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
                  f'3. Draft an answer with as many thoughts as you have.\n')

        exec_count = 0
        success = False
        while not success:
            if exec_count > 5:
                print('Fail after trying for 5 times')
                break
            exec_count += 1
            try:
                e['query'] = model.generate(prompt)
                success = True
            except:
                time.sleep(5)
        rewritten_examples.append(e)
    with open(args.output_file, 'w') as f:
        json.dump(rewritten_examples, f, indent=2)



