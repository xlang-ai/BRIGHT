import json
import argparse
import time
import os
from datasets import load_dataset
from tqdm import tqdm
import functools

from retrievers import calculate_retrieval_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def call_api(func):
    count = 0 
    while True:
        try:
            count += 1
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                logger.info("Rate limit exceeded, waiting 10 secs and retrying...")
                time.sleep(10)
            elif count < 5:
                logger.info("Encountered error, retrying...")
                time.sleep(5)
            else:
                logger.info("Skipping generation due to unknown error after 5 retries.")
                output = None
                break
    return output


def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    if include_system:
        chat = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def generate(self, prompt):
        inputs = format_chat(prompt, include_system=False)
        func = functools.partial(
            self.model.messages.create, 
            max_tokens=2048, 
            messages=inputs, 
            model=self.version, 
            temperature=0.8, 
            top_p=0.8
        )
        message = call_api(func)
        if message is not None:
            response = json.loads(message.model_dump_json(indent=2))
            return response['content'][0]['text']
        return None
    

class OpenAIModel:
    def __init__(self, model_name, temperature=0.8, top_p=0.8):
        import openai
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION 
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = 2048

    def generate(self, prompt, system_message="You are a helpful assistant", **kwargs):
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        inputs = format_chat(prompt, system_message=system_message)
        func = functools.partial(
            self.model.chat.completions.create, 
            model=self.model_name, 
            messages=inputs, 
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            return output.choices[0].message.content
        return None


class HFModel:
    def __init__(self, model_name, temperature, top_p):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = 2048
    
    def generate(self, message, **kwargs):
        inputs = self.tokenizer([message], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs,
        )
        text = self.tokenizer.decode(outputs[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--example_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--llm', type=str, required=True)
    args = parser.parse_args()

    if args.example_file is not None:
        # supports json and jsonl files
        examples = load_dataset("json", data_files=args.example_file)["train"]
    else:
        examples = load_dataset('xlangai/BRIGHT', 'examples')[args.task]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if 'claude' in args.llm:
        model = ClaudeModel(version=args.llm)
    elif 'gpt' in args.llm:
        model = OpenAIModel(model_name=args.llm)
    else:
        logger.info(f"Assuming Hugging Face model: {args.llm}")
        model = HFModel(model_name=args.llm)

    rewritten_examples = []
    for e in tqdm(examples):
        cur_post = e["query"].replace('\n', ' ')
        prompt = (f'{cur_post}\n\n'
                  f'Instructions:\n'
                  f'1. Identify the essential problem.\n'
                  f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
                  f'3. Draft an answer with as many thoughts as you have.\n')
        output = model.generate(prompt)
        if output is not None:
            e['query'] = output
        rewritten_examples.append(e)

    logger.info(f"Saving rewritten examples to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(rewritten_examples, f, indent=2)
