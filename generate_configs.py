import json
import os.path

# ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living','leetcode','pony','aops','theoremqa']

for model in ['sf','qwen','e5']:
    if not os.path.isdir(f"configs/{model}"):
        os.makedirs(f"configs/{model}")
    for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
        instructions = {
            'instructions': {
                'query': 'Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: '
            },
            'instructions_long': {
                'query': 'Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: '
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)
    instructions = {
        'instructions': {
            'query': 'Instruct: Given a {task} question, retrieve relevant passages that help answer the question\nQuery: '
        },
        'instructions_long': {
            'query': 'Instruct: Given a {task} question, retrieve relevant documents that help answer the question\nQuery: '
        },
    }
    with open(f"configs/{model}/pony.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    instructions = {
        'instructions': {
            'query': 'Instruct: Given a coding problem, retrieve relevant examples that help answer the problem\nQuery: '
        },
    }
    with open(f"configs/{model}/leetcode.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    for task in ['aops','theoremqa']:
        instructions = {
            'instructions': {
                'query': 'Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery: '
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)


for model in ['grit']:
    if not os.path.isdir(f"configs/{model}"):
        os.makedirs(f"configs/{model}")
    for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
        instructions = {
            'instructions': {
                'query': '<|user|>\nGiven a {task} post, retrieve relevant passages that help answer the post\n<|embed|>\n',
                'document': "<|embed|>\n"
            },
            'instructions_long': {
                'query': '<|user|>\nGiven a {task} post, retrieve relevant documents that help answer the post\n<|embed|>\n',
                'document': "<|embed|>\n"
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)
    instructions = {
        'instructions': {
            'query': '<|user|>\nGiven a {task} question, retrieve relevant passages that help answer the question\n<|embed|>\n',
            'document': "<|embed|>\n"
        },
        'instructions_long': {
            'query': '<|user|>\nGiven a {task} question, retrieve relevant documents that help answer the question\n<|embed|>\n',
            'document': "<|embed|>\n"
        },
    }
    with open(f"configs/{model}/pony.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    instructions = {
        'instructions': {
            'query': '<|user|>\nGiven a coding problem, retrieve relevant examples that help answer the problem\n<|embed|>\n',
            'document': "<|embed|>\n"
        },
    }
    with open(f"configs/{model}/leetcode.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    for task in ['aops','theoremqa']:
        instructions = {
            'instructions': {
                'query': '<|user|>\nGiven a Math problem, retrieve relevant examples that help answer the problem\n<|embed|>\n',
                'document': "<|embed|>\n"
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)

for model in ['inst-l','inst-xl']:
    if not os.path.isdir(f"configs/{model}"):
        os.makedirs(f"configs/{model}")
    for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
        instructions = {
            'instructions': {
                'query': "Represent the {task} post for retrieving relevant paragraphs: ",
                'document': "Represent the {task} paragraph for retrieval: "
            },
            'instructions_long': {
                'query': "Represent the {task} post for retrieving relevant documents: ",
                'document': "Represent the {task} document for retrieval: "
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)
    instructions = {
        'instructions': {
            'query': "Represent the Pony question for retrieving relevant paragraphs: ",
            'document': "Represent the Pony paragraph for retrieval: "
        },
        'instructions_long': {
            'query': "Represent the Pony question for retrieving relevant documents: ",
            'document': "Represent the Pony document for retrieval: "
        },
    }
    with open(f"configs/{model}/pony.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    instructions = {
        'instructions': {
            'query': "Represent the Coding problem for retrieving relevant examples: ",
            'document': "Represent the Coding example for retrieval: "
        },
    }
    with open(f"configs/{model}/leetcode.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    for task in ['aops','theoremqa']:
        instructions = {
            'instructions': {
                'query': "Represent the Math problem for retrieving relevant examples: ",
                'document': "Represent the Math example for retrieval: "
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)

for model in ['bge']:
    if not os.path.isdir(f"configs/{model}"):
        os.makedirs(f"configs/{model}")
    for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living']:
        instructions = {
            'instructions': {
                'query': "Represent this {task} post for searching relevant passages: ",
            },
            'instructions_long': {
                'query': "Represent this {task} post for searching relevant documents: ",
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)
    instructions = {
        'instructions': {
            'query': "Represent this Pony question for searching relevant passages: ",
        },
        'instructions_long': {
            'query': "Represent this Pony question for searching relevant documents: ",
        },
    }
    with open(f"configs/{model}/pony.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    instructions = {
        'instructions': {
            'query':  "Represent this Coding problem for searching relevant examples: ",
        },
    }
    with open(f"configs/{model}/leetcode.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    for task in ['aops','theoremqa']:
        instructions = {
            'instructions': {
                'query': "Represent this Math problem for searching relevant examples: ",
            },
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)

for model in ['bm25','sbert','openai','cohere','voyage','google']:
    if not os.path.isdir(f"configs/{model}"):
        os.makedirs(f"configs/{model}")
    for task in ['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living','leetcode','pony','aops','theoremqa']:
        instructions = {
            'instructions': {},
            'instructions_long': {},
        }
        with open(f"configs/{model}/{task}.json",'w') as f:
            json.dump(instructions,f,indent=2)