import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

metrics = ["NDCG", "Recall", "Precision"]
ks = [1, 10, 100]
metrics = [f"{m}@{k}" for m in metrics for k in ks] + ["MRR"]
metrics = ["NDCG@10"]

models = [
    'bm25',
    "bge",
    "inst-l",
    "sbert",
    "e5",
    "sf",
    "inst-xl",
    "grit",
    "qwen",
    #"qwen2",
    'openai',
]
datasets = ['aops', 'theoremqa_questions', 'theoremqa_theorems']

results = []
for d in datasets:
    for m in models:
        path = f"outputs/{d}_{m}_long_False/results.json"
        if not os.path.exists(path):
            print('missing', path)
            continue
        with open(path) as f:
            r = json.load(f)["NDCG@10"]
            results.append({'dataset': d, 'model': m, 'value': r})

df = pd.DataFrame(results)
df[df.select_dtypes(include=['number']).columns] *= 100
df = df.pivot_table(index=['model'], columns='dataset', values='value', sort=False)
print(df.to_csv())
print(df.to_latex(float_format="{:.1f}".format))


import re
from statistics import mean

import re

def process_table(input_text):
    # Split into lines
    lines = input_text.strip().split('\n')

    # Store all numeric rows for column analysis
    numeric_rows = []

    # Process each line
    processed_lines = []
    for line in lines:
        # Skip lines without numbers or with \midrule or \multicolumn
        if '\\midrule' in line or '\\multicolumn' in line:
            processed_lines.append(line)
            continue

        # Check if line contains numbers
        numbers = re.findall(r'(?<= & )([-]?\d+\.\d+)(?= \\\\| & )', line)
        if not numbers:
            processed_lines.append(line)
            continue

        # Store numbers for column analysis
        numeric_rows.append(numbers)

        # Calculate average of all but last number
        numbers_except_last = [float(x) for x in numbers[:-1]]
        avg = sum(numbers_except_last) / len(numbers_except_last)

        # Replace the last number with the average
        base_parts = line.split(' & ')[:-1]
        processed_line = ' & '.join(base_parts) + f' & {avg:.1f} \\\\'
        processed_lines.append(processed_line)

    # Find largest and second largest in each column
    if numeric_rows:
        num_cols = len(numeric_rows[0])
        column_values = []

        # Transpose the data to analyze by column
        for col in range(num_cols):
            column = [float(row[col]) for row in numeric_rows]
            column_values.append(sorted([(val, idx) for idx, val in enumerate(column)], reverse=True))

        # Apply bold and underline formatting
        final_lines = []
        for line in processed_lines:
            if '\\midrule' in line or '\\multicolumn' in line:
                final_lines.append(line)
                continue

            numbers = re.findall(r'(?<= & )([-]?\d+\.\d+)(?= \\\\| & )', line)
            if not numbers:
                final_lines.append(line)
                continue

            # Find the row index for this line
            row_idx = next(i for i, row in enumerate(numeric_rows) if row[0] == numbers[0])

            # Process each number
            parts = line.split(' & ')
            new_parts = [parts[0]]

            for col, num in enumerate(numbers):
                num_float = float(num)
                if col < len(column_values):
                    if row_idx == column_values[col][0][1]:  # Largest
                        new_parts.append(f'\\textbf{{{num}}}')
                    elif row_idx == column_values[col][1][1]:  # Second largest
                        new_parts.append(f'\\underline{{{num}}}')
                    else:
                        new_parts.append(num)

            final_lines.append(' & '.join(new_parts) + ' \\\\')

        return '\n'.join(final_lines)

    return '\n'.join(processed_lines)

# Test the function
input_text = """\\bmtwentyfive & 19.2 & 27.1 & 14.9 & 12.5 & 13.5 & 16.5 & 15.2 & 24.4 & 7.9 & 6.0 & 13.0 & 6.9 & 14.3\\
\\bge & 12.0 & 24.2 & 16.6 & 17.4 & 12.2 & 9.5 & 13.3 & 26.7 & 5.6 & 6.0 & 13.0 & 6.9 & 13.6 \\
\\instructorL & 15.6 & 21.5 & 16.0 & 21.9 & 11.5 & 11.2 & 13.2 & 20.0 & 1.3 & 8.1 & 20.9 & 9.1 & 14.0 \\
\\sentencebert & 15.5 & 20.1 & 16.6 & 22.6 & 8.4 & 9.5 & 15.3 & 26.4 & 6.9 & 5.3 & 20.0 & 10.8 & 14.6 \\
\\efive & 18.8 & 26.0 & 15.5 & 15.8 & 16.4 & 9.8 & 18.5 & 28.7 & 4.8 & 7.1 & 26.1 & 26.8 & 17.5 \\
\\sfr & 19.5 & 26.6 & 17.8 & 19.0 & 16.7 & 12.7 & \\underline{19.8} & 27.4  & 2.0 &  7.4 & 24.3 & 26.0 & 18.0 \\
\\instructorXL & 21.9 & \\underline{34.4} & \textbf{22.8} & 27.4 & \textbf{17.4} & \\underline{19.1} & 18.8 & 27.5 & 5.0 & 8.5 & 15.6 & 5.9 & 18.6 \\
\\grit & \\underline{25.0} & 32.8 & 19.0 & 19.9 & \\underline{17.3} & 11.6 & 18.0 & \\underline{29.8} & \textbf{22.0} & 8.8 & 25.1 & 21.1 & \\underline{20.6} \\
\\qwen & \textbf{30.9} & \textbf{36.2} & 17.7 & 24.6 & 13.5 & \textbf{19.9} & 14.9 & 25.5 & 14.4 & 27.8 & 32.9 & 32.9  & \textbf{22.1} \\
\\cohere & 19.0 & 27.5 & \\underline{20.2} & 21.8 & 16.2 & 16.5 & 17.7 & 26.8 & 1.8 & 6.5 & 15.1 & 7.1 & 16.3 \\
\\openai & 23.7 & 26.3 & 20.0 & \\underline{27.5} & 12.9 & 12.5 & \textbf{20.3} & 23.6 & 2.5 &  8.5 & 23.8 & 12.3 & 17.6 \\
\\voyage & 23.6 & 25.1 & 19.8 & 24.8 & 11.2 & 15.0 & 15.6 & \textbf{30.6} & 1.5 & 7.4 & \\underline{26.1} & 11.1 & 17.6 \\
\\google & 23.0 & \\underline{34.4} & 19.5 & \textbf{27.9} & 16.0 & 17.9 & 17.3 & 29.6 & 3.6 & \\underline{9.3} & 21.5 & 14.3 & 19.5 \\"""

df = []
for x in input_text.split("\n"):
    x = x.replace("\textbf{", "")
    x = x.replace("\\underline{", "")
    x = x.replace("}", "")
    x = x.split("&")
    x = x[:-1]
    x = [float(i) if "." in i else i for i in x]
    df.append({i: v for i, v in enumerate(x)})
    df[-1][13] = np.mean(x[1:])
df = pd.DataFrame(df)
print(df.columns)
print(df)

for row in df.iterrows():
    x = []
    for k, v in row[1].items():
        cols = sorted(df[k], reverse=True)
        if k == 0:
            x.append(v)
        else:
            if v == cols[0]:
                x.append("\\underline{" + f"{v:.1f}" + "}")
            elif v == cols[1]:
                x.append("\\underline{" + f"{v:.1f}" + "}")
            else:
                x.append(f"{v:.1f}")
    print(" & ".join(x) + "\\\\")

