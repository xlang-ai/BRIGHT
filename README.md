# BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval

<p align="center">
    <img src="figures/figure1.png" width="85%" alt="Overview of BRIGHT benchmark">
</p>

This repository contains the code for our paper BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.

BRIGHT is the first text retrieval benchmark that requires intensive reasoning to retrieve relevant documents. 
The queries are collected from diverse domains (StackExchange, LeetCode, and math competitions), all sourced from realistic human data.
Experiments show that existing retrieval models perform poorly on BRIGHT, where the highest score is only 21 measured by nDCG@10.
BRIGHT provides a good testbed for future retrieval research in more realistic and challenging settings.

## Installation
In your local machine, we recommend to first create a virtual environment:
```bash
conda create -n bright python=3.10
conda activate bright
git clone https://github.com/xlang-ai/BRIGHT
cd BRIGHT
pip install -r requirements.txt
```
That will create the environment bright with all the required packages installed.

## Data
BRIGHT comprises 11 diverse datasets, spanning biology, economics, robotics, math, code and more. 
The queries can be long StackExchange posts, math or code question. 
The documents can be blogs, news, articles, reports, etc.
See [Huggingface page](https://huggingface.co/datasets/xlangai/BRIGHT) for more details.

## Evaluation
We evaluate 13 representative retrieval models of diverse sizes and architectures. Run the following command to get results:
```
python run.py --task {task} --model {model}
```
* `--task`: the task/dataset to evaluate. It can take one of `biology`,`earth_science`,`economics`,`psychology`,`robotics`,`stackoverflow`,`sustainable_living`,`leetcode`,`pony`,`aops`,`theoremqa`, 
* `--model`: the model to evaluate. Current implementation supports `bm25`,`cohere`,`e5`,`google`,`grit`,`inst-l`,`inst-xl`,`openai`,`qwen`,`sbert`,`sf`,`voyage` and `bge`. \
Optional:
* `--long_context`: whether to evaluate on the long-context setting, default to `False`
* `--query_max_length`: the maximum length for the query
* `--doc_max_length`: the maximum length for the document
* `--encode_batch_size`: the encoding batch size
* `--output_dir`: the directory to output results
* `--cache_dir`: the directory to cache document embeddings
* `--config_dir`: the directory of instruction configurations
* `-checkpoint`: the specific checkpoint to use
* `--key`: key for proprietary models
* `--debug`: whether to turn on the debug mode and load only a few documents

### Add custom model?
It is very easy to add evaluate custom models on BRIGHT. Just implement the following function in `retrievers.py` and add it to the mapping `RETRIEVAL_FUNCS`:
```python
def retrieval_model_function_name(queries,query_ids,documents,doc_ids,excluded_ids,**kwargs):
    ...
    return scores
```
where `scores` is in the format:
```bash
{
  "query_id_1": {
    "doc_id_1": score_1,
    "doc_id_2": score_2,
    ...
    "doc_id_n": socre_n
  },
  ...
  "query_id_m": {
    "doc_id_1": score_1,
    "doc_id_2": score_2,
    ...
    "doc_id_n": socre_n
  }
}
```

## Bugs or questions?
If you have any question related to the code or the paper, feel free to email Hongjin (hjsu@cs.hku.hk), Howard (hyen@cs.princeton.edu) or Mengzhou (mengzhou@cs.princeton.edu). Please try to specify the problem with details so we can help you better and quicker.

## Citation
If you find our work helpful, please cite us:
```citation
@misc{BRIGHT,
  title={BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O and Chen, Danqi and Yu, Tao},
  year={2024},
}
```


