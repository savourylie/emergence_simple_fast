# Simple RAG for 79% on LongMemEval

A minimal retrieval-augmented generation (RAG) script using `sentence-transformers` and OpenAI's GPT models. This repo evaluates the ability of LLMs to recall relevant information from long-term memory, using the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark.

## Features

- Uses `all-MiniLM-L6-v2` for embedding retrieval
- Summarizes relevant context to answer questions using GPT-4o
- Evaluates responses using a reference-based prompt-checking approach

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/EmergenceAI/emergence_simple_fast.git
cd emergence_simple_fast
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Download the dataset

Download the LongMemEval dataset by following the instructions in the [LongMemEval repo](https://github.com/xiaowu0162/LongMemEval?tab=readme-ov-file#data).

Create a directory called `data` 

```
mkdir data
```

and place the file `longmemeval_s.json` there. 


### 4. Set your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY=your-api-key
```

### 5. Run the generation/evaluation script

```
python main.py
```

This should take well under an hour, for most setups.

## Requirements

- Python 3.8+
- OpenAI API key
- GPU recommended but not required
