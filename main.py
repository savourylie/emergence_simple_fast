from sentence_transformers import SentenceTransformer, util
import json, textwrap, numpy as np, openai, time, os
from tqdm import tqdm

# We use a simple local embedding model.  Depending on your GPU setup, your latency may vary.
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
client = openai.OpenAI()

def callgpt(messages, model:str, max_tokens:int):
    """Wrapper to call the LLM."""
    response = client.chat.completions.create(model=model, messages=messages, temperature=0., max_tokens=max_tokens)
    return response.choices[0].message.content

def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    """Preprocess and encode haystack turns with dates."""
    all_turns = []
    for session, date in zip(haystack_sessions, haystack_dates):
        for turn in session:
            if 'role' in turn and 'content' in turn:
                all_turns.append(f"[{date}] {turn['role']}: {turn['content']}")
    corpus_embeddings = retrieval_model.encode(all_turns, convert_to_tensor=True)
    return corpus_embeddings, all_turns

def process_question(memstruct, question: str, question_date: str) -> str:
    top_k = 42 # Decrease for faster, but less accurate answers.
    corpus_embeddings, all_turns = memstruct
    query_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    # Step 1: Retrieve top-k relevant turns
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    retrieved_turns = [all_turns[hit['corpus_id']] for hit in hits]
    # Step 2: Extract structured facts from retrieved turns (fact-level)
    summary_prompt = f"""
    You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
    1. Identify key events, dates, quantities, or named entities.
    2. Extract only information relevant to the question.
    3. Write the facts in structured bullet points.
    \nQuestion: {question}
    \nMessages:
    {json.dumps(retrieved_turns, indent=2)}
    \nNow extract the structured facts:
    -
    """
    summary_prompt = textwrap.dedent(summary_prompt)
    facts = callgpt([{"role": "system", "content": summary_prompt}], model="gpt-4o", max_tokens=512)
    # Step 3: Combine both facts and raw turns to answer
    answer_prompt = f"""
    You are a helpful assistant. Using both the extracted facts and the original conversation turns below,
    answer the question as accurately and concisely as possible.
    \nExtracted Facts:
    {facts}
    \nRetrieved Conversation Turns:
    {json.dumps(retrieved_turns, indent=2)}
    \nQuestion: {question}
    Question Date: {question_date}
    Answer:
    """
    answer_prompt = textwrap.dedent(answer_prompt)
    answer = callgpt([{"role": "system", "content": answer_prompt}], model="gpt-4o", max_tokens=256)
    return answer.strip()

################################################################ Evaluation code.
# Much below here is adapted from LongMemEval https://github.com/xiaowu0162/LongMemEval
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else: raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt

class Evaluator:
    model_zoo = {'gpt-4o-mini': 'gpt-4o-mini-2024-07-18', 'gpt-4o': 'gpt-4o-2024-08-06'}
    def __init__(self, haystacks:list[dict]):
        references = [{k: haystack[k] for k in ['answer', 'question_id', 'question', 'question_type']} for haystack in haystacks]
        self.metric_model = self.model_zoo['gpt-4o']
        self.qid2qdata = {entry['question_id']: entry for entry in references}
        self.qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
        self.qtypes = set(list(self.qid2qtype.values()))
    def evaluate(self, entry: dict) -> bool:
        assert entry['question_id'] in self.qid2qtype, 'question_id not in reference data'
        qtype = self.qid2qtype[entry['question_id']]
        q = self.qid2qdata[entry['question_id']]['question']
        ans = self.qid2qdata[entry['question_id']]['answer']
        hyp = entry['hypothesis']
        prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
        eval_response = callgpt(messages=[{"role": "user", "content": prompt}], model=self.metric_model, max_tokens=10)
        label = 'yes' in eval_response.lower()
        return label

class Stopwatch:
    def __init__(self): self._start = None
    def start(self): self._start = time.perf_counter()
    def stop(self): return time.perf_counter() - self._start

def evaluate_qa(hypotheses:list[dict], evaluator:Evaluator):
    qtype2acc = {t: [] for t in evaluator.qtypes}
    logs = []
    for entry in hypotheses:
        label = entry['label']
        logs.append(label)
        qtype2acc[evaluator.qid2qtype[entry['question_id']]].append(1 if label else 0)
    print('Accuracy:', round(np.mean(logs).item(), 4))
    for k,v in sorted(qtype2acc.items()):
        print(f'\t{k:<27}: {round(np.mean(v), 4):>6.2%} ({len(v)} obs)')

def main():
    assert os.path.exists('./data/longmemeval_s.json'), 'Please download the LongMemEval dataset and place it in the ./data directory.'
    haystacks = json.load(open('./data/longmemeval_s.json'))
    evaluator = Evaluator(haystacks)
    # Make the predictions
    num_success, nobs = 0, 0
    stopwatch = Stopwatch()
    hypotheses, haystack_time, question_time = [], [], []
    for haystack in tqdm(haystacks):
        question_id = haystack['question_id']
        question = haystack['question']
        question_date = haystack['question_date']
        haystack_dates = haystack['haystack_dates']
        haystack_sessions = haystack['haystack_sessions']
        stopwatch.start() # Time the haystack latency.
        memstruct = process_haystack(haystack_sessions, haystack_dates)
        haystack_time.append(stopwatch.stop())
        stopwatch.start() # Time the question processing latency.
        guess = process_question(memstruct, question, question_date)
        question_time.append(stopwatch.stop())
        hypothesis = {'question_id': question_id, 'hypothesis': guess}
        result = evaluator.evaluate(hypothesis)
        hypothesis['label'] = result
        hypotheses.append(hypothesis)
        num_success += result
        nobs += 1
    # Print the results
    print(f'Evaluated {nobs} hypotheses with {num_success} successes.  Accuracy: {num_success / nobs:.4f}')
    evaluate_qa(hypotheses, evaluator)
    print(f"Haystack time median: {np.median(haystack_time):.4f} seconds")
    print(f"Question time median: {np.median(question_time):.4f} seconds")

if __name__ == '__main__': main()
