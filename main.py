from sentence_transformers import SentenceTransformer, util
import torch
import json
import textwrap
import numpy as np
import time
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import plotext as plt
import seaborn as sns
import matplotlib.pyplot as mpl_plt
import pandas as pd

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.utils import DataLoadingConfig, load_benchmark_dataset

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Dataset-specific settings (matching run_benchmark.py)
DATASET_CONFIGS = {
    "msc": {
        "top_k": 15,
        "model_name": "openai/gpt-4o-mini"
    },
    "lme": {
        "top_k": 42,
        "model_name": "openai/gpt-4o-mini"
    },
    "locomo": {
        "top_k": 50,
        "model_name": "openai/gpt-4o-mini"
    }
}

# Setup logging
def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

    return logging.getLogger(__name__)

def count_tokens(text):
    """Simple token counting using whitespace splitting (fast approximation)."""
    return len(text.split())

def count_tokens_list(texts):
    """Count tokens for a list of texts."""
    return [count_tokens(text) for text in texts]

# We use a simple local embedding model.  Depending on your GPU setup, your latency may vary.
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenRouter client
client = OpenAI()


def callgpt(messages, model: str, max_tokens: int, logger=None):
    """Wrapper to call the LLM via OpenRouter with error handling."""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            if logger:
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            if logger:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")

            if attempt == max_retries - 1:
                if logger:
                    logger.error(f"All API call attempts failed: {str(e)}")
                raise

            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

    return None


def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str], logger=None):
    """Preprocess and encode haystack turns with dates."""
    try:
        all_turns = []
        for session, date in zip(haystack_sessions, haystack_dates):
            for turn in session:
                if 'role' in turn and 'content' in turn:
                    all_turns.append(f"[{date}] {turn['role']}: {turn['content']}")

        if logger:
            logger.debug(f"Processing {len(all_turns)} turns for embedding")

        corpus_embeddings = retrieval_model.encode(
            all_turns, convert_to_tensor=True)

        if logger:
            logger.debug(f"Generated embeddings with shape {corpus_embeddings.shape}")

        return corpus_embeddings, all_turns

    except Exception as e:
        if logger:
            logger.error(f"Error processing haystack: {str(e)}")
        raise


def top_p_selection(scores, candidates, p=0.9, min_k=5, max_tokens=20000):
    """Select candidates using top-p (nucleus) sampling with token-based capping."""
    scores_np = np.array(scores)
    
    # Apply softmax to similarity scores for probability distribution
    # Subtract max for numerical stability
    exp_scores = np.exp(scores_np - np.max(scores_np))
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Sort candidates by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    
    # Find cutoff point where cumulative probability >= p
    cumulative_prob = np.cumsum(sorted_probs)
    cutoff_idx = np.argmax(cumulative_prob >= p) + 1
    
    # Ensure minimum number of items (safety bound)
    cutoff_idx = max(cutoff_idx, min_k)
    
    # Apply token-based capping
    selected_indices = []
    selected_candidates = []
    selected_scores = []
    total_tokens = 0
    
    for i in range(min(cutoff_idx, len(sorted_indices))):
        idx = sorted_indices[i]
        candidate = candidates[idx]
        candidate_tokens = count_tokens(candidate)
        
        # Check if adding this candidate would exceed token limit
        if total_tokens + candidate_tokens > max_tokens and len(selected_candidates) >= min_k:
            break
            
        selected_indices.append(idx)
        selected_candidates.append(candidate)
        selected_scores.append(scores[idx])
        total_tokens += candidate_tokens
    
    return selected_candidates, selected_scores, len(selected_candidates), total_tokens


def greedy_mmr(query_embedding, candidates, candidate_embeddings, scores, lambda_param=0.65, max_tokens=10000):
    """Greedy Maximum Marginal Relevance for de-duplication with token-based capping."""
    if not candidates:
        return [], [], 0, 0
    
    selected = []
    selected_embeddings = []
    remaining_indices = list(range(len(candidates)))
    total_tokens = 0
    
    # Select first item (highest relevance)
    best_idx = np.argmax(scores)
    first_candidate = candidates[best_idx]
    first_tokens = count_tokens(first_candidate)
    
    selected.append(first_candidate)
    selected_embeddings.append(candidate_embeddings[best_idx])
    remaining_indices.remove(best_idx)
    total_tokens += first_tokens
    
    # Iteratively select items with MMR (token-based stopping)
    while remaining_indices:
        mmr_scores = []
        candidate_tokens_list = []
        
        for idx in remaining_indices:
            candidate = candidates[idx]
            candidate_tokens = count_tokens(candidate)
            candidate_tokens_list.append(candidate_tokens)
            
            # Skip if this candidate alone would exceed token limit
            if total_tokens + candidate_tokens > max_tokens:
                mmr_scores.append(-float('inf'))  # Never select
                continue
            
            # Relevance score (already computed)
            relevance = scores[idx]
            
            # Diversity score (maximum similarity to already selected)
            max_similarity = 0.0
            if selected_embeddings:
                current_emb = candidate_embeddings[idx]
                
                # Batch compute similarities for efficiency
                selected_stack = torch.stack(selected_embeddings)
                similarities = util.cos_sim(current_emb.unsqueeze(0), selected_stack)[0]
                max_similarity = torch.max(similarities).item()
            
            # MMR score: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append(mmr_score)
        
        # Check if any valid candidates remain
        if all(score == -float('inf') for score in mmr_scores):
            break
        
        # Select item with highest MMR score
        best_mmr_idx = np.argmax(mmr_scores)
        selected_idx = remaining_indices[best_mmr_idx]
        
        selected.append(candidates[selected_idx])
        selected_embeddings.append(candidate_embeddings[selected_idx])
        total_tokens += candidate_tokens_list[best_mmr_idx]
        remaining_indices.remove(selected_idx)
    
    return selected, selected_embeddings, len(selected), total_tokens


def process_question_mc(memstruct, question: str, choices: list[str], question_date: str, top_k=42, model_name=DEFAULT_MODEL, use_adaptive_k=False, logger=None) -> tuple:
    """Process a multiple-choice question using the memory structure."""
    try:
        corpus_embeddings, all_turns = memstruct

        if logger:
            logger.debug(f"Processing question: {question[:100]}...")

        query_embedding = retrieval_model.encode(question, convert_to_tensor=True)

        # Step 1: Retrieve relevant turns using selected method
        if use_adaptive_k:
            # Use larger initial retrieval for top-p filtering (increased from 50)
            initial_k = min(100, len(all_turns))  # Get more candidates for better top-p selection
            hits = util.semantic_search(
                query_embedding, corpus_embeddings, top_k=initial_k)[0]
            
            # Extract scores and candidates
            candidates = [all_turns[hit['corpus_id']] for hit in hits]
            candidate_embeddings = [corpus_embeddings[hit['corpus_id']].clone() for hit in hits]
            scores = [hit['score'] for hit in hits]
            
            # Calculate initial stage statistics
            initial_tokens = sum(count_tokens_list(candidates))
            
            # INVERTED PIPELINE: Apply MMR first on full candidate set
            mmr_candidates, _, mmr_k, mmr_tokens = greedy_mmr(
                query_embedding, candidates, candidate_embeddings, scores, lambda_param=0.65, max_tokens=15000
            )
            
            if logger:
                logger.debug(f"MMR selected {mmr_k} diverse items ({mmr_tokens} tokens) from {initial_k} candidates ({initial_tokens} tokens)")
                mmr_hit_limit = mmr_tokens >= 14800  # Near the 15000 limit
                logger.debug(f"MMR constraint: {'TOKEN LIMIT' if mmr_hit_limit else 'SIMILARITY THRESHOLD'}")
            
            # Then apply Top-P as quality filter on MMR results
            if mmr_candidates:
                # Get scores for MMR-selected candidates
                mmr_indices = [candidates.index(cand) for cand in mmr_candidates]
                mmr_scores = [scores[i] for i in mmr_indices]
                
                # Apply Top-P selection with token capping
                retrieved_turns, top_p_scores, final_k, final_tokens = top_p_selection(
                    mmr_scores, mmr_candidates, p=0.7, min_k=10, max_tokens=10000
                )
            else:
                retrieved_turns = []
                final_k = 0
                final_tokens = 0
                
            if logger:
                logger.debug(f"Top-P refined to {final_k} items ({final_tokens} tokens) as final quality filter")
                top_p_hit_limit = final_tokens >= 9800  # Near the 10000 limit
                logger.debug(f"Top-P constraint: {'TOKEN LIMIT' if top_p_hit_limit else 'NUCLEUS THRESHOLD (p=0.8)'}")
                reduction_ratio = final_k / mmr_k if mmr_k > 0 else 0
                logger.debug(f"Pipeline reduction: {reduction_ratio:.2%} items retained ({mmr_k} → {final_k})")
            
            # Store comprehensive statistics for this question
            retrieval_stats = {
                'initial_items': initial_k,
                'initial_tokens': initial_tokens,
                'top_p_items': mmr_k,  # MMR stage (was top_p_k)
                'top_p_tokens': mmr_tokens,  # MMR stage (was top_p_tokens)
                'final_items': final_k,
                'final_tokens': final_tokens,
                'efficiency_ratio': final_tokens / initial_tokens if initial_tokens > 0 else 0
            }
        else:
            # Traditional top-k retrieval
            hits = util.semantic_search(
                query_embedding, corpus_embeddings, top_k=top_k)[0]
            retrieved_turns = [all_turns[hit['corpus_id']] for hit in hits]
            final_k = len(retrieved_turns)
            final_tokens = sum(count_tokens_list(retrieved_turns))
            
            # Store basic statistics for traditional method
            retrieval_stats = {
                'initial_items': final_k,
                'initial_tokens': final_tokens,
                'top_p_items': final_k,
                'top_p_tokens': final_tokens,
                'final_items': final_k,
                'final_tokens': final_tokens,
                'efficiency_ratio': 1.0  # No filtering in traditional method
            }

        if logger:
            logger.debug(f"Retrieved {len(retrieved_turns)} relevant turns")

        # Step 2: Extract structured facts from retrieved turns (fact-level)
        summary_prompt = f"""
        You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
        1. Identify key events, dates, quantities, or named entities.
        2. Extract only information relevant to the question.
        3. Write the facts in structured bullet points.
        
Question: {question}
        
Messages:
        {json.dumps(retrieved_turns, indent=2)}
        
Now extract the structured facts:
        -
        """
        summary_prompt = textwrap.dedent(summary_prompt)

        if logger:
            logger.debug("Calling LLM for fact extraction")

        facts = callgpt(
            [{"role": "system", "content": summary_prompt}],
            model=model_name,
            max_tokens=512,
            logger=logger
        )

        # Step 3: Use facts and raw turns to select from multiple choices
        # Format choices for the prompt
        choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        
        answer_prompt = f"""
        You are a helpful assistant. Using both the extracted facts and the original conversation turns below,
        answer the multiple-choice question by selecting the correct option number (0-9).
        
Extracted Facts:
        {facts}
        
Retrieved Conversation Turns:
        {json.dumps(retrieved_turns, indent=2)}
        
Question: {question}
Question Date: {question_date}

Answer Choices:
{choices_text}

Please select the correct answer by responding with only the number (0-9) of the correct choice.
        """
        answer_prompt = textwrap.dedent(answer_prompt)

        if logger:
            logger.debug("Calling LLM for multiple choice selection")

        answer = callgpt(
            [{"role": "system", "content": answer_prompt}],
            model=model_name,
            max_tokens=10,
            logger=logger
        )

        if logger:
            logger.debug(f"Generated answer: {answer}")

        # Extract the number from the response  
        answer = answer.strip()
        try:
            choice_idx = int(answer)
            if 0 <= choice_idx < len(choices):
                return choice_idx, final_k, retrieval_stats
            else:
                if logger:
                    logger.warning(f"Invalid choice index {choice_idx}, defaulting to 0")
                return 0, final_k, retrieval_stats
        except ValueError:
            if logger:
                logger.warning(f"Could not parse choice from '{answer}', defaulting to 0")
            return 0, final_k, retrieval_stats

    except Exception as e:
        if logger:
            logger.error(f"Error processing question: {str(e)}")
        raise


def evaluate_mc_answer(predicted_idx: int, correct_idx: int) -> bool:
    """Evaluate multiple choice answer - simple index comparison."""
    return predicted_idx == correct_idx


class Stopwatch:
    def __init__(self): self._start = None
    def start(self): self._start = time.perf_counter()
    def stop(self): return time.perf_counter() - self._start


def save_intermediate_results(results, haystack_time, question_time, output_dir="results", dataset_name=None):
    """Save intermediate results to avoid data loss."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_suffix = f"_{dataset_name}" if dataset_name else ""

    # Save results
    results_file = Path(output_dir) / f"results{dataset_suffix}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save timing data
    timing_file = Path(output_dir) / f"timing{dataset_suffix}_{timestamp}.json"
    timing_data = {
        "haystack_time": haystack_time,
        "question_time": question_time,
        "total_processed": len(results)
    }
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)

    return results_file, timing_file


def load_intermediate_results(results_file):
    """Load intermediate results to resume processing."""
    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


def evaluate_qa_mc(results: list[dict], logger=None):
    """Evaluate multiple choice QA results with detailed logging."""
    # Group by question type
    qtype2results = {}
    for entry in results:
        qtype = entry['question_type']
        if qtype not in qtype2results:
            qtype2results[qtype] = []
        qtype2results[qtype].append(entry['correct'])

    # Calculate overall accuracy
    all_correct = [r['correct'] for r in results]
    accuracy = round(np.mean(all_correct).item(), 4)

    if logger:
        logger.info(f'Overall Accuracy: {accuracy}')
    else:
        print('Accuracy:', accuracy)

    # Per-question-type accuracy
    for k, v in sorted(qtype2results.items()):
        result = f'\t{k:<27}: {round(np.mean(v), 4):>6.2%} ({len(v)} obs)'
        if logger:
            logger.info(result)
        else:
            print(result)

    return accuracy


def create_comprehensive_visualizations(all_retrieval_stats, results, k_values, method_name, output_dir, dataset_name, timestamp, logger=None):
    """Create comprehensive seaborn visualizations and save as PNG files."""
    try:
        # Convert to DataFrame for easier analysis
        stats_df = pd.DataFrame(all_retrieval_stats)
        
        # Add accuracy information
        accuracy_map = {r['question_id']: r['correct'] for r in results}
        stats_df['correct'] = stats_df['question_id'].map(accuracy_map)
        
        # Set up the plotting style
        sns.set_style("whitegrid")
        mpl_plt.rcParams['figure.facecolor'] = 'white'
        
        # Create multi-panel figure
        fig, axes = mpl_plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{method_name} - Retrieval Analysis ({dataset_name.upper()})', fontsize=16, fontweight='bold')
        
        # Panel 1: Item count distribution by stage
        stage_data = []
        for _, row in stats_df.iterrows():
            stage_data.extend([
                {'stage': 'Initial', 'items': row['initial_items'], 'question_id': row['question_id']},
                {'stage': 'MMR', 'items': row['top_p_items'], 'question_id': row['question_id']},
                {'stage': 'Final (Top-P)', 'items': row['final_items'], 'question_id': row['question_id']}
            ])
        stage_df = pd.DataFrame(stage_data)
        
        sns.boxplot(data=stage_df, x='stage', y='items', ax=axes[0,0])
        axes[0,0].set_title('Item Count Distribution by Stage')
        axes[0,0].set_ylabel('Number of Items')
        
        # Panel 2: Token distribution by stage
        token_data = []
        for _, row in stats_df.iterrows():
            token_data.extend([
                {'stage': 'Initial', 'tokens': row['initial_tokens'], 'question_id': row['question_id']},
                {'stage': 'MMR', 'tokens': row['top_p_tokens'], 'question_id': row['question_id']},
                {'stage': 'Final (Top-P)', 'tokens': row['final_tokens'], 'question_id': row['question_id']}
            ])
        token_df = pd.DataFrame(token_data)
        
        sns.boxplot(data=token_df, x='stage', y='tokens', ax=axes[0,1])
        axes[0,1].set_title('Token Count Distribution by Stage')
        axes[0,1].set_ylabel('Number of Tokens')
        
        # Panel 3: Final K-distribution (traditional)
        unique_k, counts = np.unique(k_values, return_counts=True)
        axes[0,2].bar(unique_k, counts, alpha=0.7, color='skyblue')
        axes[0,2].set_title('Final K-Value Distribution')
        axes[0,2].set_xlabel('Number of Retrieved Items (K)')
        axes[0,2].set_ylabel('Frequency')
        
        # Panel 4: Option B - Accuracy vs Retrieval Efficiency
        stats_df['efficiency_ratio'] = stats_df['final_tokens'] / stats_df['initial_tokens']
        colors = ['red' if not correct else 'green' for correct in stats_df['correct']]
        scatter = axes[1,0].scatter(stats_df['efficiency_ratio'], stats_df['correct'], 
                                  c=colors, alpha=0.6, s=50)
        axes[1,0].set_title('Accuracy vs Retrieval Efficiency')
        axes[1,0].set_xlabel('Efficiency Ratio (Final Tokens / Initial Tokens)')
        axes[1,0].set_ylabel('Correct Answer (0/1)')
        axes[1,0].set_ylim(-0.1, 1.1)
        
        # Add trend line
        z = np.polyfit(stats_df['efficiency_ratio'], stats_df['correct'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(stats_df['efficiency_ratio'].sort_values(), 
                      p(stats_df['efficiency_ratio'].sort_values()), "b--", alpha=0.8)
        
        # Panel 5: Option C - Question Type Performance Analysis (Token Usage)
        if 'question_type' in stats_df.columns and len(stats_df['question_type'].unique()) > 1:
            sns.boxplot(data=stats_df, x='question_type', y='final_tokens', ax=axes[1,1])
            axes[1,1].set_title('Token Usage by Question Type')
            axes[1,1].set_ylabel('Final Tokens')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient question\ntype diversity', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Token Usage by Question Type')
        
        # Panel 6: Option C - Accuracy by Question Type
        if 'question_type' in stats_df.columns and len(stats_df['question_type'].unique()) > 1:
            accuracy_by_type = stats_df.groupby('question_type')['correct'].agg(['mean', 'count']).reset_index()
            bars = axes[1,2].bar(accuracy_by_type['question_type'], accuracy_by_type['mean'], 
                               alpha=0.7, color='lightcoral')
            axes[1,2].set_title('Accuracy Rate by Question Type')
            axes[1,2].set_ylabel('Accuracy Rate')
            axes[1,2].set_ylim(0, 1)
            axes[1,2].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, accuracy_by_type['count']):
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'n={count}', ha='center', va='bottom', fontsize=9)
        else:
            axes[1,2].text(0.5, 0.5, 'Insufficient question\ntype diversity', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Accuracy Rate by Question Type')
        
        # Adjust layout and save
        mpl_plt.tight_layout()
        
        # Save the figure
        viz_file = Path(output_dir) / f"retrieval_analysis_{dataset_name}_{timestamp}.png"
        mpl_plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
        mpl_plt.close()
        
        if logger:
            logger.info(f"Comprehensive visualizations saved to {viz_file}")
        
        return viz_file
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating visualizations: {str(e)}")
        return None




def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Emergence-style Multiple Choice Benchmark with OpenRouter API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "dataset", 
        choices=["msc", "lme", "locomo"],
        help="Dataset to benchmark (msc, lme, or locomo)"
    )

    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=10,
        help="Number of samples to process"
    )

    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Process entire dataset (overrides --num-samples)"
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Random sampling vs deterministic order"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for deterministic sampling"
    )

    parser.add_argument(
        "--stratified", "-s",
        action="store_true",
        help="Use stratified sampling to ensure all question types are represented"
    )

    parser.add_argument(
        "--top_k", "-k",
        type=int,
        help="Override default TOP_K value for memory retrieval"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--log_level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from intermediate results file"
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save intermediate results every N samples"
    )

    parser.add_argument(
        "--question-types",
        nargs="+",
        help="Filter by question types (LME only)"
    )

    parser.add_argument(
        "--adaptive-k",
        action="store_true",
        help="Use Adaptive-K with MMR instead of fixed top-k retrieval"
    )

    return parser.parse_args()


def main():
    """Main function with improved error handling and logging."""
    args = parse_arguments()

    # Validate question-types argument
    if args.question_types and args.dataset != "lme":
        print(f"Warning: --question-types is only supported for LME dataset, ignoring for {args.dataset}")
        args.question_types = None

    # Get dataset configuration
    config_settings = DATASET_CONFIGS[args.dataset]
    top_k = args.top_k if args.top_k else config_settings["top_k"]
    model_name = config_settings["model_name"]

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"run_{args.dataset}_{timestamp}.log"
    logger = setup_logging(args.log_level, log_file)

    logger.info(f"Starting {args.dataset.upper()} Emergence-style Multiple Choice processing")
    logger.info(f"Arguments: {vars(args)}")
    retrieval_method = "MMR + Top-P (λ=0.65, p=0.8)" if args.adaptive_k else f"Fixed Top-K (k={top_k})"
    logger.info(f"Retrieval Method: {retrieval_method}, Model: {model_name}")

    # Log question type filtering info for LME
    if args.dataset == "lme":
        if args.question_types:
            logger.info(f"Filtering questions by types: {', '.join(args.question_types)}")
        else:
            logger.info("Loading all question types (no filtering applied).")

    try:
        # Create configuration for dataset loading
        config = DataLoadingConfig(
            dataset_key=args.dataset,
            num_samples=args.num_samples if not args.load_all else 0,
            random_sampling=args.random or args.stratified,
            start_index=args.start_index,
            load_all=args.load_all,
            question_types=args.question_types
        )
        
        # Load dataset using the unified function
        haystacks = load_benchmark_dataset(config, logger)
        if not haystacks:
            logger.error("Failed to load dataset. Exiting.")
            return

        logger.info(f"Processing {len(haystacks)} samples")

        # Resume from previous run if specified
        results = []
        processed_ids = set()

        if args.resume:
            logger.info(f"Resuming from {args.resume}")
            results = load_intermediate_results(args.resume)
            processed_ids = {r['question_id'] for r in results}
            logger.info(f"Loaded {len(results)} previous results")

        # Initialize tracking variables
        num_success = sum(r['correct'] for r in results)
        nobs = len(results)
        stopwatch = Stopwatch()
        haystack_time, question_time = [], []
        k_values = []  # Track k values for each retrieval
        all_retrieval_stats = []  # Track comprehensive statistics

        # Process samples
        for i, haystack in enumerate(tqdm(haystacks, desc="Processing samples")):
            question_id = haystack['question_id']

            # Skip if already processed
            if question_id in processed_ids:
                continue

            try:
                question = haystack['question']
                question_date = haystack.get('question_date', '')
                choices = haystack['choices']
                correct_idx = haystack['correct_choice_index']
                question_type = haystack.get('question_type', 'unknown')
                
                # Handle dataset-specific fields
                if args.dataset == 'lme':
                    # LME has haystack_dates and haystack_sessions
                    haystack_dates = haystack['haystack_dates']
                    haystack_sessions = haystack['haystack_sessions']
                elif args.dataset == 'msc':
                    # MSC has haystack_sessions but no haystack_dates - create fake dates
                    haystack_sessions = haystack['haystack_sessions']
                    haystack_dates = [f"session_{i}" for i in range(len(haystack_sessions))]
                elif args.dataset == 'locomo':
                    # LoCoMo has different session structure
                    haystack_sessions = haystack.get('haystack_sessions', [])
                    haystack_dates = haystack.get('haystack_dates', [f"session_{i}" for i in range(len(haystack_sessions))])
                else:
                    # Fallback
                    haystack_sessions = haystack.get('haystack_sessions', [])
                    haystack_dates = haystack.get('haystack_dates', [f"session_{i}" for i in range(len(haystack_sessions))])

                logger.debug(f"Processing sample {i+1}/{len(haystacks)}: {question_id}")

                # Time haystack processing
                stopwatch.start()
                memstruct = process_haystack(haystack_sessions, haystack_dates, logger)
                haystack_time.append(stopwatch.stop())

                # Time question processing
                stopwatch.start()
                predicted_idx, k_used, retrieval_stats = process_question_mc(memstruct, question, choices, question_date, top_k, model_name, args.adaptive_k, logger)
                question_time.append(stopwatch.stop())
                k_values.append(k_used)
                
                # Add question-specific information to retrieval stats
                retrieval_stats['question_id'] = question_id
                retrieval_stats['question_type'] = question_type
                all_retrieval_stats.append(retrieval_stats)

                # Evaluate result
                is_correct = evaluate_mc_answer(predicted_idx, correct_idx)
                
                result_entry = {
                    'question_id': question_id,
                    'question_type': question_type,
                    'predicted_choice': predicted_idx,
                    'correct_choice': correct_idx,
                    'predicted_answer': choices[predicted_idx] if 0 <= predicted_idx < len(choices) else "INVALID",
                    'correct_answer': choices[correct_idx],
                    'correct': is_correct
                }
                results.append(result_entry)

                num_success += is_correct
                nobs += 1

                # Log progress
                current_accuracy = num_success / nobs if nobs > 0 else 0
                logger.info(f"Sample {i+1}: {'✓' if is_correct else '✗'} (Accuracy: {current_accuracy:.3f})")

                # Save intermediate results
                if nobs % args.save_interval == 0:
                    logger.info(f"Saving intermediate results after {nobs} samples")
                    save_intermediate_results(results, haystack_time, question_time, args.output_dir, args.dataset)

            except Exception as e:
                logger.error(f"Error processing sample {question_id}: {str(e)}")
                continue

        # Final results
        logger.info("Processing completed")
        logger.info(f'Evaluated {nobs} samples with {num_success} successes')

        if nobs > 0:
            final_accuracy = num_success / nobs
            logger.info(f'Final Accuracy: {final_accuracy:.4f}')

            # Detailed evaluation
            evaluate_qa_mc(results, logger)

            # Timing statistics
            if haystack_time:
                logger.info(f"Haystack time median: {np.median(haystack_time):.4f} seconds")
            if question_time:
                logger.info(f"Question time median: {np.median(question_time):.4f} seconds")

            # K-distribution statistics
            if k_values:
                k_array = np.array(k_values)
                logger.info("K-Distribution Statistics:")
                logger.info(f"  Mean K: {k_array.mean():.2f}")
                logger.info(f"  Median K: {np.median(k_array):.1f}")
                logger.info(f"  Std K: {k_array.std():.2f}")
                logger.info(f"  Min K: {k_array.min()}")
                logger.info(f"  Max K: {k_array.max()}")
                
                # K value frequency distribution
                unique_k, counts = np.unique(k_array, return_counts=True)
                logger.info("K-Value Distribution:")
                for k, count in zip(unique_k, counts):
                    percentage = (count / len(k_values)) * 100
                    logger.info(f"  K={k}: {count} times ({percentage:.1f}%)")
                
                # Create comprehensive visualizations using seaborn
                method_name = "Top-P + MMR" if args.adaptive_k else f"Top-K (k={top_k})"
                current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_file = create_comprehensive_visualizations(
                    all_retrieval_stats, results, k_values, method_name, 
                    args.output_dir, args.dataset, current_timestamp, logger
                )

            # Save final results with k-values
            final_files = save_intermediate_results(results, haystack_time, question_time, args.output_dir, args.dataset)
            
            # Save k-distribution data separately
            if k_values:
                current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                k_dist_file = Path(args.output_dir) / f"k_distribution_{args.dataset}_{current_timestamp}.json"
                k_dist_data = {
                    "k_values": k_values,
                    "statistics": {
                        "mean": float(k_array.mean()),
                        "median": float(np.median(k_array)),
                        "std": float(k_array.std()),
                        "min": int(k_array.min()),
                        "max": int(k_array.max())
                    },
                    "distribution": {int(k): int(count) for k, count in zip(unique_k, counts)},
                    "method": "top_p_mmr" if args.adaptive_k else "top_k",
                    "top_k_setting": top_k
                }
                with open(k_dist_file, 'w') as f:
                    json.dump(k_dist_data, f, indent=2)
                logger.info(f"K-distribution data saved to {k_dist_file}")
            
            logger.info(f"Final results saved to {final_files[0]}")
        else:
            logger.warning("No samples were processed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == '__main__':
    main()