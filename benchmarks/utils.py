from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import json
from pathlib import Path
import random
from datasets import load_dataset as hf_load_dataset


@dataclass
class DataLoadingConfig:
    """Configuration for dataset loading operations."""
    dataset_key: str
    num_samples: int = 0
    random_sampling: bool = False
    start_index: int = 0
    load_all: bool = False
    question_types: Optional[List[str]] = None
    
    @property
    def effective_num_samples(self) -> int:
        """Get the effective number of samples (0 if load_all is True)."""
        return 0 if self.load_all else self.num_samples


def apply_question_type_filter(
        dataset: List[Dict[str, Any]], 
        question_types: List[str],
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Filter dataset by question types."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Filtering questions by types: {', '.join(question_types)}")
    
    filtered_dataset = [
        item for item in dataset 
        if item.get('question_type') in question_types
    ]
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} questions")
    
    # Log counts by question type
    type_counts = {}
    for item in filtered_dataset:
        qtype = item.get('question_type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    for qtype, count in sorted(type_counts.items()):
        logger.info(f"  {qtype}: {count} questions")
    
    return filtered_dataset


def load_benchmark_dataset(
        config: DataLoadingConfig,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load benchmark dataset with unified interface.
    
    This function handles all dataset loading scenarios:
    - HuggingFace datasets (msc, lme)
    - Local file datasets (locomo)
    - Question type filtering (lme)
    - All sampling options
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    try:
        from .config import DATASET_CONFIGS
    except ImportError:
        logger.error("Cannot import DATASET_CONFIGS. Ensure benchmarks.config is available.")
        return []
    
    if config.dataset_key not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {config.dataset_key}")
        return []
    
    dataset_config = DATASET_CONFIGS[config.dataset_key]
    
    # Log loading info
    if config.load_all:
        logger.info(f"Loading {config.dataset_key.upper()} dataset (ALL samples)...")
    else:
        logger.info(f"Loading {config.dataset_key.upper()} dataset ({config.num_samples} samples)...")
    
    if config.random_sampling:
        logger.info("Using random sampling from dataset.")
    else:
        logger.info(f"Using deterministic sampling starting from index {config.start_index}.")
    
    # Load dataset based on type
    if dataset_config["dataset_id"] == "local":
        # Local file loading (LoCoMo)
        dataset = load_dataset_from_local_file(
            dataset_config["data_file"],
            config.effective_num_samples,
            config.random_sampling,
            config.start_index,
            logger
        )
    else:
        # HuggingFace loading (MSC, LME)
        if config.question_types:
            # Load full dataset first for filtering
            full_dataset = load_dataset_from_huggingface(
                dataset_config,
                num_samples=0,  # Load all for filtering
                random_sampling=False,
                start_index=0,
                logger=logger
            )
            
            if not full_dataset:
                return []
            
            # Apply question type filter
            filtered_dataset = apply_question_type_filter(full_dataset, config.question_types, logger)
            
            if not filtered_dataset:
                logger.error("No questions match the specified question types.")
                return []
            
            # Apply sampling to filtered dataset
            if config.load_all:
                dataset = filtered_dataset
            else:
                if config.random_sampling:
                    if len(filtered_dataset) <= config.num_samples:
                        dataset = filtered_dataset
                    else:
                        dataset = random.sample(filtered_dataset, config.num_samples)
                else:
                    start_idx = config.start_index
                    end_idx = start_idx + config.num_samples
                    dataset = filtered_dataset[start_idx:end_idx]
        else:
            # Direct loading without filtering
            dataset = load_dataset_from_huggingface(
                dataset_config,
                config.effective_num_samples,
                config.random_sampling,
                config.start_index,
                logger
            )
    
    if not dataset:
        logger.error("Failed to load dataset.")
        return []
    
    actual_count = len(dataset)
    requested_count = config.effective_num_samples
    
    if requested_count > 0 and actual_count < requested_count:
        logger.warning(f"Only {actual_count} questions available, less than requested {requested_count}.")
    
    logger.info(f"Successfully loaded {actual_count} questions")
    return dataset


def load_dataset_from_local_file(
        file_path: str,
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load dataset from local JSON/JSONL file."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Loading dataset from local file: {file_path}")
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return []
            
        with open(path, 'r', encoding='utf-8') as f:
            # Try JSON array format first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error("Expected dataset to be a list of questions.")
                    return []
                logger.info(f"Successfully loaded {len(data)} questions from JSON array format")
            except json.JSONDecodeError:
                # Try JSONL format
                logger.info("JSON array format failed, trying JSONL format...")
                f.seek(0)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue
                
                if not data:
                    logger.error("No valid JSON objects found in file")
                    return []
                
                logger.info(f"Successfully loaded {len(data)} questions from JSONL format")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        return []
        

def _apply_sampling(
        data: List[Dict[str, Any]], 
        num_samples: int, 
        random_sampling: bool, 
        start_index: int,
        logger: logging.Logger
    ) -> List[Dict[str, Any]]:
    """Apply sampling logic to dataset."""
    if num_samples > 0 and num_samples < len(data):
        if random_sampling:
            logger.info(f"Using random sampling: {num_samples} from {len(data)} total")
            data = random.sample(data, num_samples)
        else:
            logger.info(f"Using deterministic sampling (from index {start_index})")
            # Ensure start_index is within bounds
            if start_index >= len(data):
                logger.warning(f"Start index {start_index} exceeds dataset size {len(data)}. Using index 0.")
                start_index = 0
            # Handle wrapping around if needed
            if start_index + num_samples > len(data):
                logger.info(f"Requested {num_samples} samples from index {start_index}, but only {len(data) - start_index} samples are available. Wrapping around to the beginning.")
                end_samples = data[start_index:]
                remaining_samples = data[:num_samples - len(end_samples)]
                data = end_samples + remaining_samples
            else:
                data = data[start_index:start_index + num_samples]
    elif start_index > 0:
        logger.info(f"Starting from index {start_index}")
        data = data[start_index:]
    
    logger.info(f"Final dataset size: {len(data)} samples")
    return data


def load_dataset_from_huggingface(
        config: Dict[str, Any], 
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dataset_id = config["dataset_id"]
    data_file = config.get("data_file")
    logger.info(f"Loading dataset from HuggingFace ({dataset_id}, {data_file})...")
    try:
        if data_file:
            ds = hf_load_dataset(dataset_id, data_files=data_file, split="train")
        else:
            ds = hf_load_dataset(dataset_id, split="train")
        data = list(ds)
        logger.info(f"Successfully loaded {len(data)} samples.")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as ex:
        logger.error(f"Error loading dataset from HuggingFace: {ex}")
        return []