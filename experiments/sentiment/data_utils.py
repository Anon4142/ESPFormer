from datasets import load_dataset  # tweet_eval
from typing import Iterable, Union, List
import torch
from torch.utils.data import TensorDataset

from prenlp.data import IMDB
DATASETS_CLASSES = {'imdb': IMDB}
# Keep your InputExample and InputFeatures classes unchanged
class InputExample:
    """A single training/test example for text classification."""
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label

class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids: List[int], label_id: int):
        self.input_ids = input_ids
        self.label_id = label_id

def convert_examples_to_features(examples: List[InputExample],
                                 label_dict: dict,
                                 tokenizer,
                                 max_seq_len: int) -> List[InputFeatures]:
    pad_token_id = tokenizer.pad_token_id
    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[:max_seq_len]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        label_id = label_dict.get(example.label)
        features.append(InputFeatures(input_ids, label_id))
    return features

def create_examples(args, tokenizer, mode: str = 'train') -> TensorDataset:
    examples = []
    
    if args.dataset == "tweet_eval":
        # Load tweet_eval sentiment analysis dataset from Hugging Face
        dataset = load_dataset("tweet_eval", "sentiment")
        # Choose the appropriate split: "train", "test", or "validation"
        if mode == "train":
            data_split = dataset["train"]
        elif mode == "test":
            data_split = dataset["test"]
        else:
            data_split = dataset["validation"]
        # Each example in tweet_eval is a dict with keys "text" and "label".
        # Convert label to string if you want to keep the same structure as before.
        for item in data_split:
            examples.append(InputExample(text=item["text"], label=str(item["label"])))
    else:
        # Use the original IMDB dataset loader.
        # Here, DATASETS_CLASSES[args.dataset]() returns a tuple: (train, test)
        imdb_data = DATASETS_CLASSES[args.dataset]()
        dataset_split = imdb_data[0] if mode == "train" else imdb_data[1]
        # Assume each example in IMDB is a tuple (text, label)
        for text, label in dataset_split:
            examples.append(InputExample(text, label))

    # Create a label dictionary based on the unique labels
    labels = sorted(list(set(example.label for example in examples)))
    label_dict = {label: i for i, label in enumerate(labels)} #[0,1,2]

    
    features = convert_examples_to_features(examples, label_dict, tokenizer, args.max_seq_len)
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)
    
    return TensorDataset(all_input_ids, all_label_ids)
