import re
import abc
import json
import torch
import random
import warnings
import numpy as np
from datasets import Dataset, logging as dataset_logging
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging

# avoid boring logging
dataset_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
SEED = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(gpu: bool = True) -> torch.device:
    """
    Determines the device (CPU or GPU) to run the model on.

    Returns:
        torch.device.
    """
    if gpu and torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    if gpu and not torch.cuda.is_available():
        warnings.warn("No GPU found. CPU will be used", category=UserWarning)

    return torch.device(device_name)


class HugginfaceHelper(abc.ABC):
    """Abstract class that facilitates the usage of BERT-like models"""

    def __init__(self,
                 pretrained: str = 'bert-base-uncased',  # bert for English
                 output_hidden_states: bool = True,
                 output_attentions: bool = False,
                 return_special_tokens_mask=False,
                 subword_prefix: str = '##',
                 use_gpu: bool = True):
        """
        Load a model and its corresponding tokenizer.

        Args:
            pretrained (str, default='bert-base-uncased'): Model name or path to load.
            output_hidden_states (bool, default=False): Whether to output all hidden-states of the model. False by default.
            output_attentions (bool, default=False): Whether to output attentions weights of the model. False by default.
            return_special_tokens_mask (bool, default=False): Whether to output special tokens masks. False by default.
            subword_prefix (str, default='##'): The subword prefix used to split word in subwords. Default is '##' for BERT.
            use_gpu (bool, default=True): use gpu if available. True by default.
        """

        # set seed and device
        set_seed(SEED)
        self._device = set_device(use_gpu)

        # load hugginface tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained,
                                               output_hidden_states=output_hidden_states,
                                               output_attentions=output_attentions)

        # some attributes
        self.pretrained = pretrained  # model name
        self.subword_prefix = subword_prefix  # subword prefix
        self.num_layers = self.model.config.num_hidden_layers  # number of model's layers
        self.num_heads = self.model.config.num_attention_heads  # number of model's heads
        self._return_special_tokens_mask = return_special_tokens_mask  # flag

        # enable parallelization: [TODO] is this correct/useful?
        if torch.cuda.device_count() > 1 and use_gpu:
            self.model = torch.nn.DataParallel(self.model)

        # load model on gpu/cpu memory
        _ = self.model.to(self._device)

        # evaluation mode
        _ = self.model.eval()

    def _tokenize_factory(self, max_length: int = None) -> callable:
        """
        Factory method: create the tokenization function to tokenize text.

        Args:
            max_length(int, optional): The maximum sequence length to use. If None, the maximum length is used.

        Returns:
            function
        """
        if max_length is not None:
            max_length = max_length
        else:
            max_length = self.tokenizer.model_max_length

        def tokenize(examples) -> dict:
            """Tokenization function"""
            return self.tokenizer(examples["sentence"],
                                  return_tensors='pt',
                                  padding="max_length",
                                  max_length=max_length,
                                  truncation=True,
                                  return_special_tokens_mask=self._return_special_tokens_mask).to(self._device)

        return tokenize

    def _load_dataset(self, dataset: str) -> Dataset:
        """
        Load a dataset from file.

        Args:
            dataset (str): Path of the jsonl dataset.

        Returns:
            Dataset
        """

        records = list()
        with open(dataset, mode='r', encoding='utf-8') as f:
            for json_line in f:
                if json_line.strip() == '': continue  # avoid blank lines
                record = json.loads(json_line)
                if record is None: continue  # avoid empty record
                records.append(record)

        return Dataset.from_list(records)

    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        """
        Tokenize the dataset.

        Args:
            dataset (Dataset): Dataset to tokenize.
            max_length (int): Maximum number of tokens.

        Returns:
            Dataset
        """
        tokenize_func = self._tokenize_factory(max_length)
        dataset = dataset.map(tokenize_func, batched=True)
        dataset.set_format('torch')
        return dataset

class EmbeddingsExtraction(HugginfaceHelper):
    def extract_embeddings(self,
                           dataset:str,
                           batch_size:int=8,
                           max_length:int=None,
                           **kwargs) -> object:
        """
        Extracts word embeddings for a given dataset.

        Args:
            dataset (str): Path of the jsonl dataset.
            batch_size (int, default=8): Batch size used for extracting embeddings.
            max_length (int, default=None): Maximum sequence length used during the tokenization.

        Returns:
            A dict or tensor representing the extracted embeddings.
        """

class TargetEmbeddingsExtraction(EmbeddingsExtraction):
    def __init__(self,
                 pretrained: str ='bert-base-uncased',  # bert for English
                 subword_prefix: str = '##',
                 use_gpu: bool = True):
        super().__init__(pretrained,
                         output_hidden_states=True,
                         output_attentions=False,
                         return_special_tokens_mask=False,
                         subword_prefix=subword_prefix,
                         use_gpu=use_gpu)

    def add_token_to_vocab(self):
        self.tokenizer.add_tokens(["unkrand"])
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))
    
    def extract_embeddings(self,
                           dataset: str,
                           batch_size: int = 8,
                           max_length: int = None) -> dict:
        """
        Extracts embeddings of a target word in a given dataset.

        Args:
            dataset (str): Path of the jsonl dataset.
            batch_size (int, default=8): Batch size used for extracting embeddings.
            max_length (int, default=None): Maximum sequence length used during the tokenization.

        Returns:
            A dict or tensor representing the extracted embeddings.
        """

        # load dataset
        dataset = self._load_dataset(dataset)

        # split text from other data
        text = dataset.select_columns('sentence')
        offset = dataset.remove_columns('sentence')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text, max_length)

        # collect embedding to store on disk
        embeddings = dict()
        for i in range(0, tokenized_text.shape[0], batch_size):
            start, end = i, min(i + batch_size, text.num_rows)
            batch_offset = offset.select(range(start, end))
            batch_text = text.select(range(start, end))
            batch_tokenized_text = tokenized_text.select(range(start, end))

            model_input = dict()

            # to device
            model_input['input_ids'] = batch_tokenized_text['input_ids'].to(self._device)

            # XLM-R doesn't use 'token_type_ids'
            if 'token_type_ids' in batch_tokenized_text:
                model_input['token_type_ids'] = batch_tokenized_text['token_type_ids'].to(self._device)

            model_input['attention_mask'] = batch_tokenized_text['attention_mask'].to(self._device)

            # model prediction
            with torch.no_grad():
                model_output = self.model(**model_input)

            # hidden states
            hidden_states = torch.stack(model_output['hidden_states'])

            # select the embeddings of a specific target word
            for j, row in enumerate(batch_tokenized_text):
                # string containing tokens of the j-th sequence
                input_tokens = row['input_ids'].tolist()
                input_tokens_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_tokens))

                # string containing tokens of the target word occurrence
                word_tokens = batch_text[j]['sentence'][batch_offset[j]['start']:batch_offset[j]['end']]
                word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

                 # search the occurrence of 'word_tokens_str' in 'input_tokens_str' to get the corresponding position
                try:
                    # First occurrence of a Word
                    #match = re.search(f"( +|^){word_tokens_str}(?!\w+| {self.subword_prefix})", input_tokens_str, re.DOTALL)

                    matches, pos_offset, pos_error, pos = list(), 0, None, None
                    while True:
                        tmp = input_tokens_str[pos_offset:]
                        match = re.search(f"( +|^){word_tokens_str}(?!\w+| ##)", tmp, re.DOTALL)

                        if match is None:
                            break

                        current_pos = pos_offset + match.start()
                        current_error = abs(current_pos - batch_offset[j]['start'])

                        if pos is None or current_error < pos_error:
                            pos = current_pos
                            pos_error = current_error
                        else:
                            break

                        pos_offset += match.end()
                        matches.append(match)
                except:
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    warnings.warn(
                        f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored",
                        category=UserWarning)
                    continue

                # Truncation side effect: the target word is over the maximum input length
                #if match is None: #First occurrence
                if len(matches) == 0:
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    warnings.warn(f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored",category=UserWarning)
                    print(f"Model: {self.pretrained},\nToken: {word_tokens},\nTokenized text: {batch_text[j]},\nIndexes: {batch_offset[j]['start']}:{batch_offset[j]['end']},\nWordTokens: {word_tokens_str},\nSequenceTokens: {input_tokens_str}")
                    continue

                #pos = match.start()  # index first sub-word - First Occurrence
                n_previous_tokens = len(input_tokens_str[:pos].split())  # number of tokens before that sub-word
                n_word_token = len(word_tokens_str.split())  # number of tokens of the target word

                # Store the embeddings from each layer
                for layer in range(1, self.num_layers + 1):
                    # embeddings of each sub-words
                    sub_word_state = hidden_states[layer, j][n_previous_tokens: n_previous_tokens + n_word_token]

                    # mean of sub-words embeddings
                    word_state = torch.mean(sub_word_state, dim=0).unsqueeze(0)

                    if layer in embeddings:
                        embeddings[layer] = torch.vstack([embeddings[layer], word_state])
                    else:
                        embeddings[layer] = word_state

            # empty cache
            torch.cuda.empty_cache()

        return embeddings

class EmbeddingsExtractionTargetLayer(EmbeddingsExtraction):
    def __init__(self,
                 pretrained: str ='bert-base-uncased',  # bert for English
                 subword_prefix: str = '##',
                 use_gpu: bool = True):
        super().__init__(pretrained,
                         output_hidden_states=True,
                         output_attentions=False,
                         return_special_tokens_mask=True,
                         subword_prefix=subword_prefix,
                         use_gpu=use_gpu)

    def add_token_to_vocab(self):
        self.tokenizer.add_tokens(["unkrand"], special_tokens=False)
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))

    def extract_embeddings(self,
                           dataset:str,
                           batch_size:int=8,
                           max_length:int=None,
                           layer:int=12) -> tuple:
        """
        Extracts all word embeddings of the dataset sentences from a target layer.

        Args:
            dataset (str): Path of the jsonl dataset.
            batch_size (int, default=8): Batch size used for extracting embeddings.
            max_length (int, default=None): Maximum sequence length used during the tokenization.

        Returns:
            Tuple
        """

        # Wrapper
        target_indexes = list() # indexes of a target word
        embeddings = None # contextualized word embeddings
        special_tokens_mask = None # mask of special tokens

        # load dataset
        dataset = self._load_dataset(dataset)

        # split text from other data
        text = dataset.select_columns('sentence')
        offset = dataset.remove_columns('sentence')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text, max_length)

        for i in range(0, tokenized_text.shape[0], batch_size):
            start_batch, end_batch = i, min(i + batch_size, text.num_rows)
            batch_offset = offset.select(range(start_batch, end_batch))
            batch_text = text.select(range(start_batch, end_batch))
            batch_tokenized_text = tokenized_text.select(range(start_batch, end_batch))

            model_input = dict()

            # to device
            model_input['input_ids'] = batch_tokenized_text['input_ids'].to(self._device)

            # XLM-R doesn't use 'token_type_ids'
            if 'token_type_ids' in batch_tokenized_text:
                model_input['token_type_ids'] = batch_tokenized_text['token_type_ids'].to(self._device)

            model_input['attention_mask'] = batch_tokenized_text['attention_mask'].to(self._device)

            # model prediction
            with torch.no_grad():
                model_output = self.model(**model_input)

            # hidden states
            hidden_states = torch.stack(model_output['hidden_states']).detach().cpu()[layer]

            # store data
            if i == 0:
                special_tokens_mask = batch_tokenized_text['special_tokens_mask']
                embeddings = hidden_states
            else:
                special_tokens_mask = torch.vstack([special_tokens_mask,
                                                    batch_tokenized_text['special_tokens_mask']])
                embeddings = torch.vstack([embeddings, hidden_states])

            # Target word retrieval
            for j, row in enumerate(batch_tokenized_text):
                start, end = batch_offset[j]['start'], batch_offset[j]['end']
                sentence = batch_text[j]['sentence']

                # string containing tokens of the target word occurrence
                word_tokens = sentence[start:end]
                word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

                # string containing tokens of the j-th sequence
                input_tokens = row['input_ids'].tolist()
                input_tokens_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_tokens))

                # search the occurrence of 'word_tokens_str' in 'input_tokens_str' to get the corresponding position
                try:
                    match = re.search(f"( +|^){word_tokens_str}(?!\w+| {self.subword_prefix})", input_tokens_str,
                                      re.DOTALL)
                except:
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    warnings.warn(
                        f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored",
                        category=UserWarning)
                    continue

                # Truncation side effect: the target word is over the maximum input length
                if match is None:
                    idx_original_sent = batch_tokenized_text.num_rows * i + j
                    warnings.warn(f"An error occurred with the {idx_original_sent}-th sentence: {batch_text[j]}. It will be ignored", category=UserWarning)
                    print(f"Model: {self.pretrained},\nToken: {word_tokens},\nTokenized text: {batch_text[j]},\nIndexes: {batch_offset[j]['start']}:{batch_offset[j]['end']},\nWordTokens: {word_tokens_str},\nSequenceTokens: {input_tokens_str}")
                    continue

                pos = match.start()  # index first sub-word
                n_previous_tokens = len(input_tokens_str[:pos].split())  # number of tokens before that sub-word
                n_word_token = len(word_tokens_str.split())  # number of tokens of the target word

                # token indexes
                start = n_previous_tokens # ind sub-word position
                end = n_previous_tokens + n_word_token

                # store indexes as a string
                idx = " ".join([str(idx) for idx in range(start, end)])
                target_indexes.append(idx)

            # empty cache
            torch.cuda.empty_cache()

        return embeddings, np.array(target_indexes), special_tokens_mask
