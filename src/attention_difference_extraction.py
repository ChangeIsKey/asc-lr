import re
import math # you need this for math.nan
import torch
import warnings
import numpy as np
from src.embeddings_extraction import HugginfaceHelper


class AttentionAnalysis(HugginfaceHelper):
    """This class is used to extract attention and compute attention differences"""
    
    def __init__(self,
                 pretrained: str = 'bert-base-uncased',  # bert for English
                 subword_prefix: str = '##',
                 use_gpu: bool = True):
        
        super().__init__(pretrained=pretrained,
                         output_hidden_states=False,
                         output_attentions=True,
                         return_special_tokens_mask=True,
                         subword_prefix=subword_prefix,
                         use_gpu=use_gpu)

    def add_token_to_vocab(self):
        self.tokenizer.add_tokens(["unkrand"], special_tokens=False)
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))

    def extract_attn_distances(self,
                               dataset: str,
                               batch_size: int = 8,
                               max_length: int = None,
                               layer: int = 12) -> dict:
        """
        Extracts attention distances from a dataset.

        Args:
            dataset (str): Path of the jsonl dataset.
            batch_size (int, default=8): Batch size used for extracting attentions.
            max_length (int, default=None): Maximum sequence length used during the tokenization.
            layer (int, default=12):
        Returns:
            dict
        """
        
        layer = layer - 1 # indexes are from 0 to 11

        # load dataset
        dataset = self._load_dataset(dataset)

        # split text from other data
        text = dataset.select_columns('sentence')
        offset = dataset.remove_columns('sentence')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text, max_length)

        # distances wrapper
        target_distances_full = list()
        target_distances_from, target_distances_to = list(), list()
        left_10_distances_from, left_10_distances_to = list(), list()
        left_9_distances_from, left_9_distances_to = list(), list()
        left_8_distances_from, left_8_distances_to = list(), list()
        left_7_distances_from, left_7_distances_to = list(), list()
        left_6_distances_from, left_6_distances_to = list(), list()
        left_5_distances_from, left_5_distances_to = list(), list()
        left_4_distances_from, left_4_distances_to = list(), list()
        left_3_distances_from, left_3_distances_to = list(), list()
        left_2_distances_from, left_2_distances_to = list(), list()
        left_1_distances_from, left_1_distances_to = list(), list()
        right_10_distances_from, right_10_distances_to = list(), list()
        right_9_distances_from, right_9_distances_to = list(), list()
        right_8_distances_from, right_8_distances_to = list(), list()
        right_7_distances_from, right_7_distances_to = list(), list()
        right_6_distances_from, right_6_distances_to = list(), list()
        right_5_distances_from, right_5_distances_to = list(), list()
        right_4_distances_from, right_4_distances_to = list(), list()
        right_3_distances_from, right_3_distances_to = list(), list()
        right_2_distances_from, right_2_distances_to = list(), list()
        right_1_distances_from, right_1_distances_to = list(), list()

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

            # permute: sentence, layer, head, attn_token_out, attn_token_in
            attns = torch.stack(model_output['attentions']).permute(1, 0, 2, 3, 4)

            # average attentions head
            attns = attns.mean(axis=2)  # now: sentence, layer, attn_token_out, attn_token_in

            # special token mask
            special_tokens_masks = batch_tokenized_text['special_tokens_mask']

            for j in range(0, attns.shape[0], 2):

                # attentions
                attn_replacement = attns[j]
                attn_original = attns[j + 1]

                # special token masks
                special_tokens_mask_replacement = special_tokens_masks[j]
                special_tokens_mask_original = special_tokens_masks[j + 1]

                # idx target replacement and original
                idx_target_replacement, idx_target_original = self._get_target_index(batch_tokenized_text,
                                                                                     batch_text,
                                                                                     batch_offset,
                                                                                     i, j)

                # idx other than target (not special)
                idx_other_replacement = [k for k in range(0, attn_replacement.shape[0])
                                         if k not in idx_target_replacement and
                                         not special_tokens_mask_replacement[k]]
                idx_other_original = [k for k in range(0, attn_original.shape[0])
                                      if k not in idx_target_original and
                                      not special_tokens_mask_original[k]]

                # attention from target token to others
                attn_from_replacement = attn_replacement[:, idx_target_replacement]
                attn_from_replacement = attn_from_replacement[:, :, idx_other_replacement]
                attn_from_replacement = attn_from_replacement.mean(axis=1).detach().cpu()
                attn_from_original = attn_original[:, idx_target_original]
                attn_from_original = attn_from_original[:, :, idx_other_original]
                attn_from_original = attn_from_original.mean(axis=1).detach().cpu()
                target_distances_from.append(
                    abs(attn_from_replacement[layer].mean() - attn_from_original[layer].mean()))

                # attention from others to target token
                attn_to_replacement = attn_replacement[:, idx_other_replacement]
                attn_to_replacement = attn_to_replacement[:, :, idx_target_replacement]
                attn_to_replacement = attn_to_replacement.mean(axis=2).detach().cpu()
                attn_to_original = attn_original[:, idx_other_original]
                attn_to_original = attn_to_original[:, :, idx_target_original]
                attn_to_original = attn_to_original.mean(axis=2).detach().cpu()
                target_distances_to.append(
                    abs(attn_to_replacement[layer].mean() - attn_to_original[layer].mean()))

                # full attention
                attn_full_replacement = attn_replacement[:, idx_other_replacement, :]
                attn_full_replacement = attn_full_replacement[:, :, idx_other_replacement].detach().cpu()
                attn_full_original = attn_original[:, idx_other_original, :]
                attn_full_original = attn_full_original[:, :, idx_other_original].detach().cpu()
                target_distances_full.append(
                    abs(attn_full_replacement[layer].mean() - attn_full_original[layer].mean()))

                idx_special_original = special_tokens_mask_original.nonzero().squeeze()
                idx_special_replacement = special_tokens_mask_replacement.nonzero().squeeze()
                for k in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
                    left_id = idx_target_replacement[0] - k  # is the same for original and replacement
                    right_id_replacement = idx_target_replacement[-1] + k
                    right_id_original = idx_target_original[-1] + k

                    if left_id < 1 or left_id in idx_special_replacement:
                        exec(f'left_{k}_distances_from.append(math.nan)')
                        exec(f'left_{k}_distances_to.append(math.nan)')
                    else:
                        idx_other_replacement = [tmp for tmp in range(0, attn_replacement.shape[0])
                                                 if not tmp in idx_special_replacement and
                                                 tmp != left_id]
                        idx_other_original = [tmp for tmp in range(0, attn_original.shape[0])
                                              if not tmp in idx_special_original and
                                              tmp != left_id]

                        # attention from target token to others
                        attn_from_replacement = attn_replacement[:, [left_id]]
                        attn_from_replacement = attn_from_replacement[:, :, idx_other_replacement].detach().cpu()
                        attn_from_original = attn_original[:, [left_id]]
                        attn_from_original = attn_from_original[:, :, idx_other_original].detach().cpu()
                        exec(
                            f'left_{k}_distances_from.append(abs(attn_from_replacement[layer].mean() - attn_from_original[layer].mean()))')

                        # attention from others to target token
                        attn_to_replacement = attn_replacement[:, idx_other_replacement]
                        attn_to_replacement = attn_to_replacement[:, :, [left_id]].detach().cpu()
                        attn_to_original = attn_original[:, idx_other_original]
                        attn_to_original = attn_to_original[:, :, [left_id]].detach().cpu()
                        exec(
                            f'left_{k}_distances_to.append(abs(attn_to_replacement[layer].mean() - attn_to_original[layer].mean()))')

                    if right_id_replacement in idx_special_replacement or \
                            right_id_replacement > max_length:
                        exec(f'right_{k}_distances_from.append(math.nan)')
                        exec(f'right_{k}_distances_to.append(math.nan)')
                    else:
                        idx_other_replacement = [tmp for tmp in range(0, attn_replacement.shape[0])
                                                 if not tmp in idx_special_replacement and
                                                 tmp != right_id_replacement]
                        idx_other_original = [tmp for tmp in range(0, attn_original.shape[0])
                                              if not tmp in idx_special_original
                                              and tmp != right_id_original]

                        # attention from target token to others
                        attn_from_replacement = attn_replacement[:, [right_id_replacement]]
                        attn_from_replacement = attn_from_replacement[:, :, idx_other_replacement].detach().cpu()
                        attn_from_original = attn_original[:, [right_id_original]]
                        attn_from_original = attn_from_original[:, :, idx_other_original].detach().cpu()
                        exec(
                            f'right_{k}_distances_from.append(abs(attn_from_replacement[layer].mean() - attn_from_original[layer].mean()))')

                        # attention from others to target token
                        attn_to_replacement = attn_replacement[:, idx_other_replacement]
                        attn_to_replacement = attn_to_replacement[:, :, [right_id_replacement]].detach().cpu()
                        attn_to_original = attn_original[:, idx_other_original]
                        attn_to_original = attn_to_original[:, :, [right_id_original]].detach().cpu()
                        exec(
                            f'right_{k}_distances_to.append(abs(attn_to_replacement[layer].mean() - attn_to_original[layer].mean()))')

        return dict(attn_distance_target_from=np.array(target_distances_from),
                    attn_distance_target_to=np.array(target_distances_to),
                    attn_distance_full=np.array(target_distances_full),
                    attn_distance_10left_from=np.array(left_10_distances_from),
                    attn_distance_9left_from=np.array(left_9_distances_from),
                    attn_distance_8left_from=np.array(left_8_distances_from),
                    attn_distance_7left_from=np.array(left_7_distances_from),
                    attn_distance_6left_from=np.array(left_6_distances_from),
                    attn_distance_5left_from=np.array(left_5_distances_from),
                    attn_distance_4left_from=np.array(left_4_distances_from),
                    attn_distance_3left_from=np.array(left_3_distances_from),
                    attn_distance_2left_from=np.array(left_2_distances_from),
                    attn_distance_1left_from=np.array(left_1_distances_from),
                    attn_distance_10left_to=np.array(left_10_distances_to),
                    attn_distance_9left_to=np.array(left_9_distances_to),
                    attn_distance_8left_to=np.array(left_8_distances_to),
                    attn_distance_7left_to=np.array(left_7_distances_to),
                    attn_distance_6left_to=np.array(left_6_distances_to),
                    attn_distance_5left_to=np.array(left_5_distances_to),
                    attn_distance_4left_to=np.array(left_4_distances_to),
                    attn_distance_3left_to=np.array(left_3_distances_to),
                    attn_distance_2left_to=np.array(left_2_distances_to),
                    attn_distance_1left_to=np.array(left_1_distances_to),
                    attn_distance_10right_from=np.array(right_10_distances_from),
                    attn_distance_9right_from=np.array(right_9_distances_from),
                    attn_distance_8right_from=np.array(right_8_distances_from),
                    attn_distance_7right_from=np.array(right_7_distances_from),
                    attn_distance_6right_from=np.array(right_6_distances_from),
                    attn_distance_5right_from=np.array(right_5_distances_from),
                    attn_distance_4right_from=np.array(right_4_distances_from),
                    attn_distance_3right_from=np.array(right_3_distances_from),
                    attn_distance_2right_from=np.array(right_2_distances_from),
                    attn_distance_1right_from=np.array(right_1_distances_from),
                    attn_distance_10right_to=np.array(right_10_distances_to),
                    attn_distance_9right_to=np.array(right_9_distances_to),
                    attn_distance_8right_to=np.array(right_8_distances_to),
                    attn_distance_7right_to=np.array(right_7_distances_to),
                    attn_distance_6right_to=np.array(right_6_distances_to),
                    attn_distance_5right_to=np.array(right_5_distances_to),
                    attn_distance_4right_to=np.array(right_4_distances_to),
                    attn_distance_3right_to=np.array(right_3_distances_to),
                    attn_distance_2right_to=np.array(right_2_distances_to),
                    attn_distance_1right_to=np.array(right_1_distances_to))

    def _get_token_strings(self, sentence, start, end, input_ids):
        # string containing tokens of the target word occurrence
        word_tokens = sentence[start:end]
        word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

        # string containing tokens of the j-th sequence
        input_tokens = input_ids.tolist()
        input_tokens_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_tokens))
        return word_tokens_str, input_tokens_str

    def _get_target_index(self, batch_tokenized_text, batch_text, batch_offset, i, j):
        # sentence
        sentence_replacement = batch_text[j]['sentence']
        sentence_original = batch_text[j + 1]['sentence']

        # start/end
        start_replacement, end_replacement = batch_offset[j]['start'], batch_offset[j]['end']
        start_original, end_original = batch_offset[j + 1]['start'], batch_offset[j + 1]['end']

        # input_ids
        input_ids_replacement = batch_tokenized_text[j]['input_ids']
        input_ids_original = batch_tokenized_text[j + 1]['input_ids']

        # token strings
        word_tokens_str_replacement, input_tokens_str_replacement = self._get_token_strings(sentence_replacement,
                                                                                            start_replacement,
                                                                                            end_replacement,
                                                                                            input_ids_replacement)
        word_tokens_str_original, input_tokens_str_original = self._get_token_strings(sentence_original, start_original,
                                                                                      end_original, input_ids_original)

        # search the occurrence of 'word_tokens_str' in 'input_tokens_str' to get the corresponding position
        matches_replacement, pos_offset, pos_error, pos = list(), 0, None, None
        while True:
            tmp = input_tokens_str_replacement[pos_offset:]
            match_replacement = re.search(f"( +|^){word_tokens_str_replacement}(?!\w+| {self.subword_prefix})",
                                          input_tokens_str_replacement, re.DOTALL)

            if match_replacement is None:
                break

            current_pos = pos_offset + match_replacement.start()
            current_error = abs(current_pos - batch_offset[j]['start'])

            if pos is None or current_error < pos_error:
                pos = current_pos
                pos_error = current_error
            else:
                break

            pos_offset += match_replacement.end()
            matches_replacement.append(match_replacement)
        pos_replacement = pos

        matches_original, pos_offset, pos_error, pos = list(), 0, None, None
        while True:
            tmp = input_tokens_str_original[pos_offset:]
            match_original = re.search(f"( +|^){word_tokens_str_original}(?!\w+| {self.subword_prefix})",
                                       input_tokens_str_original, re.DOTALL)

            if match_original is None:
                break

            current_pos = pos_offset + match_original.start()
            current_error = abs(current_pos - batch_offset[j]['start'])

            if pos is None or current_error < pos_error:
                pos = current_pos
                pos_error = current_error
            else:
                break

            pos_offset += match_original.end()
            matches_original.append(match_original)
        pos_original = pos

        # Truncation side effect: the target word is over the maximum input length
        if len(matches_replacement) == 0 and len(matches_original) == 0:
            idx_replacement_sent = batch_tokenized_text.num_rows * i + j
            warnings.warn(
                f"An error occurred with the {idx_replacement_sent}-th sentence: {batch_text[j]}. It will be ignored",
                category=UserWarning)
            return None

        n_previous_tokens_replacement = len(
            input_tokens_str_replacement[:pos_replacement].split())  # number of tokens before that sub-word
        n_word_token_replacement = len(word_tokens_str_replacement.split())  # number of tokens of the target word

        n_previous_tokens_original = len(
            input_tokens_str_original[:pos_original].split())  # number of tokens before that sub-word
        n_word_token_original = len(word_tokens_str_original.split())  # number of tokens of the target word

        # token indexes
        start_replacement = n_previous_tokens_replacement  # ind sub-word position
        end_replacement = n_previous_tokens_replacement + n_word_token_replacement

        # token indexes
        start_original = n_previous_tokens_original  # ind sub-word position
        end_original = n_previous_tokens_original + n_word_token_original

        return list(range(start_replacement, end_replacement)), list(range(start_original, end_original))
