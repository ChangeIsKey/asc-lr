import os
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve


class WiC:
    def __init__(self, dataset: str, data_sets: list = ['test', 'dev', 'train']):
        """
        Args:
            dataset (str): dataset directoy
            data_sets (list, default=['test', 'dev', 'train']): data sets available
        Returns:
            dict of data sets loaded in dataframes"""
        self.dataset = dataset
        self.data_sets = data_sets

    def load_dataset(self) -> dict:
        """Load data sets"""

        df_data_sets = dict()
        for s in self.data_sets:
            filename = f'{self.dataset}/{s}.txt'
            rows = list()
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() == '': continue
                    row = json.loads(line)
                    if row is None: continue
                    rows.append(row)

            # filter relevant data
            tmp = pd.DataFrame(rows)
            df_data_sets[s] = pd.DataFrame()
            df_data_sets[s]['words'] = [tmp.lemma.values[i] for i in range(0, tmp.shape[0], 2)]
            df_data_sets[s]['pos'] = [tmp.pos.values[i] for i in range(0, tmp.shape[0], 2)]
            df_data_sets[s]['gold'] = [int(tmp['gold'].values[i]) for i in range(0, tmp.shape[0], 2)]

        return df_data_sets

    def load_embedding(self, model: str = 'bert-base-uncased'):
        """Load embeddings in RAM

        Args:
            model(str, default='bert-base-uncased'): bert model used
        """

        # embeddings wrapper
        embeddings = defaultdict(lambda: defaultdict(dict))

        # number of layers of the model
        n_model_layers = len(os.listdir(f'{self.dataset}/target_embeddings/{model}/test/'))

        for s in self.data_sets:
            for layer in range(1, n_model_layers + 1):
                # load embeddings
                filename = f'{self.dataset}/target_embeddings/{model}/{s}/{layer}.pt'
                E = torch.load(filename)

                # split embeddings for sentence1 and sentence2
                E1, E2 = list(), list()
                for i in range(0, E.shape[0], 2):
                    E1.append(E[i])
                    E2.append(E[i + 1])

                # embeddings in memory
                embeddings[s]['sent1'][layer] = torch.stack(E1)
                embeddings[s]['sent2'][layer] = torch.stack(E2)

        return embeddings

    def compute_similarities(self, embeddings: dict) -> dict:
        """Compute cosine similarities"""

        # number of layers of the model
        n_model_layers = len(embeddings['sent1'].keys())

        # wrapper for similarities
        scores = defaultdict(list)

        # number of pairs of the dataset
        n_pairs = embeddings['sent1'][1].shape[0]

        # i-th pair
        for i in range(n_pairs):
            E1, E2 = list(), list()

            # j-th layer
            for j in range(1, n_model_layers + 1):
                E1_layer_j, E2_layer_j = embeddings['sent1'][j][i].cpu(), embeddings['sent2'][j][i].cpu()
                E1.append(E1_layer_j)
                E2.append(E2_layer_j)

                # Cosine Similarity: single layer
                cs = 1 - cosine(E1_layer_j.numpy(), E2_layer_j.numpy())
                scores[f'CS{j}'].append(cs)

            # Cosine Similarity: average last 4 layers
            cs = 1 - cosine(torch.stack(E1[-4:]).mean(axis=0).numpy(),
                            torch.stack(E2[-4:]).mean(axis=0).numpy())
            scores[f'CS{j - 4}-{j}'].append(cs)

        # convert into numpy arrays
        for cs in scores:
            scores[cs] = np.array(scores[cs])

        return scores

    def set_threshold(self, y_true: np.array, y: np.array) -> float:
        """
        Find the threshold that maximize the area under the curve.

        Args:
            y(np.array): array containing predicted values
            y_true(np.array): array containing ground truth values.
        Returns:
            thr
        """

        # False Positive Rate - True Positive Rate
        fpr, tpr, thresholds = roc_curve(y_true, y)

        scores = []
        for thresh in thresholds:
            scores.append(f1_score(y_true, [m >= thresh for m in y],
                                   average='weighted'))  # roc_auc_score(y_true, [m >= thresh for m in y]))

        scores = np.array(scores)

        # Max accuracy
        max_ = scores.max()

        # Threshold associated to the maximum accuracy
        max_threshold = thresholds[scores.argmax()]

        return max_threshold

    def fit(self, model: str = 'bert-base-uncased') -> dict:
        # load dataset
        pair_data_sets = self.load_dataset()

        # compute scores
        embeddings = self.load_embedding(model)

        # number of layers of the model
        n_model_layers = len(embeddings['test']['sent1'].keys())

        # Wrapper
        dfs = defaultdict(lambda: defaultdict(dict))  # dataset wrapper
        embs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # embeddings wrapper
        cosine_similarities = defaultdict(lambda: defaultdict(dict))  # score wrapper

        # for each data set
        for s in self.data_sets:
            mask = defaultdict(list)  # pos mask

            # for each sentence pair
            for i in range(0, pair_data_sets[s].shape[0]):
                pos = pair_data_sets[s].pos.values[i]

                # update masks
                mask[pos].append(i)  # pos-specific
                mask['ALL'].append(i)  # total

            # for each pos
            for pos in mask:
                mask_pos = mask[pos]
                dfs[s][pos] = pair_data_sets[s].loc[mask_pos].reset_index(drop=True)

                # split embeddings for sentences1 and sentences2
                E1, E2 = embeddings[s].values()
                for layer in range(1, n_model_layers + 1):
                    embs[s][pos]['sent1'][layer] = E1[layer][mask[pos]]
                    embs[s][pos]['sent2'][layer] = E2[layer][mask[pos]]

                # compute scores
                scores_data_sets = self.compute_similarities(embs[s][pos])
                cosine_similarities[s][pos] = scores_data_sets

        # get all pos and all measures
        all_pos = np.unique(pair_data_sets['test'].pos.values).tolist() + ['ALL']
        all_measures = cosine_similarities['test'][all_pos[0]].keys()

        # wrapper stats
        stats = list()
        for pos in all_pos:
            for measure in all_measures:
                # data for a specific pos and measure
                record = dict(pos=pos, measure=measure)

                # binary ground truth
                if 'train' in self.data_sets:
                    binary_true_train = dfs['train'][pos]['gold'].values  # train
                binary_true_test = dfs['test'][pos]['gold'].values  # test
                binary_true_dev = dfs['dev'][pos]['gold'].values  # dev

                # graded scores
                if 'train' in self.data_sets:
                    scores_train = cosine_similarities['train'][pos][measure]  # train
                scores_test = cosine_similarities['test'][pos][measure]  # test
                scores_dev = cosine_similarities['dev'][pos][measure]  # dev

                # true positives and negatives
                if 'train' in self.data_sets:
                    train_tp = binary_true_train[binary_true_train == 1].shape[0]  # train 1
                    train_tn = binary_true_train[binary_true_train == 0].shape[0]  # train 1
                test_tp = binary_true_test[binary_true_test == 1].shape[0]  # test 1
                test_tn = binary_true_test[binary_true_test == 0].shape[0]  # test 0
                dev_tp = binary_true_dev[binary_true_dev == 1].shape[0]  # dev
                dev_tn = binary_true_dev[binary_true_dev == 0].shape[0]  # dev

                # accuracy and threshold: get the best threshold for dev set
                thr = self.set_threshold(binary_true_dev, scores_dev)  # dev

                # binary prediction through threshold
                if 'train' in self.data_sets:
                    train_preds = [int(i >= thr) for i in scores_train]  # train
                dev_preds = [int(i >= thr) for i in scores_dev]  # dev
                test_preds = [int(i >= thr) for i in scores_test]  # test

                # accuracy
                if 'train' in self.data_sets:
                    acc_train = accuracy_score(binary_true_train, train_preds)  # train
                acc_test = accuracy_score(binary_true_test, test_preds)  # test
                acc_dev = accuracy_score(binary_true_dev, dev_preds)  # test

                # f1-score
                if 'train' in self.data_sets:
                    f1_train = f1_score(binary_true_train, train_preds, average='weighted')  # train
                f1_test = f1_score(binary_true_test, test_preds, average='weighted')  # test
                f1_dev = f1_score(binary_true_dev, dev_preds, average='weighted')  # test

                # auc_roc
                if 'train' in self.data_sets:
                    roc_train = roc_auc_score(binary_true_train, train_preds)  # train
                roc_test = roc_auc_score(binary_true_test, test_preds)  # test
                roc_dev = roc_auc_score(binary_true_dev, dev_preds)  # test

                # store info
                if 'train' in self.data_sets:
                    record['acc_train'] = acc_train
                record['acc_dev'] = acc_dev
                record['acc_test'] = acc_test

                if 'train' in self.data_sets:
                    record['f1_train'] = f1_train
                record['f1_dev'] = f1_dev
                record['f1_test'] = f1_test

                if 'train' in self.data_sets:
                    record['roc_train'] = roc_train
                record['roc_dev'] = roc_dev
                record['roc_test'] = roc_test

                record['thr'] = thr

                if 'train' in self.data_sets:
                    record['train_TP'] = train_tp
                    record['train_TN'] = train_tn
                record['dev_TP'] = dev_tp
                record['dev_TN'] = dev_tn
                record['test_TP'] = test_tp
                record['test_TN'] = test_tn
                stats.append(record)

        # create dataframe
        stats = pd.DataFrame(stats)

        # set column order
        if 'train' in self.data_sets:
            column_order = ['pos', 'measure', 'acc_dev', 'acc_train', 'acc_test', 'roc_dev', 'roc_train', 'roc_test',
                            'f1_dev', 'f1_train', 'f1_test', 'dev_TP', 'dev_TN', 'train_TP', 'train_TN', 'test_TP',
                            'test_TN', 'thr']
        else:
            column_order = ['pos', 'measure', 'acc_dev', 'acc_test', 'roc_dev', 'roc_test',
                            'f1_dev', 'f1_test', 'dev_TP', 'dev_TN', 'test_TP', 'test_TN', 'thr']

        ########
        # use the best threshold of nouns for verbs
        new_stats = list()
        for measure in all_measures:
            # data for verb by using threshold of nouns - for each measure
            record = dict(pos='V_thr_N', measure=measure)

            # noun threshold
            thr = stats[(stats['measure'] == measure) & (stats['pos'] == 'N')].thr.values[0]  # noun

            # binary ground truth
            if 'train' in self.data_sets:
                binary_true_train = dfs['train']['V']['gold'].values  # train
            binary_true_test = dfs['test']['V']['gold'].values  # test
            binary_true_dev = dfs['dev']['V']['gold'].values  # dev

            # graded scores
            if 'train' in self.data_sets:
                scores_train = cosine_similarities['train']['V'][measure]  # train
            scores_test = cosine_similarities['test']['V'][measure]  # test
            scores_dev = cosine_similarities['dev']['V'][measure]  # dev

            # true positives and negativs
            if 'train' in self.data_sets:
                train_tp = binary_true_train[binary_true_train == 1].shape[0]  # train 1
                train_tn = binary_true_train[binary_true_train == 0].shape[0]  # train 0
            test_tp = binary_true_test[binary_true_test == 1].shape[0]  # test 1
            test_tn = binary_true_test[binary_true_test == 0].shape[0]  # test 0
            dev_tp = binary_true_dev[binary_true_dev == 1].shape[0]  # dev 1
            dev_tn = binary_true_dev[binary_true_dev == 0].shape[0]  # dev 0

            # binary predictions through threshold
            if 'train' in self.data_sets:
                train_preds = [int(i >= thr) for i in scores_train]  # train
            test_preds = [int(i >= thr) for i in scores_test]  # test
            dev_preds = [int(i >= thr) for i in scores_dev]  # dev

            # accuracy
            if 'train' in self.data_sets:
                acc_train = accuracy_score(binary_true_train, train_preds)  # train
            acc_test = accuracy_score(binary_true_test, test_preds)  # test
            acc_dev = accuracy_score(binary_true_dev, dev_preds)  # dev

            # f score
            if 'train' in self.data_sets:
                f1_train = f1_score(binary_true_train, train_preds, average='weighted')  # train
            f1_test = f1_score(binary_true_test, test_preds, average='weighted')  # test
            f1_dev = f1_score(binary_true_dev, dev_preds, average='weighted')  # dev

            # roc
            if 'train' in self.data_sets:
                roc_train = roc_auc_score(binary_true_train, train_preds)  # train
            roc_test = roc_auc_score(binary_true_test, test_preds)  # test
            roc_dev = roc_auc_score(binary_true_dev, dev_preds)  # test

            # store info
            if 'train' in self.data_sets:
                record['train_TP'] = train_tp
                record['train_TN'] = train_tn
                record['acc_train'] = acc_train
                record['f1_train'] = f1_train
                record['roc_train'] = roc_train
            record['dev_TP'] = dev_tp
            record['dev_TN'] = dev_tn
            record['f1_dev'] = f1_dev
            record['acc_dev'] = acc_dev
            record['test_TP'] = test_tp
            record['test_TN'] = test_tn
            record['acc_test'] = acc_test
            record['f1_test'] = f1_test
            record['thr'] = thr
            record['roc_test'] = roc_test
            record['roc_dev'] = roc_dev

            new_stats.append(record)

        # concatenate dataframe e order columns
        new_stats = pd.DataFrame(new_stats)
        new_stats = new_stats[column_order]
        stats = stats[column_order]
        stats = pd.concat([stats, new_stats])
        stats['layer'] = stats['measure'].apply(lambda x: x.replace('CS', ''))

        stats = stats.sort_values('f1_test', ascending=False)
        return stats
