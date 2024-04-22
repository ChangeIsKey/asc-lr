import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib
matplotlib.rcParams.update({'font.size': 13})


def idx2pos(idx: np.array) -> np.array:
    '''Transform indexes to positions'''

    pos = list()
    for i in range(0, idx.shape[0], 2):
        # take only the first position
        # -1: since the cls token was removed
        pos.append(int(idx[i].split()[0]) - 1)
    return np.array(pos)


def lexical_replacement_distance(X: torch.tensor, pos: int) -> float:
    '''Return the cosine distance between a word and its replacement.

    Args:
        X(torch.tensor): cosine distances.
        pos(int): target word position.
    '''
    return X[pos, pos].item()


def context_replacement_distances(X: torch.tensor, pos: int, n_words: int = 0) -> float:
    '''Return the cosine distance between a word and its replacement.

    Args:
        X(torch.tensor): cosine distances.
        pos(int): target word position.
        n_words(int, default=0): number of context word considered. Default is 0 and means all.
    '''
    if n_words == 0:
        n_words = X.shape[0]

    cd = np.concatenate([X.diagonal()[:pos], X.diagonal()[pos + 1:]])

    if n_words != 0:
        cd = cd[max(pos - n_words, 0):min(pos + n_words, X.shape[0])]

    return cd


def all_lexical_replacement_distance(X: list, pos: np.array) -> np.array:
    '''Return the cosine distance array of a word and its replacement.

    Args:
        X(torch.tensor): cosine distances.
        pos(np.array): target words positions.
    '''

    cd = list()
    for i, x in enumerate(X):
        cd.append(lexical_replacement_distance(x, pos[i]))
    return np.array(cd)


def all_context_replacement_distances(X: list, pos: np.array, n_words: int = 0) -> list:
    '''Return the cosine distance array of a word and its replacement.

    Args:
        X(torch.tensor): cosine distances.
        pos(np.array): target words positions.
        n_words(int, default=0): number of context word considered. Default is 0 and means all.
    '''
    cd = list()
    for i, x in enumerate(X):
        cd.append(context_replacement_distances(x, pos[i].item(), n_words=n_words))
    return cd


def all_apd(X: list) -> np.array:
    '''Return the average cosine distances.

    Args:
        X(torch.tensor): cosine distances.
    '''

    cd = list()
    for x in X:
        cd.extend(x.flatten())
    return np.array(cd)


def get_baselines(model, baseline_filename: str = 'random', n_layers: int = 12, n_words: int = 0) -> np.array:
    '''Returns baselines.

    Args:
        baseline_filename(str, default='random'): file with random replacement.
        --> Random replacements represents the worst case.
        n_layers(int, default=12): number of model layers.
        n_words(int, default=0): number of context word considered. Default is 0 and means all.
    '''
    # path to file containing indexes
    idx_filename = f'{model}/target_index/{baseline_filename}.npy'

    # Load and convert indexes to position of targets
    idx_target = idx2pos(np.load(idx_filename))

    # baseline for word and context distance
    wd_baseline, cd_baseline = list(), list()
    layers = list(range(1, n_layers + 1))
    for layer in layers:
        # path to file containing baseline distances
        path = f'{model}/cosine_distances/{layer}/{baseline_filename}.pkl'

        # Load layer-specific cosine_distances
        with open(path, mode='rb') as f:
            cd_matrix = pickle.load(f)

        # Context distance baseline
        tmp = all_context_replacement_distances(cd_matrix, idx_target, n_words=n_words)
        # TODO
        # flat_tmp = np.array([item for l in tmp for item in l])
        flat_tmp = np.array([item.mean() for item in tmp])
        cd_baseline.append(flat_tmp.mean())

        # Word distance baseline
        tmp = all_lexical_replacement_distance(cd_matrix, idx_target)
        wd_baseline.append(tmp.mean())

    return np.array(wd_baseline), np.array(cd_baseline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-l', '--layers',
                        type=int,
                        default=12,
                        help='# layers')
    parser.add_argument('-c', '--context_words',
                        type=int,
                        default=5,
                        help='# Context words [if context distance is considered]')
    parser.add_argument('-m', '--model_list', nargs='+', default=['bert', 'xlmr', 'mbert'])
    args = parser.parse_args()


    # layer of the model
    n_layers = args.layers
    n_words = args.context_words
    
    layers = list(range(1, n_layers+1))


    # baselines
    wd_baseline, cd_baseline = dict(), dict()
    for model in args.model_list:
        wd_baseline[model], cd_baseline[model] = get_baselines(model, 'random', n_words=n_words)

    # part of speech
    pos = {'v':'.', 'r': '*', 'a':0, 'n':'p'}


    filenames = ['antonyms', 'synonyms', 'hypernyms', 'random', 'unknown']
    pos = ['v', 'r', 'a', 'n', 's']

    # colorblindness
    colors = {'antonyms': '#D55E00', 'synonyms': '#56B4E9', 'hypernyms': '#009E73', 'random': '#000000',
              'unknown': '#F0E442'}

    markers = ['.', '+', '*', 'o', '-', ',', 'v']
    models = {model: markers[i] for i, model in enumerate(args.model_list)}

    fig, axs = plt.subplots(len(models), 4, figsize=(13.1, len(models) * 3))
    metric = 'Self-embedding Distance'

    for j, model in enumerate(list(models)):
        min_, max_ = 1, 0
        dist = defaultdict(list)
        for filename in filenames:
            for i, p in enumerate(pos):
                values = list()
                for layer in layers:
                    try:
                        # path to file containing indexes
                        rd_filename = f'{model}/metrics/{layer}/rd_{filename}_{p}.npy'
                        cd_filename = f'{model}/metrics/{layer}/cd_{filename}_{p}-full.npy'
                        tmp_filename = f'{model}/metrics/{layer}/rd_unknown_{p}.npy' #rd_unknown_{p}.npy'
                        
                        # Load and convert indexes to position of targets
                        rd = np.load(rd_filename)
                        cd = np.load(cd_filename)
                        normalizer = np.load(tmp_filename).mean()
                    except:
                        # adverb and adjective do not have hypernyms
                        continue
    
                    if metric == 'Context Distance':
                        # Adjust distance by using baseline
                        values.append(cd.mean() / cd_baseline[model][layer-1])
                        min_ = min(min_, values[-1].min())
                        max_ = max(max_, values[-1].max())
    
                    else:
                        # Adjust distance by using baseline
                        values.append(rd.mean()) # / normalizer) #wd_baseline[model][layer - 1])
                        min_ = min(min_, values[-1].min())
                        max_ = max(max_, values[-1].max())
    
                if len(values) == 0: continue
    
                if i == 0:
                    axs[j][i].plot(layers, np.array(values), color=colors[filename],
                                   label=f'{model} - {filename.replace("unknown", "syntethic")}', marker=models[model],
                                   linewidth=0.05, linestyle='--')
                else:
                    axs[j][i].plot(layers, np.array(values),
                                   color=colors[filename],
                                   marker=models[model], linewidth=0.05, linestyle='--')
    
                title = ""
                if p == 'v':
                    title = 'verb'
                if p == 'a':
                    title = 'adjective'
                if p == 'r':
                    title = 'adverb'
                if p == 'n':
                    title = 'noun'
                axs[j][i].title.set_text(title)
    
        fig.suptitle(metric)
    
        axs[j][0].set_ylabel(f'Self-embedding distance\n')
    
        for i in range(0, 4):
            axs[j][i].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            axs[j][i].set_xlabel('Layer Index')
            axs[j][i].set_ylim(min_ - 0.05, max_ + 0.05)
            axs[j][i].set_xticks(layers)

    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(bbox_to_anchor=(0.69, -0.05), title=args.model_list[0].upper(), handles=h, labels=l)
    
    h, l = axs[1][0].get_legend_handles_labels()
    lgd = fig.legend(bbox_to_anchor=(0.49, -0.05), title=args.model_list[1].upper(), handles=h, labels=l)

    h, l = axs[2][0].get_legend_handles_labels()
    lgd = fig.legend(bbox_to_anchor=(0.29, -0.05), title=args.model_list[2].upper(), handles=h, labels=l)
        
    plt.tight_layout()
    plt.savefig('sed_plots.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
