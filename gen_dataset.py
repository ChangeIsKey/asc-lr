import numpy as np
from transformers import AutoTokenizer
import os
import json
import pickle

def get_errors():
    D = {}
    with open('errors.txt') as f:
        for line in f:
            line = line[:-1].split('\t')
            if not line[0] in D:
                D[line[0]] = set()
            D[line[0]].add(int(line[1]))
    return D


def gen_tokens():
    E = get_errors()
    pretrained = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    types = ['antonyms','hypernyms','polarity','random','synonyms']

    for fname in os.listdir('replacements'):
        with open(f'replacements/{fname}', 'r') as f:
            fbase = '.'.join(fname.split('.')[:-1])
            if not fbase in E:
                E[fbase] = set()
            with open(f'replacements_tokens/{fbase}_left.txt', 'w+') as g:
                with open(f'replacements_tokens/{fbase}_right.txt', 'w+') as h:
                    with open(f'replacements_tokens/{fbase}_target.txt', 'w+') as h2:
                        #pos_tag = fname.split('_')[1].split('.')[0]
                        sentences = []
                        originals = []
                        for j,line in enumerate(f):
                            if (j%2 == 0 and not int(j/2) in E[fbase]) or (j%2 == 1 and not int(j/2)-1 in E[fbase]):
                                if j % 2 == 0:
                                    sentence = {}
                                    line = json.loads(line)
                                    sentence['text'] = line['sentence']
                                    sentence['start'] = line['start']
                                    sentence['end'] = line['end']
                                    sentence['token'] = line['token']
                                    sentences.append(sentence)
                                else:
                                    line = json.loads(line)
                                    originals.append(line['token'])
                        for j,sent in enumerate(sentences):
                            left = tokenizer.encode(sent['text'][:sent['start']])[1:-1]
                            right = tokenizer.encode(sent['text'][sent['end']:])[1:-1]
                            left = '\t'.join(tokenizer.convert_ids_to_tokens(left))
                            g.write(f'{left}\n')
                            right = '\t'.join(tokenizer.convert_ids_to_tokens(right))
                            h.write(f'{right}\n')
                            h2.write(f'{originals[j]}\t{sent["token"]}\n')


def extract_distances(model,layer):
    E = get_errors()

    errors = 0

    #with open(f'errors.txt', 'w+') as g:
    for fname in os.listdir(f'{model}/cosine_distances/{layer}'):
        left_tokens = []
        right_tokens = []

        fbase = '.'.join(fname.split('.')[:-1])
        if not fbase in E:
            E[fbase] = set()

        with open(f'replacements_tokens/{fbase}_left.txt', 'r') as f:
            for line in f:
                line = line[:-1].split('\t')
                left_tokens.append(line)

        with open(f'replacements_tokens/{fbase}_right.txt', 'r') as f:
            for line in f:
                line = line[:-1].split('\t')
                right_tokens.append(line)

        with open(f'{model}/cosine_distances/{layer}/{fname}','rb') as f:
            with open(f'replacements_distances/{layer}/{fbase}_left.txt', 'w+') as g:
                with open(f'replacements_distances/{layer}/{fbase}_right.txt', 'w+') as h:
                    with open(f'replacements_distances/{layer}/{fbase}_target.txt', 'w+') as h2:
                        sent_list = pickle.load(f)
                        new_name = fname.replace('.pkl','.npy')
                        idxs = np.load(f'{model}/target_index/{new_name}')
                        idxs = [idxs[i] for i in range(len(idxs)) if i%2==0]
                        a_j = 0
                        for j in range(len(idxs)):
                            if not j in E[fbase]:
                                target_idx = int(idxs[j].split()[0]) - 1
                                wd = sent_list[j][target_idx][target_idx]
                                ctx_dist = np.diag(sent_list[j])
                                left_dist = '\t'.join([str(v) for v in ctx_dist[:target_idx].tolist()])
                                right_dist = '\t'.join([str(v) for v in ctx_dist[target_idx+1:].tolist()])
                                #if not len(right_dist) == len(right_tokens[a_j]) or not len(left_dist) == len(left_tokens[a_j]):
                                #    print('ERROR')
                                g.write(f'{left_dist}\n')
                                h.write(f'{right_dist}\n')
                                h2.write(f'{wd}\n')
                                a_j = a_j + 1

extract_distances('bert',1)
extract_distances('bert',6)
extract_distances('bert',12)
#gen_tokens()