from nltk.corpus import wordnet as wn
import json
import random
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from tqdm import tqdm
import sys
import os


layer = int(sys.argv[1])

if not os.path.exists(f'stats_layers3/{layer}'):
    os.mkdir(f'stats_layers3/{layer}')

D = {}

with open('/mimer/NOBACKUP/groups/cik_data/datasets/LSC/SemEval-English/semeval2020_ulscd_eng/truth/binary.txt') as f:
    for line in f:
        word, label = line.split()
        label = int(label)
        target, pos = word.split('_')
        """
        synsets = []
        if pos == 'nn':
            synsets = wn.synsets(target,pos='n')
        elif pos == 'vb':
            synsets = wn.synsets(target,pos='v')
        replacements = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                lemma_name = ' '.join(lemma.name().split('_')).lower()
                if not lemma_name == target and not len(lemma_name.split())>1 and not len(lemma_name.split('-'))>1:
                    replacements.add(lemma_name)
                    break
        replacements = list(replacements)
        #if len(replacements) > 2:
        #    replacements = random.sample(replacements,2)
        """
        D[target] = pos

"""
lemmas = {'nn':set(), 'vb':set()}

for syn in wn.all_synsets():
    for lemma in syn.lemmas():
        lemma_name = ' '.join(lemma.name().split('_'))
        if syn.pos() == 'n':
            lemmas['nn'].add(lemma_name)
        elif syn.pos() == 'v':
            lemmas['vb'].add(lemma_name)
"""

replacements = {}
with open('replacements_random.txt','r') as f:
    for line in f:
        line = line[:-1].split('\t')
        replacements[line[0]] = set(['[RANDOM]'])#set(line[1:])
        """
        pos = 'v'
        if D[line[0]] == 'nn':
            pos = 'n'
        synsets = wn.synsets(line[0],pos=pos)
        hs = [h for s in synsets for h in s.hypernyms() ]
        for s in synsets+hs:
            for lemma in s.lemmas():
                replacements[line[0]].add(lemma.name().replace('_',' '))
        """

pretrained = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained)
tokenizer.add_tokens(['[RANDOM]'],special_tokens=True)
model = AutoModel.from_pretrained(pretrained, output_hidden_states=True)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda')
model.eval()
CLS = tokenizer.convert_tokens_to_ids('[CLS]')
PAD = tokenizer.convert_tokens_to_ids('[PAD]')
SEP = tokenizer.convert_tokens_to_ids('[SEP]')
MAX_LENGTH = 256
BATCH_SIZE = 64

for target in D:
    rep2distance = {}
    original_embs = {}

    with open(f'/mimer/NOBACKUP/groups/cik_data/cassotti/english_lemma_sentences/{target}.json','r') as f:
        t2s = json.load(f)
        T0 = []
        T1 = []
        for y in range(1810,1861):
            y = str(y)
            if y in t2s:
                T0 = T0 + [s for s in t2s[y] if len(s['text'])<=150]
        for y in range(1960,2011):
            y = str(y)
            if y in t2s:
                T1 = T1 + [s for s in t2s[y] if len(s['text'])<=150]
        random.shuffle(T0)
        random.shuffle(T1)
        T0 = T0[:200]
        T1 = T1[:200]

        for j,T in enumerate([T0,T1]):
            input_ids = []
            attention_mask = []
            token_type_ids = []
            idxs = []
            sep_idxs = []
            for sent in T:
                start, end = 0,0
                tokens = [CLS]
                if len(sent['text'][:sent['start']]) > 0:
                    tokens = tokens + tokenizer.encode(sent['text'][:sent['start']])[1:-1]
                start = len(tokens)
                tokens = tokens + tokenizer.encode(sent['text'][sent['start']:sent['end']])[1:-1]
                end = len(tokens)
                if len(sent['text'][sent['end']:]) > 0:
                    tokens = tokens + tokenizer.encode(sent['text'][sent['end']:])[1:-1]

                tokens = tokens + [SEP]
                sep_idxs.append(len(tokens)-1)
                att = [1] * len(tokens)  + [0] * (MAX_LENGTH - len(tokens))
                typ = [0] * len(att)
                tokens = tokens + [PAD] * (MAX_LENGTH - len(tokens))
                input_ids.append(tokens)
                attention_mask.append(att)
                token_type_ids.append(typ)
                idxs.append([start,end])

            embs = {'target':[], 'context':[]}
            for ndx in range(0,len(input_ids),BATCH_SIZE):
                idxs_batch = idxs[ndx:min(ndx + BATCH_SIZE, len(input_ids))]
                sep_batch = sep_idxs[ndx:min(ndx + BATCH_SIZE, len(input_ids))]
                model_input = {
                    'input_ids' : torch.tensor(input_ids[ndx:min(ndx+BATCH_SIZE,len(input_ids))]).to('cuda'),
                    'attention_mask' : torch.tensor(attention_mask[ndx:min(ndx+BATCH_SIZE,len(input_ids))]).to('cuda'),
                    'token_type_ids' : torch.tensor(token_type_ids[ndx:min(ndx+BATCH_SIZE,len(input_ids))]).to('cuda')
                }
                output = model(**model_input)
                hidden_states = torch.stack(output['hidden_states']).detach().cpu()[layer]
                for i in range(hidden_states.shape[0]):
                    if idxs_batch[i][1]-idxs_batch[i][0] > 1:
                        embs['target'].append(np.mean(hidden_states[i][idxs_batch[i][0]:idxs_batch[i][1]].detach().cpu().numpy(), axis=0))
                    else:
                        embs['target'].append(hidden_states[i][idxs_batch[i][0]:idxs_batch[i][1]].detach().cpu().numpy().squeeze())
                    left = hidden_states[i][1:idxs_batch[i][0]]
                    right = hidden_states[i][idxs_batch[i][1]:sep_batch[i]]
                    st = torch.cat((left,right)).detach().cpu().numpy()
                    if len(st) > 1:
                        embs['context'].append(np.mean(st,axis=0))
                    else:
                        embs['context'].append(st.squeeze())
            original_embs[j] = embs

    for replacement in tqdm(replacements[target]):
        new_embs = {}

        for j, T in enumerate([T0, T1]):
            input_ids = []
            attention_mask = []
            token_type_ids = []
            idxs = []
            sep_idxs = []
            for sent in T:
                start, end = 0, 0
                tokens = [CLS]
                if len(sent['text'][:sent['start']]) > 0:
                    tokens = tokens + tokenizer.encode(sent['text'][:sent['start']])[1:-1]
                start = len(tokens)
                tokens = tokens + tokenizer.encode(replacement)[1:-1]
                end = len(tokens)
                if len(sent['text'][sent['end']:]) > 0:
                    tokens = tokens + tokenizer.encode(sent['text'][sent['end']:])[1:-1]

                tokens = tokens + [SEP]
                sep_idxs.append(len(tokens)-1)
                att = [1] * len(tokens) + [0] * (MAX_LENGTH - len(tokens))
                typ = [0] * len(att)
                tokens = tokens + [PAD] * (MAX_LENGTH - len(tokens))
                input_ids.append(tokens)
                attention_mask.append(att)
                token_type_ids.append(typ)
                idxs.append([start, end])

            embs = {'target':[], 'context':[]}
            for ndx in range(0, len(input_ids), BATCH_SIZE):
                idxs_batch = idxs[ndx:min(ndx + BATCH_SIZE, len(input_ids))]
                sep_batch = sep_idxs[ndx:min(ndx + BATCH_SIZE, len(input_ids))]
                model_input = {
                    'input_ids': torch.tensor(input_ids[ndx:min(ndx + BATCH_SIZE, len(input_ids))]).to('cuda'),
                    'attention_mask': torch.tensor(attention_mask[ndx:min(ndx + BATCH_SIZE, len(input_ids))]).to('cuda'),
                    'token_type_ids': torch.tensor(token_type_ids[ndx:min(ndx + BATCH_SIZE, len(input_ids))]).to('cuda')
                }
                output = model(**model_input)
                hidden_states = torch.stack(output['hidden_states']).detach().cpu()[layer]
                for i in range(hidden_states.shape[0]):
                    if idxs_batch[i][1]-idxs_batch[i][0] > 1:
                        embs['target'].append(np.mean(hidden_states[i][idxs_batch[i][0]:idxs_batch[i][1]].detach().cpu().numpy(), axis=0))
                    else:
                        embs['target'].append(hidden_states[i][idxs_batch[i][0]:idxs_batch[i][1]].detach().cpu().numpy().squeeze())
                    left = hidden_states[i][1:idxs_batch[i][0]]
                    right = hidden_states[i][idxs_batch[i][1]:sep_batch[i]]
                    st = torch.cat((left,right)).detach().cpu().numpy()
                    if len(st) > 1:
                        embs['context'].append(np.mean(st,axis=0))
                    else:
                        embs['context'].append(st.squeeze())
            new_embs[j] = embs

        rep2distance[replacement] = [[],[]]
        for j in range(0,2):
            distance = np.diag(cosine_similarity(original_embs[j]['target'],new_embs[j]['target']))
            #distance2 = np.diag(cosine_similarity(original_embs[j]['context'],new_embs[j]['context']))
            rep2distance[replacement][j] = [distance]#[distance,distance2]

    #for t_idx,type in enumerate(['target','context']):
    for t_idx, type in enumerate(['target']):
        for j in range(0,2):
            with open(f'stats_layers3/{layer}/{target}_{j}_{type}.txt','w+') as f:
                for rep in rep2distance:
                    vec = '\t'.join([str(v) for v in rep2distance[rep][j][t_idx]])
                    f.write(f'{rep}\t{vec}\n')