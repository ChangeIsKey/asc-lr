import json
import string
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

random.seed(42)

# create dir
Path('WiC/dwug_de').mkdir(parents=True, exist_ok=True)

# sentence pair
s1 = list() # first sentence list
s2 = list() # second sentence list

for f_j, f_u in zip(Path(f'dwug_de_tmp/data/').glob('**/judgments.csv'), Path(f'dwug_de_tmp/data/').glob('**/uses.csv')):
    # File judgment, File uses
    f_j, f_u = str(f_j), str(f_u)

    # Lemma
    lemma = f_j.split('/')[2]

    # part of speech
    if lemma.istitle():
        pos = 'N'
    elif not lemma in ['abgebrüht', 'weitgreifend']:
        pos = 'V'
    else:
        continue

    # Uses
    uses = open(f_u, mode='r', encoding='utf-8').readlines()
    columns = uses[0][:-1].split('\t')
    uses_dict = dict()
    for i, row in enumerate(uses[1:]):
        row = dict(zip(columns, row[:-1].split('\t')))
        try:
            start, end = row['indexes_target_token'].split(':')
        except:
            start = row['indexes_target_token'][0]
            end = row['context'][int(start):].strip()
            # safe check
            if end[-1] in string.punctuation:
                end = int(start) + len(end[:-1])
            else:
                end = int(start) + len(end)

        start, end = int(start), int(end)
        # new use record
        uses_dict[row['identifier']] = dict(lemma=lemma,
                                            pos=pos,
                                            token=row['context'][int(start):int(end)],
                                            start=int(start), end=int(end),
                                            sent=row['context'],
                                            grouping=row['grouping'])

    # Judgments
    judgments_df = open(f_j, mode='r', encoding='utf-8').readlines()
    columns = judgments_df[0][:-1].split('\t')

    # get number of judgemnts for each pair
    judgements_count = defaultdict(lambda: defaultdict(int))
    for i, row in enumerate(judgments_df[1:]):
        row = dict(zip(columns, row[:-1].split('\t')))

        # idx pair
        idx_sorted = sorted([row['identifier1'], row['identifier2']])

        # judgment score
        score = row['judgment']

        # store info
        judgements_count[idx_sorted[0] + ' ' + idx_sorted[1]][score] += 1

    # assign binary label according to the maximum judgment agreement
    for k in list(judgements_count.keys()):
        judgments = list(judgements_count[k].keys())
        counts = list(judgements_count[k].values())
        n_annotators = sum(counts)

        # for the sake of quality we do not rely to single evaluation
        if n_annotators < 2:
            continue

        # get max agreement
        #idx = np.argmax(counts)
        #max_value = counts[idx]

        # number of maximum
        # for the sake of quality we do not rely on tie evaluation
        #n_max = len([v for v in counts if counts.count(max_value) == counts.count(v)])
        #if n_max > 1:
        #    continue

        #judgment = int(eval(judgments[idx]))
        judgment = np.array([int(eval(j)) for j in judgments]).mean()

        if 3.5<= judgment <=4: #== 4:  # in [3, 4]:
            gold = 1
        elif 1<=judgment <=1.5: #1:  # in [1, 2]:
            gold = 0
        else:
            continue
        
        identifier1, identifier2 = k.split()

        token1 = uses_dict[identifier1]['token']
        sent1 = uses_dict[identifier1]['sent']
        pos1 = uses_dict[identifier1]['pos']
        start1 = sent1.find(token1)
        end1 = start1 + len(token1)
        gold1 = gold

        token2 = uses_dict[identifier2]['token']
        sent2 = uses_dict[identifier2]['sent']
        pos2 = uses_dict[identifier2]['pos']
        start2 = sent2.find(token2)
        end2 = start2 + len(token2)
        gold2=gold


        # bert generate errors for this sentences due to accents, length, ...
        errors = ['Mit der Arbeit des Fremdwörterbuchs bin ich seit dem Februar zu Ende, aber – aus diesem Arbeitsjoch',
                  'Das buch ist so geordnet, dass zuerst die gedichte in dem gewöhnlichen skolientone stehn',
                  'Einsam, einsam Will ich wandeln und ziehen, Ob fiebernde Brunst auch Die Adern emporschwellt',
                  '438. Muschelmilbe I. 494. Muschelthiere I. 274. Muscicapa albicollis II. 333. Muscicapida II. 333. Musci',
                  'die grosse französische Messung zur Bestimmung des mêtre man hatte',
                  'Jch schleich’ mich hinein — versteht sich mit dem g’spannten Zwilling',
                  'Dodo ’s II. 329. Dolabella I. 339. Dolerus I. 695. Dolichopida I. 608. Dolichopus I. 595. 608. Dolichurus I. 697. Dolium I. 351. Dolomedes I',
                  'Musik und immer gute Laune" zum Beispiel acht Streicher der Dresdner',
                  'so schwer und so zahlreich sind ihre verbrechen ”. wer so a',
                  'Der letztere war übrigens nicht blos der ältere, sondern auch',
                  'Nachfolgende Fremdwörter mit logischer Bedeutung sind meist germanisiert']
      
        flag_error = False
        for error in errors:
            if error in sent1 or error in sent2:
                flag_error=True
                break
        if flag_error:
            continue
        
        
        s1.append(dict(lemma=lemma, token=token1,
                       start=start1, end=end1,
                       pos=pos1,
                       sentence=sent1, gold=gold1))
        s2.append(dict(lemma=lemma, token=token2,
                       start=start2, end=end2,
                       pos=pos2,
                       sentence=sent2, gold=gold2))

idx = list(range(0, len(s1)))
random.shuffle(idx)
s1 = np.array(s1)[idx]
s2 = np.array(s2)[idx]

percentage = 0.6
n_train = int(len(s1) * percentage)

percentage = 0.2
n_dev = int(len(s1) * percentage)
n_test = n_dev

print(0, n_train)
tokenization = list()
for i in range(0, n_train):
    record = s1[i]
    tokenization.append(json.dumps(record)+'\n')
    record = s2[i]
    tokenization.append(json.dumps(record)+'\n')
with open('WiC/dwug_de/train.txt', mode='w', encoding='utf-8') as f:
    f.writelines(tokenization)

print(n_train, n_train+n_test, n_test)
tokenization = list()
for i in range(n_train, n_train+n_test):
    record = s1[i]
    tokenization.append(json.dumps(record)+'\n')
    record = s2[i]
    tokenization.append(json.dumps(record)+'\n')
with open('WiC/dwug_de/test.txt', mode='w', encoding='utf-8') as f:
    f.writelines(tokenization)

print(n_train+n_test, n_train+n_test+n_dev, len(s1), n_dev)
tokenization = list()
for i in range(n_train+n_test, len(s1)):
    record = s1[i]
    tokenization.append(json.dumps(record)+'\n')
    record = s2[i]
    tokenization.append(json.dumps(record)+'\n')
with open('WiC/dwug_de/dev.txt', mode='w', encoding='utf-8') as f:
    f.writelines(tokenization)
