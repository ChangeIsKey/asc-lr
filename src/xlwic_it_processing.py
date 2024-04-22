import json
import string
import warnings
from pathlib import Path

characters = string.ascii_letters+string.whitespace+'àéèìòù'

def process(pair):
    sent1, sent2 = pair[6], pair[7]
    start1, end1 = int(pair[2]), int(pair[3])
    start2, end2 = int(pair[4]), int(pair[5])

    # Firs sentence check
    if sent1[end1 - 1] not in characters:  # e.g., "hello," -> "hello" ,
        warnings.warn("Target is ill-tokenized in the first sentence", category=UserWarning)
        print(pair)
        return None

    if sent1[start1][0] not in characters: # e.g., ",hello" -> , "hello"
        warnings.warn("Target is ill-tokenized in the first sentence", category=UserWarning)
        print(pair)
        return None

    # index error: eg.
    # ['0', 'N', 14, 14, '22', 22, 'sono le ore 0  e 22 minuti.', "l'aereo arriverà alle 0 e 57."]
    if end1-start1 < 1:
        warnings.warn("Target is ill-tokenized in the first sentence", category=UserWarning)
        print(pair)
        return None

    # Second sentence check
    # e.g. Dobbiamo "costruire..". e non distruggere le nostre case -> "costruire"
    if sent2[end2 - 1] not in characters:  # e.g., "hello," -> "hello" ,
        warnings.warn("Target is ill-tokenized in the second sentence", category=UserWarning)
        print(pair)
        return None

    if sent2[start2][0] not in characters:  # e.g., ",hello" -> , "hello"
        warnings.warn("Target is ill-tokenized in the second sentence", category=UserWarning)
        print(pair)
        return None

    if end2 - start2 < 1:
        warnings.warn("Target is ill-tokenized in the second sentence", category=UserWarning)
        print(pair)
        return None

    return [str(i) for i in pair]


for s in ['test', 'valid', 'train']:
    if s == 'test':
        data = open(f'xlwic/it_test_data.txt', mode='r', encoding='utf-8').readlines()
        gold = open(f'xlwic/it_test_gold.txt', mode='r', encoding='utf-8').readlines()
    else:
        data = open(f'xlwic/it_{s}.txt', mode='r', encoding='utf-8').readlines()
        gold = [line.strip()[-1] for line in data]

    data = [line.strip() for line in data]
    gold = [int(line.strip()) for line in gold]

    # wrapper
    records = list()
    for i, pair in enumerate(data):
        pair = process(pair.split("\t"))

        # pair contains errors and can't be processed
        if pair is None:
            continue

        sentence1 = pair[6]
        start1, end1 = int(pair[2]), int(pair[3])

        sentence2 = pair[7]
        start2, end2 = int(pair[4]), int(pair[5])

        for j in range(0, 2):
            record = dict()
            record['lemma'] = pair[0]
            record['pos'] = pair[1]

            if j == 0:
                start, end = int(pair[2]), int(pair[3])
                sent = pair[6]
            else:
                start, end = int(pair[4]), int(pair[5])
                sent = pair[7]

            record['token'] = sent[start:end]
            record['start'] = start
            record['end'] = end
            record['sentence'] = sent
            record['gold'] = gold[i]
            records.append(json.dumps(record) + '\n')

    Path(f'xlwic_it').mkdir(parents=True, exist_ok=True)
    with open(f'xlwic_it/{s if s != "valid" else "dev"}.txt', mode='w', encoding='utf-8') as f:
        f.writelines(records)
