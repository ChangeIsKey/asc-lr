import json

# data sets available
for s in ['dev', 'train', 'test']:
    data = open(f'wic_en/{s}.data.txt', mode='r', encoding='utf-8').readlines()
    gold = open(f'wic_en/{s}.gold.txt', mode='r', encoding='utf-8').readlines()

    records = list()
    for line, line_gold in zip(data, gold):
        line = line[:-1].split('\t')
        gold_value = int(line_gold.strip() == 'T')

        index1, index2 = line[2].split('-') # target word position in the sentence (i.e., i-th token)
        tokens1, tokens2 = line[3].split(), line[4].split()
        n_previous_token1 = " ".join(tokens1[:int(index1)])
        n_previous_token2 = " ".join(tokens2[:int(index2)])

        start1 = len(n_previous_token1)
        start2 = len(n_previous_token2)

        if len(n_previous_token1) > 0:
            start1+=1
        if len(n_previous_token2) > 0:
            start2+=1

        record = dict()
        record['lemma'] = line[0]
        record['pos'] = line[1]
        record['sentence'] = line[3]
        record['start'] = start1
        record['end'] = start1 + len(tokens1[int(index1)])
        record['token'] = record['sentence'][record['start']:record['end']]
        record['gold'] = gold_value
        records.append(json.dumps(record)+'\n')

        record = dict()
        record['lemma'] = line[0]
        record['pos'] = line[1]
        record['sentence'] = line[4]
        record['start'] = start2
        record['end'] = start2 + len(tokens2[int(index2)])
        record['token'] = record['sentence'][record['start']:record['end']]
        record['gold'] = gold_value
        records.append(json.dumps(record)+'\n')

    with open(f'wic_en/{s}.txt', mode='w', encoding='utf-8') as f:
        f.writelines(records)
