import json

for s in ['dev', 'test', 'train']:
    data = open(f'wicita/{s}.jsonl', mode='r', encoding='utf-8')

    # wrapper
    records = list()
    for line in data.readlines():
        # blanck line
        if line.strip() == '':
            continue

        line = json.loads(line.strip())

        for j in range(0, 2):
            record = dict()
            sentence = line[f'sentence{j+1}']
            record['lemma'] = line['lemma']

            pos = line['id'].split('.')[1]
            if pos == 'noun':
                record['pos'] = 'N'
            elif pos == 'verb':
                record['pos'] = 'V'
            elif pos == 'adv':
                record['pos'] = 'R'
            elif pos == 'adj':
                record['pos'] = 'A'

            start, end = int(line[f'start{j+1}']), int(line[f'end{j+1}'])
            record['token'] = sentence[start:end]
            record['start'] = start
            record['end'] = end
            record['sentence'] = sentence
            record['gold'] = line['label']
            records.append(json.dumps(record) + '\n')

    with open(f'wicita/{s if s != "valid" else "dev"}.txt', mode='w', encoding='utf-8') as f:
        f.writelines(records)
