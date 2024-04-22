import json
from pathlib import Path

for lang, sets in zip(['en', 'fr'], [['dev', 'train', 'test'], ['dev', 'test']]):
    for s in sets:
        data = json.load(open(f'mclwic/{s}.{lang}-{lang}.data', mode='r', encoding='utf-8'))
        gold = json.load(open(f'mclwic/{s}.{lang}-{lang}.gold', mode='r', encoding='utf-8'))

        records = list()

        for i, pair in enumerate(data):
            for j in range(1, 3):
                record=dict()
                record['lemma']=pair['lemma']

                # different pos available
                if pair['pos'] == 'NOUN':
                    record['pos'] = 'N'
                elif pair['pos'] == 'VERB':
                    record['pos'] = 'V'
                elif pair['pos'] == 'ADJ':
                    record['pos'] = 'A'
                elif pair['pos'] == 'ADV':
                    record['pos'] = 'R'

                record['token']=pair[f'sentence{j}'][int(pair[f'start{j}']):int(pair[f'end{j}'])]
                record['start']=int(pair[f'start{j}'])
                record['end']=int(pair[f'end{j}'])
                record['sentence']=pair[f'sentence{j}']
                record['gold']=int(gold[i]['tag']=='T')
                records.append(json.dumps(record)+'\n')

        Path(f'mclwic_{lang}').mkdir(parents=True, exist_ok=True)
        with open(f'mclwic_{lang}/{s}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(records)
