import os
import json
import argparse
from pathlib import Path

def convert(dataset):
    records = list()
    with open(dataset, mode='r', encoding='utf-8') as f:
        for i, json_line in enumerate(f):
            if json_line.strip() == '': continue  # avoid blank lines
            record = json.loads(json_line)
            if record is None: continue  # avoid empty record

            # first line is the synthetic sentence, second line is the original one
            if (i+1)%2==0:
                records.append(json.dumps(record)+'\n')
                continue
            record['lemma'] = 'unkrand'
            record['token'] = 'unkrand'
            record['sentence'] = record['sentence'][:record['start']] + 'unkrand' + record['sentence'][record['end']:]
            record['end'] = record['start'] + len(record['token'])
            records.append(json.dumps(record)+'\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Create replacement files with unknown token: unkrand', add_help=True)
    parser.add_argument('-d', '--dir',
                        default='replacements',
                        type=str,
                        help='Directory containing Lexical Replacement datasets')
    args = parser.parse_args()

    # get all replacement files
    paths = list(Path(args.dir).glob("*.txt"))

    for filename in paths:
        if str(filename).endswith('random.txt') or 'random_' not in str(filename): continue
        semantic_class, pos = os.path.basename(str(filename))[:-4].split('_')
        semantic_class = 'unknown'
        open(f'{args.dir}/{semantic_class}_{pos}.txt', mode='w', encoding='utf-8').writelines(convert(filename))
