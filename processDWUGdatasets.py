import os
import csv
import string
import pandas as pd
from pathlib import Path


def indexes_target_token(row: dict) -> tuple:
    '''Return the position (start, end) of the token in the sentence'''
    
    try:
        start, end = [int(i) for i in row['indexes_target_token'].split(':')]
    except:  # end is missing (e.g. '5:')
        start = int(row['indexes_target_token'][0])
        end = len(row['context'].strip())

        if row['context'].strip()[-1] in string.punctuation:  # safe check on last character
            end = end - 1

    return start, end


def part_of_speech(row, benchmark) -> dict:
    '''Change part-of-speech'''
    if benchmark == 'DWUG-English':
        if row['lemma'].endswith('_vb'):
            row['lemma'] = row['lemma'][:-3]
            row['pos'] = 'V'
        elif row['lemma'].endswith('_nn'):
            row['lemma'] = row['lemma'][:-3]
            row['pos'] = 'N'
        return row

    if benchmark in ['DURel-German', 'DWUG-German', 'RefWUG-German', 'SURel-German']:
        if row['pos'] in ['NN', 'NE']:
            row['pos'] = 'N'
        elif row['pos'] in ['ADJA', 'ADJD']:
            row['pos'] = 'A'
        elif row['pos'] in ['VVPP', 'VVFIN', 'VVINF', 'VVPP', 'VVIZU', 'VVIMP']:
            row['pos'] = 'V'
        if row['pos'] == 'FM':
            row['pos'] = 'foreign words'
        return row

    if row['pos'] == 'VERB':
        row['pos'] = 'V'
    elif row['pos'] == 'NOUN':
        row['pos'] = 'N'
    elif row['pos'] == 'ADJ':
        row['pos'] = 'A'
    elif row['pos'] == 'ADV':
        row['pos'] = 'R'
    return row

import argparse

parser = argparse.ArgumentParser(prog='Processing zenodo LSC benchmark', add_help=True)
parser.add_argument('-b', '--benchmark_folder',
                    type=str,
                    help='Folder of the benchmark')
parser.add_argument('-c', '--corpora_folder',
                    type=str,
                    help='Folder of the text corpora')
parser.add_argument('-t', '--tokenization_folder',
                    type=str,
                    help='Folder of the processed benchmark')
args = parser.parse_args()

# time interval <corpus1, corpus2>
benchmark = dict(corpus1=list(), corpus2=list())

counter = 0  # sentence id
print(args.benchmark_folder)
for target in os.listdir(f'{args.benchmark_folder}/data'):

    # load file
    uses_filename = f'{args.benchmark_folder}/data/{target}/uses.csv'  # uses
    uses = open(uses_filename, mode='r', encoding='utf-8').readlines()
    columns = uses[0][:-1].split('\t')

    # split uses
    corpus1, corpus2 = list(), list()
    for i, row in enumerate(uses[1:]):
        row = dict(zip(columns, row[:-1].split('\t')))

        # position of the target word
        start, end = indexes_target_token(row)

        # pos
        row = part_of_speech(row, args.benchmark_folder)

        token = row[f'context'][start:end]
        left = row[f'context'][:start]
        right = row[f'context'][end:]
        sentence = left + ' ' + token.replace('’', ' ').replace('„', ' ') + ' ' + right
        start, end = start + 1, end + 1
        lemma = row['lemma']
        pos = row['pos']

        record = dict(sentidx=counter,
                      lemma=lemma,
                      token=token,
                      start=start, end=end,
                      sentence=sentence,
                      pos=pos)
        counter += 1

        if row['grouping'] == '1700-1916' and args.benchmark_folder in ['DWUG-RuShiftEval12-Russian',
                                                                        'DWUG-RuShiftEval13-Russian'] or row[
            'grouping'] == '1918-1990' and args.benchmark_folder == 'DWUG-RuShiftEval23-Russian':
            row['grouping'] = 1
        elif args.benchmark_folder in ['DWUG-RuShiftEval12-Russian', 'DWUG-RuShiftEval23-Russian',
                                       'DWUG-RuShiftEval13-Russian']:
            row['grouping'] = 2

        if row['grouping'] == '1929-1965' and args.benchmark_folder == 'NorDiaChange12-Norwegian' or row[
            'grouping'] == '1980-1990' and args.benchmark_folder == 'NorDiaChange23-Norwegian':
            row['grouping'] = 1
        elif args.benchmark_folder in ['NorDiaChange12-Norwegian', 'NorDiaChange23-Norwegian']:
            row['grouping'] = 2

        if int(row['grouping']) == 1:
            corpus1.append(record)
        else:
            corpus2.append(record)

    # only sentences
    benchmark['corpus1'].extend([record['sentence'] + '\n' for record in corpus1])
    benchmark['corpus2'].extend([record['sentence'] + '\n' for record in corpus2])

    # store tokenization
    for corpus in ['corpus1', 'corpus2']:
        folder = f'{args.tokenization_folder}/{args.benchmark_folder}/{corpus}/token'
        Path(folder).mkdir(parents=True, exist_ok=True)
        output = f'{folder}/{lemma}.txt'

        df = pd.DataFrame(eval(corpus))
        df.to_json(output, orient='records', lines=True)

# store diachronic corpus
for corpus in ['corpus1', 'corpus2']:
    folder = f'{args.corpora_folder}/{args.benchmark_folder}/{corpus}/token/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f'{folder}/{corpus}.txt', mode='w+', encoding='utf-8') as f:
        f.writelines(benchmark[f'{corpus}'])

if args.benchmark_folder in ['DURel-German',
                             'SURel-German',
                             'DWUG-RuSemShift12-Russian',
                             'DWUG-RuSemShift23-Russian',
                             'DWUG-RuShiftEval12-Russian',
                             'DWUG-RuShiftEval23-Russian',
                             'DWUG-RuShiftEval13-Russian',
                             'NorDiaChange12-Norwegian',
                             'NorDiaChange23-Norwegian']:
    df = pd.read_csv(f'{args.benchmark_folder}/stats/stats_groupings.csv', sep='\t', quoting=csv.QUOTE_NONE)
    targets = [word.replace('_nn', '').replace('_vb', '') + '\n' for word in df['lemma'].values]
    with open(f'{args.corpora_folder}/{args.benchmark_folder}/targets.txt', mode='w+', encoding='utf-8') as f:
        f.writelines(targets)

    df['lemma'] = targets
    df_subtask2 = df[['lemma', 'COMPARE']]

    folder = f'{args.corpora_folder}/{args.benchmark_folder}/truth/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    df_subtask2.to_csv(f'{folder}/graded.txt', sep='\t', index=False, header=False)

else:
    df = pd.read_csv(f'{args.benchmark_folder}/stats/opt/stats_groupings.csv', sep='\t', quoting=csv.QUOTE_NONE)
    targets = [word.replace('_nn', '').replace('_vb', '') + '\n' for word in df['lemma'].values]
    with open(f'{args.corpora_folder}/{args.benchmark_folder}/targets.txt', mode='w+', encoding='utf-8') as f:
        f.writelines(targets)

    df['lemma'] = targets
    df_subtask1 = df[['lemma', 'change_binary']]
    df_subtask2 = df[['lemma', 'change_graded']]

    folder = f'{args.corpora_folder}/{args.benchmark_folder}/truth/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    df_subtask1.to_csv(f'{folder}/binary.txt', sep='\t', index=False, header=False)
    df_subtask2.to_csv(f'{folder}/graded.txt', sep='\t', index=False, header=False)
