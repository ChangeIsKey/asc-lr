from nltk.corpus import wordnet as wn
import os
import json
import wn as swn
import pandas as pd
import joypy
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


swn.download('odenet')



for data_set in ['train', 'test', 'dev']:
    languages = {}
    for dirname in os.listdir('WiC'):
        if len(dirname.split('_')) > 1:
            _,lang = dirname.split('_')
        else:
            lang = 'it'
        print(dirname,lang)
        postags = {}
        if data_set == 'train' and 'fr' == lang: continue
        with open(f'WiC/{dirname}/{data_set}.txt') as f:
            for line in f:
                line = json.loads(line)
                lemma = line['lemma']
                pos = line['pos'].lower()
                if not pos in postags:
                    postags[pos] = set()
                postags[pos].add(lemma)
        if not lang in languages:
            languages[lang] = postags
        else:
            for postag in languages[lang]:
                if postag in postags:
                    languages[lang][postag] = languages[lang][postag].union(postags[postag])


    lang2code = {'it':'ita','en':'eng','fr':'fra','de':'ger'}
    for language in languages:
        df = list()
        poscount = {}
        for postag in sorted(languages[language]):
            count = {}
            for word in languages[language][postag]:
                if not language == 'de':
                    n = wn.synsets(word,pos=postag,lang=lang2code[language])
                else:
                    n = swn.synsets(word, pos=postag, lang=language)
                if not len(n) in count:
                    count[len(n)] = 1
                else:
                    count[len(n)] = count[len(n)] + 1
                poscount[postag] = count

        print(f'{language}')

        columns = 'N.senses\tpos\tcount\tdata_set'.split('\t')
    
        pos_sums = {}
        for pos in poscount:
            pos_sums[pos] = sum(poscount[pos].values())
        for j in range(20):
            if 'n' in poscount and j in poscount['n']:
                nouns = round(poscount['n'][j] / pos_sums['n'],2)
                nouns_2 = poscount['n'][j]
            else:
                nouns = 0
                nouns_2 = 0
            if 'v' in poscount and j in poscount['v']:
                verbs = round(poscount['v'][j] / pos_sums['v'],2)
                verbs_2 = poscount['v'][j]
            else:
                verbs = 0
                verbs_2 = 0
            if 'a' in poscount and j in poscount['a']:
                adjectives = round(poscount['a'][j] / pos_sums['a'],2)
                adjectives_2 = poscount['a'][j]
            else:
                adjectives = 0
                adjectives_2 = 0
            if 'r' in poscount and j in poscount['r']:
                adverbs = round(poscount['r'][j] / pos_sums['r'],2)
                adverbs_2 = poscount['r'][j]
            else:
                adverbs = 0
                adverbs_2 = 0
            if j == 0:
                final_line_verb = f'Not in WN\tverb\t{verbs}\t{data_set}'
                final_line_noun = f'Not in WN\tnoun\t{nouns}\t{data_set}'
                final_line_adjective = f'Not in WN\tadjective\t{adjectives}\t{data_set}'

            else:
                tmp_verb = f'{j}\tverb\t{verbs_2}\t{data_set}'
                tmp_noun = f'{j}\tnoun\t{nouns_2}\t{data_set}'
                tmp_adjective = f'{j}\tadjectives\t{adjectives_2}\t{data_set}'

                df.append(dict(zip(columns, tmp_verb.split('\t'))))
                df.append(dict(zip(columns, tmp_noun.split('\t'))))
                df.append(dict(zip(columns, tmp_adjective.split('\t'))))
                #print(f'{j}\t{nouns}\t{verbs}\t{adjectives}\t{adverbs}')

        #df.append(dict(zip(columns, final_line_verb.split('\t'))))
        #df.append(dict(zip(columns, final_line_noun.split('\t'))))
        #df.append(dict(zip(columns, final_line_adjective.split('\t'))))
        df = pd.DataFrame(df)
        df = df[['count', 'pos', 'N.senses']]
        df['count'] = df['count'].astype(int)
        df['N.senses'] = df['N.senses'].astype(int)
        
        fig, ax = joypy.joyplot(pd.DataFrame(df), 
                        by = 'pos',
                        kind='values',
                        column='count', 
                        figsize = (5, 3),
                        fade = True,
                        x_range=[0,19])
        
        plt.title(f'{data_set} - {language}')
        plt.savefig(f'JOY_{data_set}_{language}.png')
