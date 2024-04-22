import json
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
from sklearn.metrics import dcg_score
from scipy.stats import kendalltau
import pickle

language = 'English'
language_short = language[:3].lower()

D = {}
with open(f'/mimer/NOBACKUP/groups/cik_data/datasets/LSC/SemEval-{language}/semeval2020_ulscd_{language_short}/truth/graded.txt') as f:
    for line in f:
        word, label = line.split()
        label = float(label)
        D[word] = label

D2 = {}
with open(f'/mimer/NOBACKUP/groups/cik_data/datasets/LSC/SemEval-{language}/semeval2020_ulscd_{language_short}/truth/binary.txt') as f:
    for line in f:
        word, label = line.split()
        label = int(label)
        D2[word] = label

language = language.lower()

word2sub = {}
#mGPT
#Llama-2-13b-hf
#Llama-2-7b-hf
#bloom-7b1
for model in ['Llama-2-7b-hf']:#os.listdir(f'semeval_predictions'):
    languages = os.listdir(f'semeval_predictions/{model}')
    if language in languages:
        for fname in os.listdir(f'semeval_predictions/{model}/{language}'):
            tempname = fname.split('.')[0]
            word = '_'.join(tempname.split('_')[:-1])
            time = tempname.split('_')[-1]
            if not word in word2sub:
                word2sub[word] = {}
            if not time in word2sub[word]:
                word2sub[word][time] = {}
            if not model in word2sub[word][time]:
                word2sub[word][time][model] = []
            with open(f'semeval_predictions/{model}/{language}/{fname}') as f:
                for line in f:
                    word2sub[word][time][model].append(json.loads(line))


def comp_function(**kwargs):
    model_rep = {}

    words_set = {}

    for word in word2sub:
        word_orig = word.split()[0]
        for time in sorted(word2sub[word]):
            for model in word2sub[word][time]:
                count = {}
                if not model in model_rep:
                    model_rep[model] = {}
                for sentence in word2sub[word][time][model]:
                    if not word in words_set:
                        words_set[word] = {}
                    if not time in words_set[word]:
                        words_set[word][time] = []
                    tokens_set = []
                    predictions = sentence['output'].replace('<pad>','').replace('[PAD]','').split('<|answer|>')[1]
                    if '<|end|>' in predictions:
                        predictions = predictions.split('<|end|>')[0].split('<|s|>')
                    else:
                        predictions = predictions.split('<|s|>')[:5]
                    for pred in predictions:
                        token = pred.strip()
                        if len(token) >= 3 and token.replace(' ','').isalpha() and not word_orig.split('_')[0] in token:
                            tokens_set.append(token)
                    words_set[word][time].append(tokens_set)

    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def rbo(l1, l2, p=0.98):
        """
            Calculates Ranked Biased Overlap (RBO) score.
            l1 -- Ranked List 1
            l2 -- Ranked List 2
        """
        if l1 == None: l1 = []
        if l2 == None: l2 = []

        sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
        s, S = sl
        l, L = ll
        if s == 0: return 0

        # Calculate the overlaps at ranks 1 through l
        # (the longer of the two lists)
        ss = set([])  # contains elements from the smaller list till depth i
        ls = set([])  # contains elements from the longer list till depth i
        x_d = {0: 0}
        sum1 = 0.0
        for i in range(l):
            x = L[i]
            y = S[i] if i < s else None
            d = i + 1

            # if two elements are same then
            # we don't need to add to either of the set
            if x == y:
                x_d[d] = x_d[d - 1] + 1.0
            # else add items to respective list
            # and calculate overlap
            else:
                ls.add(x)
                if y != None: ss.add(y)
                x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
                # calculate average overlap
            sum1 += x_d[d] / d * pow(p, d)

        sum2 = 0.0
        for i in range(l - s):
            d = s + i + 1
            sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

        sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

        # Equation 32
        rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
        return rbo_ext

    """
    truth = []
    predict = []
    for word in words_set:
        set1 = set([s for s1 in words_set[word]['1'] for s in s1])
        set2 = set([s for s2 in words_set[word]['2'] for s in s2])
        print(word)
        print(jaccard(set1,set2))
        predict.append(jaccard(set1,set2))
        truth.append(D[word])

    print(spearmanr(truth,predict))
    """

    truth = []
    predict = []
    for word in words_set:
        scores = []
        for s1 in words_set[word]['1']:
            for s2 in words_set[word]['2']:
                if len(s1) >= 5 and len(s2) >= 5:
                #if len(s1)>3 and len(s2) > 4:
                #    m = min(len(s1),len(s2))
                #    print(s1)
                #    print(s2)
                    #if not np.isnan(spearmanr(s1[:m],s2[:m]).statistic):
                    #    scores.append(spearmanr(s1[:m],s2[:m]).statistic)
                    scores.append(jaccard(s1,s2))
        print(word)
        print(1-(sum(scores)/len(scores)))
        predict.append(1-(sum(scores)/len(scores)))
        truth.append(D[word])

    print(spearmanr(truth,predict))

    """
    truth = []
    predict = []
    binary_predict = []
    binary_truth = []

    for model in model_rep:
        for word in sorted(model_rep[model]):
            word_orig = word.split()[0]
            p0 = [0.]* len(model_rep[model][word])
            p1 = [0.]* len(model_rep[model][word])
            for j,rep in enumerate(model_rep[model][word]):
                if rep in time_counts[model][word][0]:
                    p0[j] = time_counts[model][word][0][rep]
                else:
                    p0[j] = 0.
                if rep in time_counts[model][word][1]:
                    p1[j] = time_counts[model][word][1][rep]
                else:
                    p1[j] = 0.
            p0 = np.array(p0)/np.sum(p0)
            p1 = np.array(p1)/np.sum(p1)
            predict.append(jensenshannon(p0,p1))
            binary_predict.append(0)
            for j in range(len(p0)):
                if p0[j] < 0.001 and p1[j] > 0.05:
                    binary_predict[-1] = 1
                elif p1[j] < 0.001 and p0[j] > 0.05:
                    binary_predict[-1] = 1
            binary_truth.append(D2[word])
            truth.append(D[word])

        spvalue = spearmanr(truth,predict).statistic
        f1 = f1_score(binary_truth,binary_predict,average='weighted')
    """

    """
    print(spvalue)
    print(f1)
    for j,word in enumerate(sorted(model_rep[model])):
        print(word,predict[j])
    """

    #return (1/2*f1) + (1/2*spvalue)




comp_function()
