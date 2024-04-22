# Analyzing Semantic Change through Lexical Replacements
This repository contains the code accompanying the paper titled "Analyzing Semantic Change through Lexical Replacements"

## Paper Abstract
In this paper, we  analyse the _tug of war_ between the contextualisation and the pre-trained knowledge of BERT models. If BERT models excessively rely on pre-trained knowledge to represent words, they may falter when faced with words or meanings that lie beyond their training data, e.g., words outside their pre-trained vocabulary or words that have experienced semantic change. We conduct analysis via a replacement schema, which generates replacement sets with graded lexical relatedness, allowing examination of the models' degree of contextualisation.  We find that a large part of the representation of a word stems from information stored in the model itself and that the degree of contextualisation varies across parts of speech. Furthermore, we leverage the replacement schema as a basis for a novel interpretable approach to Lexical Semantic Change, surpassing the state-of-the-art for English.

<b> Citation </b>

```
@inproceedings{analyzing_semantic_change_through_lexical_replacements,
  author    = {Francesco Periti and
               Pierluigi Cassotti and
               Haim Dubossarsky and
               Nina Tahmasebi},
  title     = {Analyzing Semantic Change through Lexical Replacements},
  year      = {2024},
}
```

## Contents
This repository includes the following files:

`src`: Python code.


## SemCor dataset

## Lexical substitutes generation
### Masked Language Modeling

### LLama 7B Generation
[https://huggingface.co/ChangeIsKey/llama-7b-lexical-substitution/](https://huggingface.co/ChangeIsKey/llama-7b-lexical-substitution/)


## Reproducing the Analysis: Execution Instructions

Convert and create unknown replacements
```bash
python src/unknwon_replacements.py
```

Extract embeddings for every word in each sentence from each replacement file across all layers of the model
```bash
layers=12 # layers of the model
for layer in $(seq 1 $layers);
do
    python store_embeddings.py --dir replacements --batch_size 16 --layer "${layer}" --model bert-base-uncased --use_gpu -s "##"
    python store_embeddings.py --dir replacements --batch_size 16 --layer "${layer}" --model xlm-roberta-base --use_gpu -s "_"
done
```

Download and process WiC datasets
```bash
bash process_wic_datasets.sh
```

Extract embeddings for every target word of each WiC dataset 
```bash
python store_target_embeddings.py -d WiC/mclwic_en --model bert-base-uncased --batch_size 16 --train_set --test_set --dev_set --use_gpu
python store_target_embeddings.py -d WiC/wic_en --model bert-base-uncased --batch_size 16 --train_set --test_set --dev_set --use_gpu
python store_target_embeddings.py -d WiC/mclwic_fr --model dbmdz/bert-base-french-europeana-cased --batch_size 16 --test_set --dev_set --use_gpu # no train set available
python store_target_embeddings.py -d WiC/xlwic_it --model dbmdz/bert-base-italian-uncased --batch_size 16 --train_set --test_set --dev_set --use_gpu
python store_target_embeddings.py -d WiC/wicita --model dbmdz/bert-base-italian-uncased --batch_size 16 --train_set --test_set --dev_set --use_gpu
python store_target_embeddings.py -d WiC/dwug_de --model bert-base-german-cased --batch_size 16 --train_set --test_set --dev_set --use_gpu
```

Compute distances between embeddings in each layer
```bash
layers=12 # layers of the model
for layer in $(seq 1 $layers);
do
    python embedding_distances.py -e "bert/embeddings" -l "${layer}" -t "bert/target_index" -s "bert/special_token_mask" --model_type bert
    python embedding_distances.py -e "xlmr/embeddings" -l "${layer}" -t "xlmr/target_index" -s "xlmr/special_token_mask" --model_type xlmr
done
```

Compute distances between word and context embeddings in each layer. Context embeddings is computed as average of the other token embeddings in the sentence.
```bash
layers=12 # layers of the model
for layer in $(seq 1 $layers);
do
    python word_context_embedding_distances.py -e "bert/embeddings" -l "${layer}" -t "bert/target_index" -s "bert/special_tokens_mask" --model_type bert
    python word_context_embedding_distances.py -e "xlmr/embeddings" -l "${layer}" -t "xlmr/target_index" -s "xlmr/special_tokens_mask" --model_type xlmr
done
```

Compute attention differences between attention scores in each layer
```bash
layers=12 # layers of the model
for layer in $(seq 1 $layers);
do
    python attention_differences.py -d replacements -b 16 -l "${layer}" -m bert-base-uncased -s "##" --use_gpu
    python attention_differences.py -d replacements -b 16 -l "${layer}" -m xlm-roberta-base -s "_" --use_gpu
done
```

Plot Self-embedding distances
```bash
python sed_plots.py
```

Compute stats for Word-in-Context
```bash
python wic_stats.py -d WiC/mclwic_en -m bert-base-uncased --test_set --train_set --dev_set
python wic_stats.py -d WiC/wic_en -m bert-base-uncased --test_set --train_set --dev_set
python wic_stats.py -d WiC/mclwic_fr -m dbmdz_bert-base-french-europeana-cased --test_set --dev_set
python wic_stats.py -d WiC/xlwic_it -m dbmdz_bert-base-italian-uncased --test_set --train_set --dev_set
python wic_stats.py -d WiC/wicita -m dbmdz_bert-base-italian-uncased --test_set --train_set --dev_set
python wic_stats.py -d WiC/dwug_de -m bert-base-german-cased --test_set --train_set --dev_set
```

Ridgeline plot for polysemy
```bash
python joyplot_wic_stats.py
```

Download data for LSC
```bash
wget https://zenodo.org/record/7441645/files/dwug_de.zip?download=1
unzip dwug_de.zip\?download\=1
mv dwug_de DWUG-German

wget https://zenodo.org/record/6433667/files/dwug_es.zip?download=1
unzip dwug_es.zip?download=1
mv dwug_es DWUG-Spanish

wget https://zenodo.org/record/7389506/files/dwug_sv.zip?download=1
unzip dwug_sv.zip?download=1
mv dwug_sv DWUG-Swedish

wget https://zenodo.org/record/7387261/files/dwug_en.zip?download=1
unzip dwug_en.zip\?download\=1
mv dwug_en DWUG-English
```

Process data for LSC
```bash
python processDWUGdatasets.py  -b DWUG-German -c datasets/LSC/ -t tokenization/LSC
python processDWUGdatasets.py  -b DWUG-Swedish -c datasets/LSC/ -t tokenization/LSC
python processDWUGdatasets.py  -b DWUG-English -c datasets/LSC/ -t tokenization/LSC
python processDWUGdatasets.py  -b DWUG-Spanish -c datasets/LSC/ -t tokenization/LSC
```

Generated artificial LSC datasets
```bash
Replacements.ipynb
```

Store embeddings for LSC
```bash
python embs-lsc.py -d DWUG-English-repl --model bert-base-multilingual-cased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-English-repl --model bert-base-uncased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-English-repl --model xlm-roberta-base --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-German-repl --model bert-base-multilingual-cased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-German-repl --model bert-base-german-cased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-German-repl --model xlm-roberta-base --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-Swedish-repl --model bert-base-multilingual-cased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-Swedish-repl --model KBLab/bert-base-swedish-cased-new --batch_size 8 --use_gpu
python embs-lsc.py -d DWUG-Swedish-repl --model xlm-roberta-base --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-Spanish-repl --model bert-base-multilingual-cased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-Spanish-repl --model dccuchile/bert-base-spanish-wwm-uncased --batch_size 16 --use_gpu
python embs-lsc.py -d DWUG-Spanish-repl --model xlm-roberta-base --batch_size 16 --use_gpu
```

Test PRT and JSD for LSC datasets
```bash
Replacements.ipynb
```

New approach for LSC
```bash
python3 lsc_compute.py
```
