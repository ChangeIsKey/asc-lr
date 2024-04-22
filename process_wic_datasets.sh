wget https://github.com/SapienzaNLP/mcl-wic/raw/master/SemEval-2021_MCL-WiC_test-gold-data.zip
wget https://github.com/SapienzaNLP/mcl-wic/raw/master/SemEval-2021_MCL-WiC_all-datasets.zip
wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
wget https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip

unzip SemEval-2021_MCL-WiC_all-datasets.zip -d mclwic
unzip SemEval-2021_MCL-WiC_test-gold-data.zip -d mclwic_test
unzip WiC_dataset.zip -d wic_en
unzip xlwic_datasets.zip -d xlwic

# WiC-English processing
mv wic_en/*/* wic_en
python src/wic_en_processing.py
rm -rf wic_en/dev wic_en/train wic_en/test wic_en/README.txt wic_en/*.data* wic_en/*.gold*

# MCL-WiC processing
mv mclwic_test/test.en-en.gold mclwic
mv mclwic_test/test.fr-fr.gold  mclwic
rm -rf mclwic_test
mv mclwic/MCL-WiC/*/multilingual/*.fr-fr.* mclwic
mv mclwic/MCL-WiC/*/multilingual/*.en-en.* mclwic
mv mclwic/MCL-WiC/training/training.en-en.* mclwic
mv mclwic/training.en-en.data mclwic/train.en-en.data
mv mclwic/training.en-en.gold mclwic/train.en-en.gold
python src/mclwic_processing.py
rm -rf mclwic

# XL-WiC processing
mv xlwic/xlwic_datasets/xlwic_wikt/*/*.txt xlwic
rm -rf xlwic/xlwic_datasets
python src/xlwic_it_processing.py
rm -rf xlwic

# WiC-ITA processing
mkdir wicita
cd wicita
wget https://raw.githubusercontent.com/wic-ita/data/main/gold_truth/binary/test.jsonl
wget https://raw.githubusercontent.com/wic-ita/data/main/binary/dev.jsonl
wget https://raw.githubusercontent.com/wic-ita/data/main/binary/train.jsonl
cd ..
python src/wicita_processing.py
rm -rf wicita/*.jsonl

# remove zip
rm *.zip

# WiC
mkdir WiC
mv wic_en wicita mclwic_en mclwic_fr xlwic_it WiC

# DWUG-DE
wget https://zenodo.org/record/5796871/files/dwug_de.zip?download=1
unzip 'dwug_de.zip?download=1'
unzip 'dwug_de.zip?download=1'
mv dwug_de dwug_de_tmp
python src/dwug_de_processing.py
rm -rf dwug_de_tmp
