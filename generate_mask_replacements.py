from transformers import AutoTokenizer, pipeline
import torch
import os
import json
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self,sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,i):
        return self.sentences[i]


monolingual_models = {'Swedish':'KBLab/bert-base-swedish-cased', 'German':'bert-base-german-cased'}#'English':'roberta-base', 
multilingual_models = ['dbmdz/bert-base-historic-multilingual-cased', 'xlm-roberta-base']


for language in monolingual_models:
    model = 'xlm-roberta-base'#'dbmdz/bert-base-historic-multilingual-cased'
    #model = monolingual_models[language]
    tokenizer = AutoTokenizer.from_pretrained(model, max_length=512, truncation=True)
    tokenizer_kwargs = {'truncation': True, 'max_length': 512}
    mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

    model_path = 'mask_predictions/'+model.replace('/','_')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    os.mkdir(f'{model_path}/{language}')
    
    for fname in os.listdir(f'/mimer/NOBACKUP/groups/cik_data/cassotti/all_sentences/{language.lower()}'):
        word = fname.split('.')[0]
        sentences = {}
        with open(f'/mimer/NOBACKUP/groups/cik_data/cassotti/all_sentences/{language.lower()}/{fname}') as f:
            sentences = json.load(f)

        for time in ["1","2"]:
            batch_sentences = []
            for sentence in sentences[time]:
                batch_sentences.append(sentence['text'][:sentence['start']] + tokenizer.mask_token + sentence['text'][sentence['end']:])

            dataset = SentenceDataset(batch_sentences)

            with open(f'{model_path}/{language}/{word}_{time}.jsonl','w+') as f:
                for result in mask_filler(dataset, batch_size=32, top_k=20, **tokenizer_kwargs):
                    result = [{'token':r['token_str'],'score':r['score']} for r in result]
                    f.write(f'{json.dumps(result)}\n')
