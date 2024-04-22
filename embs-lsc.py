import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from src.embeddings_extraction import TargetEmbeddingsExtraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Embeddings Extraction for WiC pairs', add_help=True)
    parser.add_argument('-d', '--dataset',
                    default='DWUG-English-repl')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='bert-base-uncased',
                        help='Pre-trained bert-like model')
    parser.add_argument('-s', '--subword_prefix',
                        type=str,
                        default='##',
                        help='Subword_prefix')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=16,
                        help='batch_size')
    parser.add_argument('-M', '--max_length',
                        type=int,
                        default=None,
                        help='Max length used for tokenization')
    parser.add_argument('-g', '--use_gpu',
                        action='store_true',
                        help='If true, use gpu for embeddings extraction')
    args = parser.parse_args()

    # create extractor
    extractor = TargetEmbeddingsExtraction(args.model, subword_prefix=args.subword_prefix, use_gpu=args.use_gpu)

    corpora = ['corpus1', 'corpus2']    
    bar = tqdm(corpora, total=len(corpora))
    for corpus in bar:
        bar.set_description(corpus)
        
        input_filename = f'tokenization/LSC/{args.dataset}/{corpus}/token'

        for p in Path(input_filename).glob('*.txt'):
            target = str(p).split('/')[-1].replace('.txt', '')
            print(p)
            # extraction
            embeddings = extractor.extract_embeddings(dataset=str(p),
                                                  batch_size=args.batch_size,
                                                  max_length=args.max_length)

            # layers
            layers = embeddings.keys()

            # store embeddings
            for layer in layers:
                # create directories
                Path(f'lsc_embeddings/{args.dataset}/{args.model.replace("/", "_")}/{corpus}/{layer}').mkdir(parents=True, exist_ok=True)
                output_filename = f'lsc_embeddings/{args.dataset}/{args.model.replace("/", "_")}/{corpus}/{layer}/{target}.pt'
                torch.save(embeddings[layer].to('cpu'), output_filename)
        
        # update bar
        bar.update(1)
