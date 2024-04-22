import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from src.embeddings_extraction import TargetEmbeddingsExtraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Embeddings Extraction for WiC pairs', add_help=True)
    parser.add_argument('-d', '--dir',
                    type=str,
                    help='Directory containing WiC datasets processed')
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
    parser.add_argument('-T', '--train_set',
                        action='store_true',
                        help='If true, extract embddings for train set')
    parser.add_argument('-t', '--test_set',
                        action='store_true',
                        help='If true, extract embddings for test set')
    parser.add_argument('-D', '--dev_set',
                        action='store_true',
                        help='If true, extract embddings for dev set')
    args = parser.parse_args()

    # create extractor
    extractor = TargetEmbeddingsExtraction(args.model, subword_prefix=args.subword_prefix, use_gpu=args.use_gpu)
    extractor.add_token_to_vocab() # add token [RANDOM]

    sets = list()
    if args.dev_set:
        sets.append('dev')
    if args.train_set:
        sets.append('train')
    if args.test_set:
        sets.append('test')
    
    bar = tqdm(sets, total=len(sets))
    for s in bar:
        bar.set_description(s)
        
        # create directories
        Path(f'{args.dir}/target_embeddings/{args.model.replace("/", "_")}/{s}').mkdir(parents=True, exist_ok=True)

        input_filename = f'{args.dir}/{s}.txt'

        # extraction
        embeddings = extractor.extract_embeddings(dataset=input_filename,
                                                  batch_size=args.batch_size,
                                                  max_length=args.max_length)

        # layers
        layers = embeddings.keys()

        # store embeddings
        for layer in layers:
            output_filename = f'{args.dir}/target_embeddings/{args.model.replace("/", "_")}/{s}/{layer}.pt'
            torch.save(embeddings[layer].to('cpu'), output_filename)
        
        # update bar
        bar.update(1)
