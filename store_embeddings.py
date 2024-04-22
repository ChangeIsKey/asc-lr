import os
import argparse
from tqdm import tqdm
from pathlib import Path
from src.embeddings_extraction import EmbeddingsExtractionTargetLayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Embeddings Extraction for Lexical Replacement pairs', add_help=True)
    parser.add_argument('-d', '--dir',
                        default='replacements',
                        type=str,
                        help='Directory containing Lexical Replacement datasets')
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
    parser.add_argument('-l', '--layer',
                        default=12,
                        type=int,
                        help='Layer from wich extract embeddings')
    parser.add_argument('-g', '--use_gpu',
                        action='store_true',
                        help='If true, use gpu for embeddings extraction')
    args = parser.parse_args()
    
    model_type = args.model.split('-')[0].replace('xlm', 'xlmr') # i.e., bert or xlmr
    
    # create directories
    Path(f'{model_type}/embeddings/{args.layer}').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/special_tokens_mask').mkdir(parents=True, exist_ok=True)
    Path(f'{model_type}/target_index').mkdir(parents=True, exist_ok=True)
    
    # create extractor
    extractor = EmbeddingsExtractionTargetLayer(pretrained=args.model, subword_prefix=args.subword_prefix, use_gpu=args.use_gpu)
    extractor.add_token_to_vocab() # add token [RANDOM]
    
    # get all replacement files
    paths = list(Path(args.dir).glob("*.txt"))
    
    # create progress bar
    bar = tqdm(paths, total=len(paths))
    for rep_file in bar:
        rep_file = str(rep_file) # filename
        filename = os.path.basename(rep_file)[:-4]
        bar.set_description(filename)
        
        # get data
        embeddings, target_indexes, special_tokens_mask = extractor.extract_embeddings(dataset=rep_file,
                                                                                       batch_size=args.batch_size,
                                                                                       max_length=args.max_length,
                                                                                       layer=args.layer)
        # output filenames
        embs_filename = f'{model_type}/embeddings/{args.layer}/{filename}.pt'
        mask_filename = f'{model_type}/special_tokens_mask/{filename}.pt'
        idx_filename = f'{model_type}/target_index/{filename}.npy'
        
        # store data
        torch.save(embeddings, embs_filename)
        np.save(idx_filename, target_indexes)
        torch.save(special_tokens_mask, mask_filename)
        
        # update bar
        bar.update(1)
