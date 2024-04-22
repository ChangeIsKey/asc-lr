from src.attention_difference_extraction import AttentionAnalysis

if __name__ == '__main__':

    import os
    import numpy as np
    import argparse
    from tqdm import tqdm
    from pathlib import Path

    parser = argparse.ArgumentParser(prog='Computing cosine distances', add_help=True)
    parser.add_argument('-d', '--dir',
                        type=str,
                        help='Directory containing Lexical Replacement datasets')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='gpt2',  # 'bert-base-uncased',
                        help='Pre-trained bert-like model')
    parser.add_argument('-s', '--subword_prefix',
                        type=str,
                        default='##',
                        help='Subword_prefix')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=8,
                        help='batch_size')
    parser.add_argument('-M', '--max_length',
                        type=int,
                        default=512,
                        help='batch_size')
    parser.add_argument('-l', '--layer',
                        default=12,
                        type=int,
                        help='Layer from wich extract embeddings')
    parser.add_argument('-g', '--use_gpu',
                        action='store_true',
                        help='Use gpu if available')
    args = parser.parse_args()

    model_type = args.model.split('-')[0]  # i.e., bert or xlmr

    # create CD dir
    a = AttentionAnalysis(pretrained=args.model, subword_prefix=args.subword_prefix, use_gpu=args.use_gpu)
    a.add_token_to_vocab()
    paths = list(Path(args.dir).glob("*.txt"))
    bar = tqdm(paths, total=len(paths))
    for rep_file in bar:
        rep_file = str(rep_file)
        filename = os.path.basename(rep_file)[:-4]
        bar.set_description(filename)

        distances = a.extract_attn_distances(dataset=rep_file,
                                             batch_size=args.batch_size,
                                             max_length=args.max_length,
                                             layer=args.layer)

        for k in distances:
            Path(f'{model_type}/attention_differences/{k}/{args.layer}').mkdir(parents=True, exist_ok=True)
            output_filename = f'{model_type}/attention_differences/{k}/{args.layer}/{filename}.npy'
            np.save(output_filename, distances[k])

        bar.update(1)
