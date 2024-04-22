from src.wic import WiC

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='WiC evaluation', add_help=True)
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Dirname to a tokenize dataset for LSC detection')
    parser.add_argument('-m', '--model',
                        type=str,
                        help='Hugginface pre-trained model')
    parser.add_argument('-t', '--test_set',
                        action='store_true',
                        help='If test set is available')
    parser.add_argument('-T', '--train_set',
                        action='store_true',
                        help='If train set is available')
    parser.add_argument('-D', '--dev_set',
                        action='store_true',
                        help='If dev set is available')
    args = parser.parse_args()

    data_sets = list()
    if args.test_set:
        data_sets.append('test')
    if args.train_set:
        data_sets.append('train')
    if args.dev_set:
        data_sets.append('dev')

    w = WiC(args.dataset, data_sets)
    w.fit(args.model).to_csv(f'{args.dataset}/wic_stats.tsv', sep='\t', index=False)
