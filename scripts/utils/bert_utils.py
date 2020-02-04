import pandas as pd
from scripts.utils.data_utils import parse_raw_dataset
from scripts.utils.helpers import read_vocab
from scripts.config import args


def convert_text_line_data_to_csv(file_path,out_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    vocab = read_vocab(args.VOCAB_PATH)
    poses_vocab = read_vocab(args.POSES_PATH)
    relations_vocab = read_vocab(args.RELATIONS_PATH)

    words, _, labels, _, _, _, _ = \
        parse_raw_dataset(lines)

    idx_labels = [int(x[0] == 'CID') for x in labels]

    line_words = [' '.join(x) for x in words]

    df = pd.DataFrame({
        'id': range(len(words)),
        'label': idx_labels,
        'alpha': ['a'] * len(line_words),
        'text': line_words,
    })

    df.to_csv(out_path,sep='\t',index=False,header=False)


