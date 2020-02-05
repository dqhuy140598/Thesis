import pandas as pd
from scripts.utils.data_utils import parse_raw_dataset

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}


def convert_text_line_data_to_csv(train_sdp_path, train_label_path, out_path):
    # words, _, labels, _, _, _, _ = \
    #     parse_raw_dataset(lines)

    with open(train_sdp_path, 'r') as f:
        sdps = f.readlines()

    with open(train_label_path, 'r') as f:
        labels = f.readlines()

    # print(sdps)
    # print(labels)

    idx_labels = [class2label[x.strip()] for x in labels]

    print(idx_labels)

    line_words = [x.strip() for x in sdps]

    df = pd.DataFrame({
        'id': range(len(idx_labels)),
        'label': idx_labels,
        'alpha': ['a'] * len(line_words),
        'text': line_words,
    })

    df.to_csv(out_path, sep='\t', index=False, header=False)
