
from scripts.thirdparty.data_generator import main
from scripts.utils.bert_utils import convert_text_line_data_to_csv

if __name__ == '__main__':

    train_sdp_path = 'data/Semeval_8/train/sdp.txt'
    train_label_path = 'data/Semeval_8/train/labels.txt'
    out_train_path = 'data/bert_data/Semeval_8/train.tsv'

    dev_sdp_path = 'data/Semeval_8/val/sdp.txt'
    dev_label_path = 'data/Semeval_8/val/labels.txt'
    out_dev_path = 'data/bert_data/Semeval_8/dev.tsv'

    convert_text_line_data_to_csv(train_sdp_path,train_label_path,out_train_path)
    convert_text_line_data_to_csv(dev_sdp_path,dev_label_path,out_dev_path)
    # convert_text_line_data_to_csv(test_path,out_test_path)
