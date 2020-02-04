
from scripts.thirdparty.data_generator import main
from scripts.utils.bert_utils import convert_text_line_data_to_csv

if __name__ == '__main__':

    train_path = 'data/processed/sdp_data_acentors.train.txt'
    out_train_path = 'data/bert_data/train.tsv'
    dev_path = 'data/processed/sdp_data_acentors.dev.txt'
    out_dev_path = 'data/bert_data/dev.tsv'
    test_path = 'data/processed/sdp_data_acentors.test.txt'
    out_test_path = 'data/bert_data/test.tsv'
    convert_text_line_data_to_csv(train_path,out_train_path)
    convert_text_line_data_to_csv(dev_path,out_dev_path)
    convert_text_line_data_to_csv(test_path,out_test_path)
