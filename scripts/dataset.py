from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from scripts.utils.data_utils import process_dataset, pad_to_same
from scripts.utils.helpers import read_vocab
from scripts.config.args import VOCAB_PATH, POSES_PATH, RELATIONS_PATH, BATCH_SIZE, NUM_WORKERS, MAX_LENGTH_TO_FIT


class CDRDataset(Dataset):

    def __init__(self, file_path):
        super(CDRDataset, self).__init__()
        self.file_path = file_path
        self.vocab_path = VOCAB_PATH
        self.poses_path = POSES_PATH
        self.relations_path = RELATIONS_PATH

        if not os.path.exists(self.file_path):
            raise FileNotFoundError('Your Dataset Path Is Not Exists')

        with open(self.file_path, 'r') as f:
            self.lines = f.readlines()

        self.vocab = read_vocab(self.vocab_path)
        self.poses_vocab = read_vocab(self.poses_path)
        self.relations_vocab = read_vocab(self.relations_path)

        self.words, self.positions_1, self.positions_2, self.labels, self.poses, self.relations, self.directions = \
            process_dataset(self.lines, self.vocab, self.poses_vocab, self.relations_vocab)

        # print(self.vocab)
        assert len(self.words) == len(self.labels)
        assert len(self.words) == len(self.positions_1)
        assert len(self.words) == len(self.positions_2)
        assert len(self.words) == len(self.poses)
        assert len(self.words) == len(self.relations)
        assert len(self.words) == len(self.directions)

        self.max_length = MAX_LENGTH_TO_FIT

        # print(self.words[0])
        # print(self.labels[0])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        data = pad_to_same(self.words[idx], 0, self.max_length)
        label = self.labels[idx]
        position_1 = pad_to_same(self.positions_1[idx], 0, self.max_length)
        position_2 = pad_to_same(self.positions_2[idx], 0, self.max_length)
        pos = pad_to_same(self.poses[idx], 0, self.max_length)
        relation = pad_to_same(self.relations[idx], 0, self.max_length)
        direction = pad_to_same(self.directions[idx], 0, self.max_length)

        return np.array(data), np.array(label), np.array(position_1), np.array(position_2), np.array(pos), np.array(
            relation), np.array(direction)


if __name__ == '__main__':
    train_path = 'data/processed/sdp_data_acentors.test.txt'
    train_dataset = CDRDataset(file_path=train_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    for batch in train_loader:
        for tensor in batch:
            print(tensor.size())
        break
