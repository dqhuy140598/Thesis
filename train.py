from scripts.dataset import CDRDataset
from scripts.config import args
from scripts.models.cnn import CNN
from torch.utils.data import DataLoader
from scripts.utils.helpers import load_pretrained_words_vector, countNumElements, get_mean, EarlyStopping
from sklearn.metrics import classification_report
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    train_path = 'data/processed/sdp_data_acentors.train.txt'
    dev_path = 'data/processed/sdp_data_acentors.test.txt'

    train_dataset = CDRDataset(file_path=train_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                              shuffle=True)

    dev_dataset = CDRDataset(file_path=dev_path)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                            shuffle=False)

    pretrained_words_vectors = load_pretrained_words_vector(args.EMBEDDINGS_PATH)

    n_poses = countNumElements(args.POSES_PATH)

    n_relations = countNumElements(args.RELATIONS_PATH)

    early_stopping = EarlyStopping(verbose=args.VERBOSE, patience=args.PATIENT)

    model = CNN(word_embeddings=pretrained_words_vectors,
                word_embedding_size=args.EMBEDIDNG_SIZE,
                pos_size=n_poses,
                pos_embedding_size=args.POS_EMBEDDING_SIZE,
                depend_size=n_relations,
                depend_embedding_size=args.RELATION_EMBEDDING_SIZE,
                position_size=args.MAX_LENGTH,
                position_embedding_size=args.POSITION_EMBEDDING_SIZE,
                n_filters=args.N_FILTERS,
                n_classes=2,
                drop_out=args.DROP_OUT,
                n_hidden=args.HIDDEN,
                filters_size=args.FILTERS_SIZE
                )

    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.LR)

    schedule = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.2, last_epoch=-1)

    for epoch in range(1, args.EPOCH):

        loss_epoch = []

        val_loss_epoch = []

        model.train()

        schedule.step()

        for batch in tqdm(train_loader, desc='Epoch:{}'.format(epoch)):
            optimizer.zero_grad()
            data = batch[0]
            label = batch[1]
            position_1 = batch[2]
            position_2 = batch[3]
            pos = batch[4]
            relations = batch[5]

            data = data.to(device)
            label = label.to(device)
            position_1 = position_1.to(device)
            position_2 = position_2.to(device)
            pos = pos.to(device)
            relations = relations.to(device)

            label = torch.LongTensor(label)

            logits = model([data, pos, relations, position_1, position_2])

            loss_fn = model.loss(input=logits, target=label)

            loss_epoch.append(loss_fn.item())

            nn.utils.clip_grad_norm_(model.parameters(), args.CLIP_GRAD)

            loss_fn.backward()

            optimizer.step()

        model.eval()
        train_out_put = []
        train_labels = []
        for batch in tqdm(dev_loader, desc='Eval'):
            data = batch[0]
            label = batch[1]
            position_1 = batch[2]
            position_2 = batch[3]
            pos = batch[4]
            relations = batch[5]

            data = data.to(device)
            label = label.to(device)
            position_1 = position_1.to(device)
            position_2 = position_2.to(device)
            pos = pos.to(device)
            relations = relations.to(device)

            label = torch.LongTensor(label)

            logits = model([data, pos, relations, position_1, position_2])

            loss_fn = model.loss(input=logits, target=label)

            val_loss_epoch.append(loss_fn.item())

            train_out_put.extend(torch.argmax(torch.sigmoid(logits), dim=-1).detach().numpy().tolist())

            label_idx = label.detach()

            train_labels.extend(label_idx.numpy().tolist())

        print('Epoch:{0}, train_loss:{1}, val_loss:{2}'.format(epoch, get_mean(loss_epoch), get_mean(val_loss_epoch)))

        print('Result In Validation Set:')

        print(classification_report(train_labels, train_out_put, target_names=['CID', 'NONE']))

        if args.EARLY_STOPPING:
            early_stopping(get_mean(val_loss_epoch), model)

            if early_stopping.early_stop:
                print('Execute Early Stopping.....')

                break


if __name__ == '__main__':
    train()
