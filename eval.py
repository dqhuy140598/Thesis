import torch
from scripts.dataset import CDRDataset
from scripts.config import args
from scripts.utils.helpers import get_mean
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval(data_path, model_path):
    test_dataset = CDRDataset(file_path=data_path)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.BATCH_SIZE, shuffle=False,
                             num_workers=args.NUM_WORKERS)

    model = torch.load(model_path)

    model.eval()

    loss = []

    eval_out_put = []

    target = []

    for batch in tqdm(test_loader):
        data = batch[0]

        label = batch[1]
        position_1 = batch[2]
        position_2 = batch[3]
        pos = batch[4]
        relations = batch[5]

        label = torch.LongTensor(label)

        logits = model([data, pos, relations, position_1, position_2])

        loss_fn = model.loss(input=logits, target=label)

        loss.append(loss_fn.item())

        eval_out_put.extend(torch.argmax(torch.sigmoid(logits), dim=-1).detach().numpy().tolist())

        label_idx = label.detach()

        target.extend(label_idx.numpy().tolist())

    print('Test loss: {0:.2f}'.format(get_mean(loss)))

    print('Classification Report:')

    print(classification_report(target, eval_out_put, target_names=['CID', 'NONE']))


if __name__ == '__main__':
    print('Evaluate .......')

    eval('data/original_data/sdp_data_acentors.test.txt', 'trained_models/model.pt')
