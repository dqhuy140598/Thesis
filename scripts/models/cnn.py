import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self, word_embeddings,
                 word_embedding_size,
                 pos_size,
                 pos_embedding_size,
                 depend_size,
                 depend_embedding_size,
                 position_size,
                 position_embedding_size,
                 n_filters,
                 filters_size,
                 n_classes,
                 drop_out=0.5,
                 n_hidden=250):
        """
        constructor of CNN model class
        @param word_embeddings: embedding matrix of words vocabulary
        @param pos_size: part of speech tagging size
        @param depend_size: depend size
        @param params: hyper parameters of the cnn model
        """
        super(CNN, self).__init__()

        self.word_embedding = torch.nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(word_embeddings),
                                                                 freeze=False)
        self.p1_embedding = torch.nn.Embedding(num_embeddings=position_size,embedding_dim=position_embedding_size)
        self.p2_embedding = torch.nn.Embedding(num_embeddings=position_size,embedding_dim=position_embedding_size)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=pos_size, embedding_dim=pos_embedding_size)
        self.depend_embedding = torch.nn.Embedding(num_embeddings=depend_size,embedding_dim=depend_embedding_size)

        drop_out_rate = drop_out

        self.drop_out = torch.nn.Dropout(p=drop_out_rate)

        self.convs = []

        feature_dims = word_embedding_size + pos_embedding_size + 2* position_embedding_size + depend_embedding_size

        for filter_size in filters_size:
            conv = torch.nn.Conv2d(in_channels=1, out_channels=n_filters,
                                   kernel_size=(filter_size, feature_dims))

            self.convs.append(conv)

        flat_size = n_filters * len(filters_size)

        n_classes = n_classes

        self.linear0 = torch.nn.Linear(flat_size, n_hidden)

        self.linear = torch.nn.Linear(n_hidden, n_classes)

        # Binary cross entropy loss for binary classification problem
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        feed forward the input
        @param x: input data
        @return: output data
        """

        # use only shortest dependency path and part of speech tagging
        sdp_idx = x[0]
        pos_sdp_idx = x[1]
        depend_idx = x[2]
        position1 = x[3]
        position2 = x[4]

        # print(position1)

        sdp_embeddings = self.word_embedding(sdp_idx)
        pos_sdp_embeddings = self.pos_embedding(pos_sdp_idx)
        depend_embeddings = self.depend_embedding(depend_idx)
        pos1_embeddings = self.p1_embedding(position1)
        pos2_embeddings = self.p2_embedding(position2)

        # print(sdp_embeddings.size())

        feature = torch.cat([sdp_embeddings, pos_sdp_embeddings, depend_embeddings, pos1_embeddings ,pos2_embeddings], dim=2)

        # feature = sdp_embeddings

        feature = torch.unsqueeze(feature, dim=1)  # batch_size x 1 x max_length x (embedding_size + pos_size)

        out_feature = []

        for conv in self.convs:
            out = conv(feature)  # batch_size x n_filters x (max_length - filter_size + 1) x 1

            out = torch.squeeze(out, dim=-1)  # batch_size x n_filters x (max_length - filter_size + 1)

            out_feature.append(out)

        out_feature = [torch.relu(x) for x in out_feature]

        out_pool = [F.max_pool1d(x, kernel_size=(x.size(2))).squeeze(dim=2) for x in
                    out_feature]  # [batch_size x n_filters]

        sentence_features = torch.cat(out_pool, dim=1)

        output = self.drop_out(sentence_features)

        output = self.linear0(output)

        output = self.linear(output)

        return output

