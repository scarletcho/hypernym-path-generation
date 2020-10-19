import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_normal_


class PathEncoder(nn.Module):
    """
    The LSTM layer encoding the path from root to direct hypernyms for all models containing an LSTM layer
    """
    def __init__(self, batch_size, dim_embedding, dim_hidden, drop_rate, max_len, cuda=False):
        super(PathEncoder, self).__init__()
        self.lstm = nn.LSTM(dim_embedding, dim_hidden // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)

        self.dim_hidden = dim_hidden
        self.max_len = max_len
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.hidden = (torch.zeros(2, batch_size, dim_hidden // 2).to(self.device),
                       torch.zeros(2, batch_size, dim_hidden // 2).to(self.device))

    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(2, batch_size, self.dim_hidden // 2).to(self.device),
                       torch.zeros(2, batch_size, self.dim_hidden // 2).to(self.device))

    def forward(self, path, lengths):
        """

        :param path: long tensor in the shape (batch_size, path_len)
        :param lengths: long tensor in the shape (batch_size)
        :return: float tensor of the concatenated last hidden state in the shape (batch_size, dim_hidden)
        """
        path = pack_padded_sequence(path, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(path, self.hidden)
        return self.dropout(torch.cat((self.hidden[0][0,:,:], self.hidden[0][1,:,:]), dim=1))


class EmbeddingLayer(nn.Module):
    """
    The look-up table for synset embeddings
    """
    def __init__(self, pretrain_embedding, num_embedding, dim_embedding, padding_idx):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embedding, dim_embedding)
        self.embedding.weight.data.copy_(pretrain_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, sent):
        return self.embedding(sent)


class BinaryClassifier(nn.Module):
    def __init__(self, batch_size, pretrain_embedding, num_embedding, dim_embedding, dim_out, dim_hidden, padding_idx=0,
                 drop_rate=0, max_len=20, cuda=False):
        super(BinaryClassifier, self).__init__()
        self.embedding = EmbeddingLayer(pretrain_embedding, num_embedding, dim_embedding, padding_idx)
        self.path_encoder = PathEncoder(batch_size, dim_embedding, dim_hidden, drop_rate, max_len, cuda=cuda)
        self.project = nn.Linear(dim_hidden + dim_embedding, 1024)
        self.top = nn.Linear(1024, 2)
        # self.cuda = cuda

        for layer in [self.project, self.top]:
            for p in layer.parameters():
                if p.requires_grad and p.dim() > 2:
                    xavier_normal_(p)

    def forward(self, path, word, lengths, batch_size):
        path_encoded = self.path_encoder(self.embedding(path), lengths)
        word_emebed = self.embedding(word)
        _ = ''
        # print(path_encoded.size(), word_emebed.size())
        return self.top(F.relu(self.project(torch.cat((path_encoded, word_emebed), 1)))), _


class EmbeddingReg(nn.Module):
    """
    The actual PathEncoder model
    """
    def __init__(self, batch_size, pretrain_embedding, num_embedding, dim_embedding, dim_out, dim_hidden, padding_idx=0,
                 drop_rate=0, max_len=20, cuda=False):
        super(EmbeddingReg, self).__init__()
        self.embedding = EmbeddingLayer(pretrain_embedding, num_embedding, dim_embedding, padding_idx)
        self.path_encoder = PathEncoder(batch_size, dim_embedding, dim_hidden, drop_rate, max_len, cuda=cuda)
        self.project = nn.Linear(dim_embedding, dim_hidden)
        self.top_word = nn.Linear(dim_hidden, dim_out)
        self.top_path = nn.Linear(dim_hidden, dim_out)
        # self.cuda = cuda

        for layer in [self.top_path]:
            for p in layer.parameters():
                if p.requires_grad and p.dim() > 2:
                    xavier_normal_(p)

    def forward(self, path, word, lengths, batch_size):
        label = self.path_encoder(self.embedding(path), lengths)
        label = self.top_path(label)
        # print(path_encoded.size(), word_emebed.size())
        word_emebed = self.embedding(word)
        # word_emebed = self.top_word(F.relu(self.project(word_emebed)))
        return word_emebed, label


class NearestNeighbor(nn.Module):
    """
    The baseline model performs nearest neighbor search in the embedding space
    """
    def __init__(self, batch_size, pretrain_embedding, num_embedding, dim_embedding, dim_out, dim_hidden, padding_idx=0,
                 drop_rate=0, max_len=20, cuda=False):
        super(NearestNeighbor, self).__init__()
        self.embedding = EmbeddingLayer(pretrain_embedding, num_embedding, dim_embedding, padding_idx)
        self.path_encoder = PathEncoder(batch_size, dim_embedding, dim_hidden, drop_rate, max_len, cuda=cuda)
        # self.project = nn.Linear(dim_embedding, dim_hidden)
        # self.top_word = nn.Linear(dim_hidden, dim_out)
        self.top_path = nn.Linear(dim_hidden, dim_out)
        # self.cuda = cuda

    def forward(self, path, word, lengths, batch_size):

        path = path[list(range(batch_size)), list(lengths.squeeze().cpu().to(torch.long) - torch.ones(batch_size).to(torch.long))]
        label = self.embedding(path)
        word_emebed = self.embedding(word)
        return word_emebed, label


class NNEmbeddingReg(nn.Module):
    """
    The regression model implemented with FFNs
    """
    def __init__(self, batch_size, pretrain_embedding, num_embedding, dim_embedding, dim_out, dim_hidden,
                 layers=4, padding_idx=0,
                 drop_rate=0, max_len=20, cuda=False):
        super(NNEmbeddingReg, self).__init__()
        self.embedding = EmbeddingLayer(pretrain_embedding, num_embedding, dim_embedding, padding_idx)
        self.project = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) if i != 0 else nn.Linear(dim_embedding, dim_hidden)
                                      for i in range(layers)])
        self.top = nn.Linear(dim_hidden, dim_out)
        # self.cuda = cuda

    def forward(self, path, word, lengths, batch_size):
        label = self.embedding(path)
        word = self.embedding(word)
        for layer in self.project:
            word = torch.tanh(layer(word))
        word = self.top(word)
        return word, label


class TestEmbeddingReg(nn.Module):
    def __init__(self, batch_size, pretrain_embedding, num_embedding, dim_embedding, dim_out, dim_hidden, padding_idx=0,
                 drop_rate=0, max_len=20, cuda=False):
        super(TestEmbeddingReg, self).__init__()
        self.embedding = EmbeddingLayer(pretrain_embedding, num_embedding, dim_embedding, padding_idx)
        self.path_encoder = PathEncoder(batch_size, dim_embedding, dim_hidden, drop_rate, max_len, cuda=cuda)
        self.project = nn.Linear(dim_embedding, dim_hidden)
        self.top_word = nn.Linear(dim_hidden, dim_out)
        self.top_path = nn.Linear(dim_hidden, dim_out)
        # self.cuda = cuda

        for layer in [self.top_path, self.project, self.top_word]:
            for p in layer.parameters():
                if p.requires_grad and p.dim() > 2:
                    xavier_normal_(p)

    def forward(self, path, word, lengths, batch_size):
        label = self.path_encoder(self.embedding(path), lengths, batch_size)
        label = self.top_path(F.hardtanh(label))
        # print(path_encoded.size(), word_emebed.size())
        word_out = self.top_word(F.hardtanh(self.project(F.hardtanh(self.embedding(word)))))
        return word_out, label


# test scripts
if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    embedding = torch.as_tensor(nn.Embedding(100, 10, padding_idx=0).weight)
    embedding.requires_grad = False
    # print(embedding)
    model = BinaryClassifier(100, embedding, 100, 10, 128)

    path = []
    word = []
    tgt = []
    for i in range(100):
        path.append(np.random.randint(0, 100, 5))
        word.append(np.random.randint(0, 100, 1))
        tgt.append(np.random.randint(0,2, 1))
    # print(np.asarray(word))
    pa = torch.tensor(np.asarray(path), dtype=torch.long, requires_grad=False)
    # print(path.size())
    pa = pa.transpose(0, 1)
    wo = torch.tensor(np.asarray(word), dtype=torch.long, requires_grad=False)
    tgt = torch.tensor(np.asarray(tgt), dtype=torch.long, requires_grad=False)
    wo = wo.squeeze()
    tgt = tgt.squeeze()

    # print(word.size())
    # print(word)
    # print(model(path, word))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_func = nn.CrossEntropyLoss()

    epochs = tqdm(range(1000))
    for epoch in epochs:
        model.zero_grad()
        model.path_encoder.init_hidden()
        output = model(pa, wo)
        # print(output)
        loss = loss_func(output, tgt)
        _, predicted = torch.max(output, 1)
        # print(predicted)
        # print(tgt)
        acc = sum(predicted == tgt) / 100
        # print(loss.item(), epoch)
        loss.backward()
        optimizer.step()
        epochs.set_description(f'acc = {acc:.4f} loss = {loss.item() : .6f}')


