import torch
from torch.utils.data import Dataset, DataLoader


def load_data(file, synset2idx, model_opt, opt, max_seq_len, shuffle=True):
    # load training/validation data
    if model_opt != 'nn':
        data = DataLoader(
            DataProducer(file, synset2idx, cuda=opt.cuda, max_len=max_seq_len, debug=opt.debug),
            batch_size=opt.batch_size, shuffle=shuffle)
    else:
        data = DataLoader(
            NNDataProducer(file, synset2idx, cuda=opt.cuda, max_len=max_seq_len),
            batch_size=opt.batch_size, shuffle=shuffle)
    return data


def load_query(file, synset2idx, model_opt, opt, max_seq_len):
    if model_opt != 'nn':
        if opt.debug:
            query_raw = QueryProducerDebug(file, synset2idx, cuda=opt.cuda, max_len=max_seq_len)
        else:
            query_raw = QueryProducer(file, synset2idx, cuda=opt.cuda, max_len=max_seq_len)
    else:
        query_raw = NNQueryProducer(opt.query, synset2idx, cuda=opt.cuda, max_len=max_seq_len)
    return query_raw, DataLoader(query_raw, batch_size=opt.batch_size_valid * 2, shuffle=False)


def load_hyponym(file, synset2idx):
    with open(file) as in_f:
        hyponyms = []
        for idx, line in enumerate(in_f):
            l = line.strip().split()

            # load positive examples only
            if l[-1] != '1':
                continue
            synset = l[-2]

            if synset not in synset2idx:
                print(synset)
                print(synset2idx['clangor.v.01'])
                raise ValueError(f'Synset {synset} not found in indexer')

            hyponyms.append(synset2idx[synset] if synset in synset2idx else len(synset2idx))
    return hyponyms


class DataProducer(Dataset):
    def __init__(self, file, synset2idx, neg=False, cuda=False, max_len=20, debug=False):
        with open(file) as in_f:
            lines = in_f.readlines()
        if debug:
            lines = lines[:1024]
        # print(len(lines))

        self.seqs = sorted([line.strip().split() for line in lines], key=lambda x: len(x), reverse=True)
        # print(self.seqs[:100])
        # print(len(self.seqs))
        if not neg:
            self.seqs = [instance for instance in self.seqs if instance[-1] == '1']

            # print(f'neg: {neg} len: {len(self.seqs)}')
        self.syn2idx = synset2idx

        self.cuda = cuda
        self.max_len = max_len

    def __getitem__(self, item):
        instance = self.seqs[item]

        seq = [self.syn2idx[synset] if synset in self.syn2idx else len(self.syn2idx) for synset in instance[:-2]]
        seq_len = len(seq)
        if seq_len < self.max_len:
            for i in range(self.max_len - seq_len):
                seq.append(0)
        elif seq_len > self.max_len:
            seq = seq[-self.max_len:]

        seq_len = min(seq_len, self.max_len)

        if instance[-2] in self.syn2idx:
            word = self.syn2idx[instance[-2]]
        else:
            word = len(self.syn2idx) - 1
        label = int(instance[-1])

        path = torch.tensor(seq, device='cpu' if not self.cuda else 'cuda')
        word = torch.tensor(word, device='cpu' if not self.cuda else 'cuda')
        label = torch.tensor(label, dtype=torch.long, device='cpu' if not self.cuda else 'cuda')
        seq_len = torch.tensor(seq_len, device='cpu' if not self.cuda else 'cuda')

        return path, seq_len, word, label

    def __len__(self):
        return len(self.seqs)


class QueryProducer(Dataset):
    def __init__(self, query, synset2idx, cuda=False, max_len=20):
        with open(query) as in_f:
            raw_querys = in_f.readlines()

        self.querys = sorted([line.strip().split() for line in raw_querys], key=lambda x: len(x), reverse=True)
        self.syn2idx = synset2idx

        self.cuda = cuda
        self.max_len = max_len

    def __getitem__(self, item):
        instance = self.querys[item]

        seq = [self.syn2idx[synset] if synset in self.syn2idx else len(self.syn2idx) for synset in instance]
        seq_len = len(seq)
        if seq_len < self.max_len:
            for i in range(self.max_len - seq_len):
                seq.append(0)
        elif seq_len > self.max_len:
            seq = seq[-self.max_len:]

        seq_len = min(seq_len, self.max_len)

        path = torch.tensor(seq, device='cpu' if not self.cuda else 'cuda')
        seq_len = torch.tensor(seq_len, device='cpu' if not self.cuda else 'cuda')

        return path, seq_len, instance

    def __len__(self):
        return len(self.querys)


class QueryProducerDebug(Dataset):
    def __init__(self, query, synset2idx, cuda=False, max_len=20):
        with open(query) as in_f:
            raw_querys = in_f.readlines()[:1024]

        self.querys = sorted([line.strip().split() for line in raw_querys], key=lambda x: len(x), reverse=True)
        # print(self.querys[:100])
        # print(len(self.querys))
        self.querys = [instance[:-2] for instance in self.querys if instance[-1] == '1']

        self.syn2idx = synset2idx

        self.cuda = cuda
        self.max_len = max_len

    def __getitem__(self, item):
        instance = self.querys[item]

        seq = [self.syn2idx[synset] if synset in self.syn2idx else len(self.syn2idx) for synset in instance]
        seq_len = len(seq)
        if seq_len < self.max_len:
            for i in range(self.max_len - seq_len):
                seq.append(0)
        elif seq_len > self.max_len:
            seq = seq[:self.max_len]

        seq_len = min(seq_len, self.max_len)

        path = torch.tensor(seq, device='cpu' if not self.cuda else 'cuda')
        seq_len = torch.tensor(seq_len, device='cpu' if not self.cuda else 'cuda')

        return path, seq_len, instance

    def __len__(self):
        return len(self.querys)


class NNDataProducer(Dataset):
    def __init__(self, file, synset2idx, cuda=False, max_len=20, test=False):
        with open(file) as in_f:
            lines = in_f.readlines()
        if test:
            lines = lines[:1024]

        self.seqs = [line.strip().split()[-3:] for line in lines]

        self.syn2idx = synset2idx

        self.cuda = cuda
        self.max_len = max_len

    def __getitem__(self, item):
        instance = self.seqs[item]
        seq = self.syn2idx[instance[-3]] if instance[-3] in self.syn2idx else len(self.syn2idx)
        seq_len = 1

        if instance[-2] in self.syn2idx:
            word = self.syn2idx[instance[-2]]
        else:
            word = len(self.syn2idx) - 1
        label = int(instance[-1])

        path = torch.tensor(seq, device='cpu' if not self.cuda else 'cuda')
        word = torch.tensor(word, device='cpu' if not self.cuda else 'cuda')
        label = torch.tensor(label, dtype=torch.long, device='cpu' if not self.cuda else 'cuda')
        seq_len = torch.tensor(seq_len, device='cpu' if not self.cuda else 'cuda')

        return path, seq_len, word, label

    def __len__(self):
        return len(self.seqs)


class NNQueryProducer(Dataset):
    def __init__(self, query, synset2idx, cuda=False, max_len=20):
        with open(query) as in_f:
            raw_querys = in_f.readlines()

        self.querys = [line.strip().split()[-1] for line in raw_querys]
        self.syn2idx = synset2idx

        self.cuda = cuda
        self.max_len = max_len

    def __getitem__(self, item):
        instance = [self.querys[item]]

        seq = self.syn2idx[instance[0]] if instance[0] in self.syn2idx else len(self.syn2idx)
        seq_len = 1

        path = torch.tensor(seq, device='cpu' if not self.cuda else 'cuda')
        seq_len = torch.tensor(seq_len, device='cpu' if not self.cuda else 'cuda')

        return path, seq_len, instance

    def __len__(self):
        return len(self.querys)

