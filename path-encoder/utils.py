import numpy as np
import torch
import os
import logging
from tqdm import tqdm


def get_synset2index(dataset):
    synsets2idx = {'<PAD>': 0}
    with open(dataset) as in_f:
        for line in in_f:
            for synset in line.strip().split('\t'):
                if synset not in synsets2idx:
                    synsets2idx[synset] = len(synsets2idx)
    return synsets2idx


def load_embeddings(embeddings_f, synset2index, debug=False):
    # print(f'loading pretrained embeddings from: {embeddings_f}')
    synset2embedding = {}

    with open(embeddings_f) as in_f:
        lines = tqdm(in_f.readlines(), desc=f'loading embeddings from {embeddings_f}', ascii=True)
        for line in lines:
            l = line.strip().split()
            if len(l) > 1:
                synset2embedding[l[0]] = np.asarray([float(item) for item in l[1:]])
        lines.set_description('loading embeddings ---done!')
    embedding_dim = len(synset2embedding[next(iter(synset2embedding))])
    embeddings = []
    num_unseen = 0
    unseen_synsets = []

    if debug:
        f = open('embedding_log', 'w')
        print([i for i in zip(synset2index, range(len(synset2index)))], file=f)
        syns = []

    for synset in synset2index:
        if debug:
            syns.append(synset)

        if synset in synset2embedding:
            embeddings.append(synset2embedding[synset])
        else:
            num_unseen += 1
            unseen_synsets.append(synset)
            if len(embeddings) == 0:
                embeddings.append(np.zeros(embedding_dim))
            else:
                embeddings.append(np.random.uniform(-0.8, 0.8, embedding_dim))

    if debug:
        print(syns, file=f)
        print(len(syns), file=f)

    embeddings.append(np.random.uniform(-0.8, 0.8, embedding_dim))

    if debug:
        print(embeddings[:300], file=f)

    embeddings = torch.from_numpy(np.asarray(embeddings))

    if debug:
        print(embeddings[:300], file=f)
        f.close()

    # print(embeddings.size())
    print(f'{num_unseen} synsets not found in embeddings')
    print(f'{len(embeddings) - num_unseen} synsets found in embeddings')
    print(unseen_synsets)
    return embeddings


# write output from the model to the corresponding folder
def save_output(data, query, output, id_to_synset, epoch, arg, test=False):
    if not test:
        if not os.path.isdir('output'):
            os.mkdir('output')
        out_dir = os.path.realpath('output')
    else:
        if not os.path.isdir('test_output'):
            os.mkdir('test_output')
        out_dir = os.path.realpath('test_output')

    if not test:
        file_name = arg.train.split('/')[-1]
    else:
        file_name = arg.test.split('/')[-1]

    if not os.path.isdir(os.path.join(out_dir, f'{file_name}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}')):
        os.mkdir(os.path.join(out_dir, f'{file_name}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}'))

    with open(f'{out_dir}/{file_name}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}/epoch{epoch}.out', 'w') as out_f:
        assert len(output) == len(data)
        for idx, item in enumerate(output):
            assert len(item) == arg.hit
            for out in item:
                _, __, path = query[out]
                if isinstance(path, list):
                    for node in path:
                        out_f.write(node)
                        out_f.write(' ')
                else:
                    out_f.write(path + ' ')
                hypo = data[idx]
                out_f.write(id_to_synset[hypo])
                out_f.write('\n')


# save the model/optimizer parameters to certain file
def save_model(model_state, optimizer_state, loss, epoch, arg):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    chkpt_dir = os.path.realpath('checkpoint')
    train_f = arg.train.split('/')[-1]
    if not os.path.isdir(os.path.join(chkpt_dir, f'{train_f}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}')):
        os.mkdir(os.path.join(chkpt_dir, f'{train_f}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}'))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss}, f'checkpoint/{train_f}_dr_{arg.drop_out}_lr_{arg.lr}_gamma_{arg.gamma}_hid_{arg.hidden}/epoch{epoch}.chkpt')

    logging.info(f'Best performance @ epoch: {epoch} dev_loss = {loss : .6f}')
