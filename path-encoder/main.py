import torch
import argparse
import logging
from torch import optim
from model import BinaryClassifier, EmbeddingReg, NNEmbeddingReg, NearestNeighbor
from utils import get_synset2index, load_embeddings, save_output
from dataloading import load_data, load_query, load_hyponym
from train import train
from validation import valid


OUT_DIM = 300


def setup_model(embeddings, hidden_dim, out_dim, opt, max_seq_len):
    model_opt = opt.model.lower()

    # initialize the model
    model_type = {'pathclassifier': BinaryClassifier, 'pathencoder': EmbeddingReg, 'nn': NNEmbeddingReg,
                  'baseline': NearestNeighbor}
    if model_opt in model_type:
        model = model_type[model_opt](opt.batch_size, embeddings, len(embeddings), opt.embedding_dim, out_dim,
                                      hidden_dim, drop_rate=opt.drop_out, cuda=opt.cuda, max_len=max_seq_len)
    else:
        raise ValueError('model must be PathEncoder, PathClassifier, NN or baseline')
    return model


def main(opt):
    HIDDEN_DIM = opt.hidden
    max_seq_len = opt.seq_len if opt.seq_len > 0 else 20
    print(f'max_seq_len is set to be {max_seq_len}')

    logging.basicConfig(filename=f'logger.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f'Prediction Mode : {opt.model}')
    logging.info(f'START TRAINING')
    logging.info(f'max_seq_len is set to be {max_seq_len}')
    logging.info(f'SETTINGS : epoch : {opt.epoch} batch_size : {opt.batch_size}')
    logging.info(f'SETTINGS : learning_rate : {opt.lr} drop_rate : {opt.drop_out}')
    logging.info(f'SETTINGS : hidden_dim : {HIDDEN_DIM} gamma:{opt.gamma}')
    logging.info(f'TRAINING DATA: {opt.train}')

    synset2idx = get_synset2index('data/all_wn18.path')
    idx2synset = list(synset2idx.keys())

    embeddings = load_embeddings(opt.embeddings, synset2idx.keys(), debug=opt.debug)

    model_opt = opt.model.lower()

    # load queries (path from root to direct hypernyms)
    query_data, query_loader = load_query(opt.query, synset2idx, model_opt, opt, max_seq_len)

    # initialize the model
    model = setup_model(embeddings, HIDDEN_DIM, OUT_DIM, opt, max_seq_len)

    baseline_indicator = False
    if model_opt == 'baseline':
        baseline_indicator = True

    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    _epoch = 0

    if opt.load_model:
        checkpoint = torch.load(opt.load_model)
        print(f'loading model from {opt.load_model}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _epoch = checkpoint['epoch'] + 1

    # start training loops
    if not opt.test_only:
        # load training/validation data
        train_data = load_data(opt.train, synset2idx, model_opt, opt, max_seq_len, shuffle=True)
        dev_data = load_data(opt.dev, synset2idx, model_opt, opt, max_seq_len)

        # also load hyponyms for writing output
        valid_hyponyms = load_hyponym(opt.dev, synset2idx)

        train(model, optimizer, train_data, dev_data, query_loader, query_data, valid_hyponyms, idx2synset, _epoch, model_opt,
              opt, device, baseline=baseline_indicator)

    if opt.test:
        test_hyponyms = load_hyponym(opt.test, synset2idx)
        output = valid(model, query_loader, query_data, test_hyponyms, idx2synset, model_opt, opt, device)
        save_output(*output, opt.epoch, opt, test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default='ft-embs-all-lower.vec')
    parser.add_argument("--embedding_dim", default=300)
    parser.add_argument("--train", default='data/train11.path')
    parser.add_argument("--dev", default='data/valid.path')
    parser.add_argument("--test", default=None)
    parser.add_argument("--valid", '-v', action='store_const', const=True, default=False)
    parser.add_argument("--test_only", '-o', action='store_const', const=True, default=False)
    parser.add_argument("--query", default='data/all_wn18_path')
    parser.add_argument("--epoch", '-e', type=int, default=10)
    parser.add_argument("--load_model", '-l', default=None)
    parser.add_argument("--batch_size", '-b', type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch_size_valid", type=int, default=512)
    parser.add_argument("--drop_out", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_model", '-s', action='store_const', const=True, default=False)
    parser.add_argument("--save_best", action='store_const', const=True, default=False)
    parser.add_argument("--debug", action='store_const', const=True, default=False)
    parser.add_argument("--model", "-m", default='PathEncoder', help='{PathEncoder, PathClassifier, NN, baseline}')
    parser.add_argument("--cuda", action='store_const', const=True, default=False)
    parser.add_argument("--neg", action='store_const', const=True, default=False)
    parser.add_argument("--gamma", default=0.0, type=float, help='margin_loss_hyper-parameter')
    parser.add_argument("--hit", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=2)

    options = parser.parse_args()

    main(options)
