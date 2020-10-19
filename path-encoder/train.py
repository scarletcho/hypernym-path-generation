import copy
import math
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_output, save_model
from validation import valid


# a wrapper for loss functions to accommodate different models
# classification models                   -> CrossEntropy loss
# regression models W/O negative examples -> Mean Square loss
# regression models W/  negative examples -> Margin Ranking loss
def criterion(*args, reg=False, neg=False):
    if not reg:
        assert len(args) == 2
        return F.cross_entropy(*args)
    else:
        if not neg:
            assert len(args) == 2
            return F.mse_loss(*args)
        else:
            assert len(args) == 8
            output, label, neg_output, neg_label, y, pos_count, neg_count, margin = args
            return F.margin_ranking_loss(
                torch.norm((output - label) , 2, dim=1),
                torch.norm((neg_output - neg_label) , 2, dim=1), y,
                margin=margin, reduction='sum')  # * math.sqrt(neg_count / (pos_count + neg_count))


# split output into positive/negative output based on their labels during training and validating
def get_output_neg(label, output, target, device, arg):
    pos_count = sum(label == 1).item()
    neg_count = arg.batch_size - pos_count
    label = label.view(-1, 1).repeat(1, 300).float()

    neg_label = (label == 0).float()

    pos_output, pos_target = output * label, target * label
    neg_output, neg_target = output * neg_label, target * neg_label

    y = torch.tensor([-1], dtype=torch.float).to(device)
    return pos_output, pos_target, neg_output, neg_target, y, pos_count, neg_count, arg.gamma


def evaluate_batch(model, batch, total_seq, total_loss, correct, device, model_opt, opt):
    path, seq_len, word, label = batch
    if model_opt != 'nn':
        model.path_encoder.init_hidden(len(path))
    if model_opt == 'pathclassifier':
        output, ___ = model(path.t(), word, seq_len, len(path))
        outputs = (output, label)
    else:
        output, target = model(path, word, seq_len, len(path))
        if not opt.neg:
            outputs = (F.softmax(output, dim=1), F.softmax(target, dim=1))
        else:
            outputs = get_output_neg(label, output, target, device, opt)

    loss = criterion(*outputs, reg=model_opt != 'pathclassifier', neg=opt.neg)

    total_seq += len(label)
    total_loss += loss.item() * len(label)
    if model_opt == 'pathclassifier':
        _, predicted = torch.max(output, 1)
        correct += sum(predicted == label).item()

    return loss, total_seq, total_loss, correct


def train(model, optimizer, train_data, dev_data, query_data, query_raw, hyponym, idx2synset, _epoch, model_opt, opt,
          device, baseline=False):
    best_dev_loss = 1000
    for epoch in range(_epoch, opt.epoch):
        model.train()
        batches = tqdm(train_data, ascii=True)

        total_seq = 0
        correct = 0
        total_loss = 0.

        for i, batch in enumerate(batches):
            optimizer.zero_grad()
            loss, total_seq, total_loss, correct = evaluate_batch(model, batch, total_seq, total_loss, correct, device,
                                                                  model_opt, opt)
            if model_opt == 'pathclassifier':
                batches.set_description(
                    f'epoch: {epoch} train_acc = {correct / total_seq:.4f} train_loss = {total_loss / total_seq : .6f}')
            else:
                batches.set_description(
                    f'epoch: {epoch} train_loss = {total_loss / total_seq : .6f}')

            if not baseline:
                loss.backward()
                optimizer.step()

        # start to test the model on the validation dataset
        with torch.no_grad():
            model.eval()
            batches_dev = tqdm(dev_data, ascii=True)
            total_seq_dev = 0
            correct_dev = 0
            total_loss_dev = 0.

            for batch in batches_dev:
                loss, total_seq_dev, total_loss_dev, correct_dev = evaluate_batch(model, batch, total_seq_dev,
                                                                                  total_loss_dev, correct_dev, device,
                                                                                  model_opt, opt)
                if model_opt == 'pathclassifier':
                    batches_dev.set_description(
                        f'epoch: {epoch} __dev_acc = {correct_dev / total_seq_dev:.4f} __dev_loss = {total_loss_dev / total_seq_dev : .6f}')
                else:
                    batches_dev.set_description(
                        f'epoch: {epoch} __dev_loss = {total_loss_dev / total_seq_dev : .6f}')

            # if opt.valid is set to True, the querying process will begin
            # this will generate predicted hypernyms on the validation dataset
            if opt.valid:
                output = valid(model, query_data, query_raw, hyponym, idx2synset, model_opt, opt, device)
        # if opt.save_best is True, only the best result (based on validation loss) will be stored
        # otherwise output from each epoch will be written into a different file
        if opt.save_best and total_loss_dev < best_dev_loss:
            best_output = output
        else:
            save_output(*output, epoch, opt)

        if opt.save_model:
            # if opt.save_best is True, only the best model parameters (based on validation loss) will be stored
            if opt.save_best:
                if total_loss_dev < best_dev_loss:
                    print(f'best_model set to epoch{epoch}')
                    best_model = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                                  total_loss_dev, epoch, opt)
                    best_dev_loss = total_loss_dev
            else:
                save_model(model.state_dict(), optimizer.state_dict(), total_loss_dev, epoch, opt)

        logging.info(f'epoch: {epoch} train_loss = {total_loss / total_seq  : .6f} '
                     f'dev_loss = {total_loss_dev / total_seq_dev : .6f}')

    if opt.save_best:
        if opt.save_model:
            save_model(*best_model)
        if opt.valid:
            save_output(*best_output)
