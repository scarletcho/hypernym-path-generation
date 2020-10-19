import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm


def valid(model, query_data, query_raw, hyponym, idx2synset, model_opt,  opt, device):
    validation = tqdm(hyponym, ascii=True)
    valid_outputs = []

    # for all models, need to generate query_path/synsey(hyponym) pairs for the model
    if model_opt == 'pathclassifier':
        # loop over all synset in the validation dataset
        for synset in validation:
            # prepare queries and output
            query_bacthes = query_data
            query_output = None
            # loop over all queries and store corresponding values in $query_output
            for batch in query_bacthes:
                # feed path from queries and synset from validation dataset into the model
                path, seq_len, _ = batch
                synset_batch = [synset for i in range(len(path))]
                synset_batch = torch.tensor(synset_batch, device='cpu' if not opt.cuda else 'cuda')
                model.path_encoder.init_hidden(len(path))
                output = model(path, synset_batch, seq_len, len(path))[0]

                if query_output is not None:
                    query_output = torch.cat((query_output, output))
                else:
                    query_output = output
                validation.set_description(f'---validating---')
            # get top $hit outputs
            _, predicted = torch.topk(query_output, opt.hit, dim=0)
            valid_outputs.append(predicted[:, 1].tolist())
    else:
        # in regression models, the embeddings for the query paths are identical for each input hyponym
        # for faster evaluation, all these embeddings are generated in the beginning

        # this synset_batch is created only for consistent input
        synset_batch = [0 for i in range(opt.batch_size_valid)]
        synset_batch = torch.tensor(synset_batch, device='cpu' if not opt.cuda else 'cuda')
        queries = None
        tmp_path, tmp_seq_len = None, None
        # loop over queries and store embeddings of query paths in $queries
        for i, batch in enumerate(query_data):
            path, seq_len, _ = batch

            if model_opt != 'nn':
                model.path_encoder.init_hidden(len(path))
            _, label = model(path, synset_batch, seq_len, len(path))

            if not opt.neg:
                _, label = F.softmax(_), F.softmax(label)
            # print(queries)
            if queries is not None:
                queries = torch.cat((queries, label))
            else:
                queries = label.clone()

            if i == len(query_data) - 1:
                if model_opt != 'nn':
                    tmp_path, tmp_seq_len = path[0, :].clone().detach().unsqueeze(0), \
                                            torch.tensor([seq_len[0]]).to(device)
                else:
                    tmp_path, tmp_seq_len = path[0].clone().detach(), seq_len[0].clone().detach()

        # loop over all input hyponym from the validation dataset
        for i, synset in enumerate(validation):
            synset_batch = [synset]
            synset_batch = torch.tensor(synset_batch, device='cpu' if not opt.cuda else 'cuda')
            if model_opt != 'nn':
                model.path_encoder.init_hidden(1)
            output, __ = model(tmp_path, synset_batch, tmp_seq_len, 1)

            if not opt.neg:
                output = F.softmax(output)

            output = output.repeat(len(queries), 1)
            out = torch.sqrt((output - queries) ** 2)

            _, predicted = torch.topk(torch.sum(out, dim=1), opt.hit, dim=0, largest=False)
            valid_outputs.append(predicted.tolist())
            validation.set_description(f'---validating---')
    return hyponym, query_raw, copy.deepcopy(valid_outputs), idx2synset
