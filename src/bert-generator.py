import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

model_version = "bert-base-uncased"
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(
    model_version, do_lower_case=model_version.endswith("uncased")
)


def tokenize_batch(batch):
    return np.array([tokenizer.convert_tokens_to_ids(sent) for sent in batch])


def untokenize_batch(batch):
    return np.array([tokenizer.convert_ids_to_tokens(sent) for sent in batch])


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch)


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]


class BertGenerator(nn.Module):
    def __init__(self, cuda=False):
        super(BertGenerator, self).__init__()

        # Load pre-trained model (weights)
        model_version = "bert-base-uncased"
        self.model = BertForMaskedLM.from_pretrained(model_version)
        self.model.eval()
        if cuda:
            self.model = self.model.cuda()

    def forwardf(self, x):
        # not implemented yet (step has the same functionality)
        pass

    def step(self, x, idx):
        out = self.model(x)
        logits = out[:, idx]
        return F.softmax(logits)

    def sample(self, batch, max_len):
        for i in range(1, 1 + max_len):
            inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
            logits = self.step(inp, i)
            batch[:, i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
        return batch


class BertDiscriminator(nn.Module):
    def __init__(self, cuda):
        super(BertDiscriminator, self).__init__()

        # Load pre-trained model (weights)
        model_version = "bert-base-uncased"
        self.model = BertForMaskedLM.from_pretrained(model_version)
        self.model.eval()
        if cuda:
            self.model = self.model.cuda()

        self.lin1 = nn.Linear(30522, 1)

    def forward(self, x):
        # Assume  that the first item is a classifier token
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)

        y = self.model(inp)
        y = y[:, 0, :]
        y = self.lin1(y)
        y = F.sigmoid(y)

        return y



