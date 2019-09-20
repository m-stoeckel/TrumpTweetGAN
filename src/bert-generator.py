import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 
class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 
class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 class BertGenerator(nn.Module):
  def __init__(self, cuda = False):
    super(BertGenerator, self).__init__()
    
    # Load pre-trained model (weights)
    model_version = 'bert-base-uncased'
    self.model = BertForMaskedLM.from_pretrained(model_version)
    self.model.eval()
    if cuda:
        self.model = self.model.cuda()
        
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))
    
    # Define special tokens
    CLS = '[CLS]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    
  def tokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_tokens_to_ids(sent) for sent in batch])

  def untokenize_batch(self, batch):
    return np.array([self.tokenizer.convert_ids_to_tokens(sent) for sent in batch])

  def detokenize(self, sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

        
  def forwardf(self, x):
    # not implemented yet (step has the same functionality)
    pass
  
  
  def step(self, x, idx):
    out = model(x)
    logits = out[:, idx]
    return F.softmax(logits)
    
  def get_init_text(self, seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    
    return self.tokenize_batch(batch)
  
  def sample(self, batch, max_len):
    for i in range(1, 1+max_len):
      inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
      logits = self.step(inp, i)
      batch[:,i] = logits.multinomial(1).cpu().detach().numpy().reshape(-1)
    return batch
 