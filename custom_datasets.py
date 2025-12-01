import csv
#from pickle import NONE
import torch
from transformers import BartTokenizer, BertTokenizer, RobertaTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, AutoModelForTokenClassification
from torch.utils.data import Dataset
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from transformers.models.idefics.image_processing_idefics import valid_images
from datasets import load_dataset
from collections import Counter
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ckpt = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_ckpt)
word_ckpt = 'vblagoje/bert-english-uncased-finetuned-pos'
word_tokenizer = AutoTokenizer.from_pretrained(word_ckpt)
model_pos = AutoModelForTokenClassification.from_pretrained(word_ckpt).to(device)
RoBERTa_ckpt = "roberta-base"
roBerta_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_ckpt)
label_map = model_pos.config.id2label

class GANBARTDataset(Dataset):
    def __init__(self, sentence1_list, sentence2_list, sentence3_list = None, max_length=512, is_unsupervised = False, is_noun = False):
        """
        Args:
            sentence1_list (list of str): List of sentence 1 strings.
            sentence2_list (list of str): List of sentence 2 strings.
            labels (list of int, optional): List of labels for the sentences (default is None).
            max_length (int, optional): Maximum token length for padding/truncation (default is 512).
        """
        self.sentence1_list = sentence1_list
        self.sentence2_list = sentence2_list
        self.max_length = max_length
        self.is_unsupervised = is_unsupervised
        self.is_noun = is_noun
        if self.is_noun:
           self.sentence3_list = sentence3_list
    def __len__(self):
        return len(self.sentence1_list)

    def __getitem__(self, idx):

        sentence = self.sentence1_list[idx]
        labels = self.sentence2_list[idx]
        if self.is_noun:
          key_nouns= self.sentence3_list[idx]


        # Tokenize the Lecture and Summary
        encoding = tokenizer(
            sentence, text_target = labels,  # using Lecture and Summary
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        bert_encoding = roBerta_tokenizer(labels, # using Summary only
            padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        )
        if self.is_noun:
            noun_encoding = roBerta_tokenizer(key_nouns, # using Summary only
                padding="max_length", truncation=True, max_length=32, return_tensors="pt"
            )
        # Get the tokenized inputs (input_ids, attention_mask)
        input_ids = encoding['input_ids'].squeeze(0)  # Shape (1, seq_len) -> remove batch dimension
        label = encoding['labels'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        #print( bert_encoding)
        bert_input_id = bert_encoding['input_ids'].squeeze(0)
        bert_mask = bert_encoding['attention_mask'].squeeze(0)
        bert_noun = {}
        mask_noun = {}
        if self.is_noun:
            bert_noun = noun_encoding['input_ids'].squeeze(0)
            mask_noun = noun_encoding['attention_mask'].squeeze(0) 

        #bert_finput_id = bert_fencoding['input_ids'].squeeze(0)
        #bert_fmask = bert_fencoding['attention_mask'].squeeze(0)

        if not self.is_unsupervised:
            return {'input_ids':input_ids , 
            'attention_mask': attention_mask , 
            'label': label, 
            'bert_input_id': bert_input_id,
            'bert_mask': bert_mask,
            'mask_noun': mask_noun,
            'bert_noun': bert_noun
            }
        else:
            return {'input_ids':input_ids , 
            'attention_mask': attention_mask , 
            'bert_input_id': bert_input_id, 
            'bert_mask': bert_mask,
            'bert_noun': bert_noun,
            'mask_noun': mask_noun
            }

class pseudoDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def create_dataset(is_argument = False ,lecture_path=None, summary_path=None, is_test = False):
    # create datasets
    if is_argument:
        with open('./Datasets/lecture_data_aug.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[1][2480:])
        test_summary = list(data_set[2][2480:])
        t_Lecture = list(data_set[1][:2400])
        t_summary = list(data_set[2][:2400])
        v_Lecture = list(data_set[1][2400:2480])
        v_summary = list(data_set[2][2400:2480])
    else:
        with open('./Datasets/generated_lectures.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[3][900:])
        test_summary = list(data_set[4][900:])
        t_Lecture = list(data_set[3][:300])
        t_summary = list(data_set[4][:300])
        u_Lecture = list(data_set[3][300:800])
        v_Lecture = list(data_set[3][800:900])
        v_summary = list(data_set[4][800:900])
        # create and return datasets
    test_dataset = GANBARTDataset(test_Lecture, test_summary)
    train_dataset = GANBARTDataset(t_Lecture, t_summary)
    validation_dataset = GANBARTDataset(v_Lecture, v_summary)
    del data_set
    del t_Lecture
    del t_summary
    del v_Lecture
    del v_summary
    del test_Lecture
    del test_summary
    return train_dataset, validation_dataset, test_dataset

def samsum_dataset(is_argument = False ,lecture_path=None, summary_path=None, is_test = False):
    # create datasets
    if is_argument:
        with open('/content/drive/MyDrive/COMP_8730_samsum/Datasets/samsum.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[1][2480:])
        test_summary = list(data_set[2][2480:])
        t_Lecture = list(data_set[1][:2400])
        t_summary = list(data_set[2][:2400])
        v_Lecture = list(data_set[1][2400:2480])
        v_summary = list(data_set[2][2400:2480])
    else:
        with open('/content/drive/MyDrive/COMP_8730_samsum/Datasets/samsum.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        print(len(data_set[1]))
        test_Lecture = list(data_set[1][4400:4800]) # 12786
        test_summary = list(data_set[2][4400:4800]) # 12786
        t_Lecture = list(data_set[1][:4000]) # 11786
        t_summary = list(data_set[2][:4000]) # 11786
        v_Lecture = list(data_set[1][4000:4400])
        v_summary = list(data_set[2][4000:4400])
        # create and return datasets
    test_dataset = GANBARTDataset(test_Lecture, test_summary)
    train_dataset = GANBARTDataset(t_Lecture, t_summary)
    validation_dataset = GANBARTDataset(v_Lecture, v_summary)
    del data_set
    del t_Lecture
    del t_summary
    del v_Lecture
    del v_summary
    del test_Lecture
    del test_summary
    return train_dataset, validation_dataset, test_dataset

def get_samsum(is_unsupervised=True):
    ds_raw = load_dataset("knkarthick/samsum")
    ds = ds_raw.map(extract_nouns_batch, batched=True, batch_size=16)
    #print(list(ds["train"]["nouns"])[0])
    if not is_unsupervised:
      #print(ds["train"]["dialogue"].shape)
      train_dataset = GANBARTDataset(list(ds["train"]["dialogue"]), list(ds["train"]["summary"]), list(ds["train"]["nouns"]), is_noun = True)
      validation_dataset = GANBARTDataset(list(ds["validation"]["dialogue"]), list(ds["validation"]["summary"]))
      test_dataset = GANBARTDataset(list(ds["test"]["dialogue"]), list(ds["test"]["summary"]))
      del ds
      return train_dataset, validation_dataset, test_dataset
    else:
      #total_dataset = GANBARTDataset(list(ds["train"]["dialogue"]), list(ds["train"]["summary"]))
      split_dataset = ds["train"].train_test_split(test_size=0.95,seed=42)
      #print(split_dataset["train"]["nouns"][0])
      train_dataset = GANBARTDataset(list(split_dataset['train']["dialogue"]), list(split_dataset['train']["summary"]), list(split_dataset["train"]["nouns"]), is_noun = True)
      u_train_dataset = GANBARTDataset(list(split_dataset['test']["dialogue"]), list(split_dataset['test']["summary"]), list(split_dataset["test"]["nouns"]), is_unsupervised = True, is_noun = True)
      validation_dataset = GANBARTDataset(list(ds["validation"]["dialogue"]), list(ds["validation"]["summary"]),is_noun = False)
      test_dataset = GANBARTDataset(list(ds["test"]["dialogue"]), list(ds["test"]["summary"]),is_noun = False)
      del ds
      return train_dataset, validation_dataset, test_dataset, u_train_dataset

def extract_top_nouns_hf(text, top_k=8):
    tokens = word_tokenizer(text, return_tensors="pt", truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model_pos(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    words = word_tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    labels = [label_map[p.item()] for p in predictions]

    noun_list = []
    for word, label in zip(words, labels):
        if label in ["NOUN", "PROPN"]:
            noun_list.append(word.replace("##", ""))  # remove WordPiece prefix
    freq = Counter(noun_list)
    return [w for w, _ in freq.most_common(top_k)]

def extract_nouns_batch(batch):
    batch_nouns = []
    for text in batch["dialogue"]:
        dia = ', '.join(extract_top_nouns_hf(text))
        batch_nouns.append(dia)  # uses GPU if model on cuda
    return {"nouns": batch_nouns}

def concatenate_summary_keyword(batch_input_id, batch_mask, batch_knoun, batch_knoun_mask, device, ro_tokenizer):
    input_token = batch_input_id
    batch_size = input_token.shape[0]
    prompt_summary = torch.tensor([48600, 35]).unsqueeze(0).repeat(batch_size, 1).type_as(input_token)
    prompt_keyward = torch.tensor([32712, 35]).unsqueeze(0).repeat(batch_size, 1).type_as(input_token)
    prompt_mask = torch.tensor([1, 1]).unsqueeze(0).repeat(batch_size, 1).type_as(input_token)
    input_token[:, 0] = ro_tokenizer.sep_token_id
    input_token = torch.cat([input_token[:,:1], prompt_summary, input_token[:,1:]], dim=1)
    batch_mask = torch.cat([batch_mask[:,:1], prompt_mask, batch_mask[:,1:]], dim=1)
    batch_knoun = torch.cat([batch_knoun[:,:1], prompt_keyward, batch_knoun[:,1:]], dim=1)
    batch_knoun_mask = torch.cat([batch_knoun_mask[:,:1], prompt_mask, batch_knoun_mask[:,1:]], dim=1)
    new_torch = torch.cat((batch_knoun, input_token), dim=1)
    new_mask = torch.cat((batch_knoun_mask, batch_mask), dim=1)
    cleaned = [ids[mask.bool()] for ids, mask in zip(new_torch, new_mask)]
    padded_again = pad_sequence(cleaned, batch_first=True, padding_value=1)
    padded_again_ids = F.pad(padded_again, (0, 256 - padded_again.shape[1]), value=1).to(device)
    padded_attention_masks = (padded_again_ids != ro_tokenizer.pad_token_id).long().to(device)
    return padded_again_ids, padded_attention_masks


