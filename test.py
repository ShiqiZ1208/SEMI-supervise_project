from transformers.models.maskformer.modeling_maskformer import pair_wise_sigmoid_focal_loss
from custom_datasets import create_dataset, samsum_dataset, get_samsum, concatenate_summary_keyword, pseudoDataset
from transformers import RobertaTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ganmodel import BART_base_model, Roberta_discriminator, classification_loss, RoBERTa_base_model
is_load = True
device = "cuda"
data_1,data_2,data_3,data_4 = get_samsum()
eval_dataloader = DataLoader(data_4, shuffle=False, batch_size=8, worker_init_fn=lambda worker_id: np.random.seed(seed))
BART_model_ckpt = 'facebook/bart-base'
RoBERTa_model_ckpt = "roberta-base"
BaseD_model = RoBERTa_base_model(RoBERTa_model_ckpt)
NetD = BaseD_model
BaseG_model = BART_base_model(BART_model_ckpt)
NetG = BaseG_model
if is_load:
    print("loadding from ckpt")
    ckptG = "/content/drive/MyDrive/semi_gan_bart/SaveModel/lora_bartGAN_G_epoch2_8.pt"
    ckptG = torch.load(ckptG, map_location="cuda")
    NetG.load_state_dict(ckptG['model_state'])
    ckptD = "/content/drive/MyDrive/semi_gan_bart/SaveModel/lora_bartGAN_D_epoch2_8.pt"
    ckptD = torch.load(ckptD, map_location="cuda")
    NetD.load_state_dict(ckptD['model_state'])
RBE_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_model_ckpt)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
NetG = NetG.to(device)
NetD = NetD.to(device)
for param in NetG.parameters():
    param.requires_grad = False
for param in NetD.parameters():
    param.requires_grad = False
NetD.eval()
NetG.eval()

psudo_ids = []
psudo_label = []
psudo_mask = []
breaks = 0
num_pass = 0
for batch in eval_dataloader:
    print(len(psudo_ids))
    passed = 0
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_knoun = batch['bert_noun'].to(device)
    batch_knoun_mask = batch['mask_noun'].to(device)


    #print(type(input_ids))
    #print(attention_mask)
    genrated = NetG.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=False, 
            num_beams=4,
            early_stopping=True
            )   
    s_token_id = torch.tensor(0, device=device)
    genrated = genrated[:, 1:]

    genrated = genrated.detach()
    padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1).detach()
    roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()
    roberta_finput_ids, roberta_fattention_masks = concatenate_summary_keyword(padded_input_ids, roberta_attention_masks,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)
    flogits = NetD(roberta_finput_ids, attention_mask=roberta_fattention_masks)
    pred = torch.sigmoid(flogits.logits.detach())
    print(pred)
    threshold = 0.7
    mask = pred > threshold
    passed = mask.sum().item()
    mask = mask.squeeze(-1)
    num_pass += passed
    print(mask)
    padded_output_ids = F.pad(genrated, (0, 512 - genrated.shape[1]), value=1).detach()
    print(padded_output_ids.shape)
    #label_list = padded_output_ids[mask].tolist()
    label_list = [seq for seq in padded_output_ids[mask]]
    mask_list = [seq for seq in attention_mask[mask]]
    ids_list = [seq for seq in input_ids[mask]]
    psudo_label.extend(label_list)
    psudo_mask.extend(label_list)
    psudo_ids.extend(ids_list)
    breaks += 1
    if breaks == 5:
      break

pseudo_data = pseudoDataset(psudo_label,psudo_mask,psudo_ids)
p_dataset = DataLoader(pseudo_data, shuffle=False, batch_size=3, worker_init_fn=lambda worker_id: np.random.seed(seed))

for batch in p_dataset:
    #batch["input_ids"]
    #batch["attention_mask"]
    #batch["labels"]
    tokens1 = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    tokens2 = tokenizer.convert_ids_to_tokens(batch["labels"][0])
    print(tokens1)
    print(tokens2)
#print(len(psudo_label))
#print(len(psudo_ids))
#print(len(psudo_mask))
'''
#print(data_1[0])
device = "cuda"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#tokens1 = tokenizer.convert_ids_to_tokens(data_1[0:2]['input_ids'])
tokens2 = tokenizer.convert_ids_to_tokens(data_1[0]['label'])
tokens3 = tokenizer.convert_ids_to_tokens(data_1[0]['bert_noun'])
batch_input_id = data_1[0:2]['bert_input_id']
batch_mask = data_1[0:2]['bert_mask']
batch_knoun = data_1[0:2]['bert_noun']
batch_knoun_mask = data_1[0:2]['mask_noun']


new_token_id, new_mask = concatenate_summary_keyword(batch_input_id,batch_mask,batch_knoun,batch_knoun_mask,device,tokenizer)
print(new_token_id)
print(new_mask)






input_token = batch_input_id
print(batch_input_id.shape[0])
prompt_summary = torch.tensor([48600, 35]).unsqueeze(0).repeat(2, 1).type_as(input_token)
prompt_keyward = torch.tensor([32712, 35]).unsqueeze(0).repeat(2, 1).type_as(input_token)
prompt_mask = torch.tensor([1, 1]).unsqueeze(0).repeat(2, 1).type_as(input_token)
input_token[:, 0] = tokenizer.sep_token_id
input_token = torch.cat([input_token[:,:1], prompt_summary, input_token[:,1:]], dim=1)
batch_mask = torch.cat([batch_mask[:,:1], prompt_mask, batch_mask[:,1:]], dim=1)
batch_knoun = torch.cat([batch_knoun[:,:1], prompt_keyward, batch_knoun[:,1:]], dim=1)
batch_knoun_mask = torch.cat([batch_knoun_mask[:,:1], prompt_mask, batch_knoun_mask[:,1:]], dim=1)
new_torch = torch.cat((batch_knoun, input_token), dim=1)
new_mask = torch.cat((batch_knoun_mask, batch_mask), dim=1)
cleaned = [ids[mask.bool()] for ids, mask in zip(new_torch, new_mask)]
padded_again = pad_sequence(cleaned, batch_first=True, padding_value=1)
padded_again_ids = F.pad(padded_again, (0, 256 - padded_again.shape[1]), value=1)
padded_attention_masks = (padded_again_ids != tokenizer.pad_token_id).long()
#tokens5 = tokenizer.convert_ids_to_tokens(padded_again_ids[0])
#tokens5 = tokenizer.convert_ids_to_tokens(new_torch[0])
#print(tokens5)
#print(data_1[0]['label'])

#data_1[0]['input_ids']
'''