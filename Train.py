from torch.nn.parallel.distributed import _tree_unflatten_with_rref
from transformers import BartTokenizer, RobertaTokenizer
#from transformers.models.bert.modeling_tf_bert import _CHECKPOINT_FOR_DOC
from custom_datasets import create_dataset, samsum_dataset, get_samsum, concatenate_summary_keyword, pseudoDataset
import evaluate
from accelerate.test_utils.testing import get_backend
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch
from ganmodel import BART_base_model, Roberta_discriminator, classification_loss, RoBERTa_base_model
import numpy as np
import torch.nn.functional as F
import os
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(n_epochs, minibatch_sizes, is_save, is_load, load_pathG, load_pathD, seed, BART_only=False):

########################################## load the tokenizer and model ##################################################
    # load model ckpt from huggingface and use it to tokenizer
    threshold = 0.75
    fine_tuning = True
    BART_model_ckpt = 'facebook/bart-base'
    RoBERTa_model_ckpt = "roberta-base"
    BA_tokenizer = BartTokenizer.from_pretrained(BART_model_ckpt)
    RBE_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_model_ckpt)

      # if there is no model create a model using pretrain model from huggingface
    BaseG_model = BART_base_model(BART_model_ckpt)
    NetG = BaseG_model
    BaseD_model = RoBERTa_base_model(RoBERTa_model_ckpt)
    NetD = BaseD_model
    if is_load:
      print("loadding from ckpt")
      ckptG = "/content/drive/MyDrive/semi_gan_bart/SaveModel/lora_bartGAN_G_epoch1_8.pt"
      ckptG = torch.load(ckptG, map_location="cuda")
      NetG.load_state_dict(ckptG['model_state'])
      ckptD = "/content/drive/MyDrive/semi_gan_bart/SaveModel/lora_bartGAN_D_epoch1_8.pt"
      ckptD = torch.load(ckptD, map_location="cuda")
      NetD.load_state_dict(ckptD['model_state'])

    device, _, _ = get_backend() # make sure the device is in gpu
    NetG.to(device)
    NetD.to(device)
    t_dataset, v_dataset, test_dataset, u_t_dataset = get_samsum() #get_samsum() # load datasets
    train_dataloader = DataLoader(t_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    eval_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    rouge = evaluate.load("rouge")

    optimizerG = AdamW(NetG.parameters(), lr=3e-5) # set up optimizer for Generator
    optimizerD = AdamW(NetD.parameters(), lr=2e-5) # set up optimizer for Discrimnator

    num_epochs = n_epochs # training epochs

    # set up learning schedualer for both discriminator and generator
    num_training_steps = num_epochs * len(train_dataloader)
    num_warm_up = 100
    #lr_schedulerG = get_scheduler(
        #name="linear", optimizer=optimizerG, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps
    #)
    #lr_schedulerD = get_scheduler(
        #name="linear", optimizer=optimizerD, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps
    #)
    if is_load:
      optimizerD.load_state_dict(ckptD['optimizer_state'])
      optimizerG.load_state_dict(ckptG['optimizer_state'])
      #lr_schedulerD.load_state_dict(ckptD['lr_scheduler'])
      #lr_schedulerG.load_state_dict(ckptG['lr_scheduler'])
    if fine_tuning:
      fine_tune_model(NetG, NetD, device, 2, minibatch_sizes, train_dataloader, eval_dataloader, optimizerG, optimizerD, RBE_tokenizer, BA_tokenizer, is_save, rouge)
    small_train_dataloader = DataLoader(t_dataset, shuffle=False, batch_size=int(minibatch_sizes/2), worker_init_fn=lambda worker_id: np.random.seed(seed))
    for epoch in range(num_epochs):
      pseudo_dataloader = generate_pseudo_label(NetD, NetG, RBE_tokenizer, u_t_dataset, minibatch_sizes, threshold, seed)
      combine_dataset_training(epoch, small_train_dataloader, pseudo_dataloader, eval_dataloader, minibatch_sizes, NetG, NetD, optimizerD, optimizerG, RBE_tokenizer, BA_tokenizer, is_save, rouge)



def fine_tune_model(NetG, NetD, device, num_epochs, minibatch_sizes, train_dataloader, eval_dataloader, optimizerG, optimizerD, RBE_tokenizer, BA_tokenizer, is_save, rouge):
    print("\n=============================================fine tuning==================================")
    num_training_steps = num_epochs * len(train_dataloader)
    print(f"\nNum_Epochs:{num_epochs}, Batch_size:{minibatch_sizes}")

    progress_bar = tqdm(range(num_training_steps))


    epochs = 0
    #loss_record = []
    #Rouge_record = []
    #max_alpha = 1
    batches = 0
    dis_only = False
    bart_loss = 0
    A_t = []
    A_f = []
    for epoch in range(num_epochs):
        #num_valid_steps = len(eval_dataloader)
        #progress_bar2 = tqdm(range(num_valid_steps))
        batches = 0
        NetD.train()
        NetG.train()
        for batch in train_dataloader:
            current_size = batch['input_ids'].shape[0]
            real = 1.0
            fake  = 0.0
            reallabel  = torch.full((current_size,), real).unsqueeze(1).to(device) 
            fakelabel  = torch.full((current_size,), fake).unsqueeze(1).to(device)

            input_ids = batch['input_ids'].to(device)
            attention_mask =batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            discrimnator_input_id = batch['bert_input_id'].to(device)
            discrimnator_mask =batch['bert_mask'].to(device)

            batch_knoun = batch['bert_noun'].to(device)
            batch_knoun_mask = batch['mask_noun'].to(device)

            optimizerD.zero_grad()

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

            padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1)
            roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()

            roberta_finput_ids, roberta_fattention_masks = concatenate_summary_keyword(padded_input_ids, roberta_attention_masks,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)
            roberta_tinput_ids, roberta_tattention_masks = concatenate_summary_keyword(discrimnator_input_id, discrimnator_mask,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)


            attention_mask_hidden = torch.ones(minibatch_sizes, 256).long()
            flogits = NetD(roberta_finput_ids, attention_mask=roberta_fattention_masks)
            tlogits = NetD(roberta_tinput_ids, attention_mask=roberta_tattention_masks)

            loss1 = classification_loss(tlogits.logits, reallabel)
            loss2 = classification_loss(flogits.logits, fakelabel)

            preds = (torch.sigmoid(tlogits.logits.detach()) >= 0.5).long()
            accuracyT = (preds == labels).float().mean()
            A_t.append(accuracyT)
            preds = (torch.sigmoid(flogits.logits.detach()) <= 0.5).long()
            accuracyF = (preds == labels).float().mean()
            A_f.append(accuracyF)
            alpha = 0.5
            Dloss = alpha*loss1 + (1-alpha)*loss2
            Dloss.backward()
            #clip_grad_norm_(NetD.parameters(), max_norm=1.0)
            optimizerD.step()
            #lr_schedulerD.step()

            #for p in NetG.parameters():
                #p.requires_grad = True
            if epochs > 0:
                for p in NetD.parameters():
                    p.requires_grad = False

                optimizerG.zero_grad()

                output_g = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
                bart_loss = output_g.loss

                Gloss = bart_loss
                Gloss.backward()
                #clip_grad_norm_(NetG.parameters(), max_norm=1.0)
                optimizerG.step()
                #lr_schedulerG.step()
                
                for p in NetD.parameters():
                    p.requires_grad = True
            if batches % 20 == 0:
              print("\nEpoch:{: <5}| Batch:{: <5}| Gtrain_loss:{: <5.4f}| Dtrain_loss:{: <5.4f}|{: <5.4f}".format(epochs, batches, bart_loss , loss1, loss2))
              print("\n discrimnator prediction:{: .2%}|{: .2%}".format(sum(A_t)/len(A_t), sum(A_f)/len(A_f)))
              A_t = []
              A_f = []
            progress_bar.update(1)
            batches += 1
        print(f"\n============================Start Validation for fine tune Epoch: {epochs}================================")
        NetG.eval()
        pred_list = []
        ref_list = []
        loss_list = []

        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = NetG(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                gen_tokens = NetG.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )

            loss_list.append(outputs.loss.item())

            # decode
            for i in range(len(gen_tokens)):
                pred_list.append(
                    BA_tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
                )
                ref_list.append(
                    BA_tokenizer.decode(labels[i], skip_special_tokens=True)
                )
            #progress_bar2.update(1)
        # compute ROUGE once
        r_score = rouge.compute(predictions=pred_list, references=ref_list)
        average_loss = sum(loss_list) / len(loss_list)

        print("\nEpoch:{: <5}| validation_loss:{: <5.4f}".format(epochs, average_loss))
        print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']))
        #rouge_list = [r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']]
        #Rouge_record.append(rouge_list)
        print(f"\n=============================End Validation for fine tune Epoch: {epochs}==================================")

        epochs += 1
        print(f"\n==============================saving model for fine tune: {epochs}==================================")
        G_path = f"./SaveModel/lora_bartGAN_G_epoch{epoch}_{minibatch_sizes}_fine_tune.pt"
        D_path = f"./SaveModel/lora_bartGAN_D_epoch{epoch}_{minibatch_sizes}_fine_tune.pt"
        if is_save:
          torch.save({
              "model_state": NetG.state_dict(),
              "optimizer_state": optimizerG.state_dict(),
              #"lr_scheduler": lr_schedulerG.state_dict(),
              "epoch": epoch
          }, G_path)

          torch.save({
              "model_state": NetD.state_dict(),
              "optimizer_state": optimizerD.state_dict(),
              #"lr_scheduler": lr_schedulerD.state_dict(),
              "epoch": epoch
          }, D_path)
    print("\n=============================================end fine tuning==================================")


def generate_pseudo_label(NetD, NetG, RBE_tokenizer, u_t_dataset, minibatch_sizes, threshold, seed):
    unlabeled_dataloader = DataLoader(u_t_dataset, shuffle=False, batch_size=30, worker_init_fn=lambda worker_id: np.random.seed(seed))
    num_label_steps = len(unlabeled_dataloader)
    progress_bar = tqdm(range(num_label_steps))
    for param in NetG.parameters():
        param.requires_grad = False
    for param in NetD.parameters():
        param.requires_grad = False
    NetD.eval()
    NetG.eval()
    psudo_ids = []
    psudo_label = []
    psudo_mask = []
    summary_score = []
    breaks = 0
    num_pass = 0
    test_flag = 0
    print("\n=============================================generate pseudo label==================================")
    for batch in unlabeled_dataloader:
        #print(len(psudo_ids))
        passed = 0
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_knoun = batch['bert_noun'].to(device)
        batch_knoun_mask = batch['mask_noun'].to(device)


        genrated = NetG.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                do_sample=False, 
                num_beams=4,
                early_stopping=True
                )   
        #s_token_id = torch.tensor(0, device=device)
        genrated = genrated[:, 1:]

        genrated = genrated.detach()
        padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1).detach()
        roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()
        roberta_finput_ids, roberta_fattention_masks = concatenate_summary_keyword(padded_input_ids, roberta_attention_masks,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)
        flogits = NetD(roberta_finput_ids, attention_mask=roberta_fattention_masks)
        pred = torch.sigmoid(flogits.logits.detach())
        #print(pred.shape)
        #threshold = 0.7
        mask = pred > threshold
        passed = mask.sum().item()
        mask = mask.squeeze(-1)
        num_pass += passed
        #print(mask.shape)
        padded_output_ids = F.pad(genrated, (0, 512 - genrated.shape[1]), value=1).detach()
        #print(padded_output_ids.shape)
        #label_list = padded_output_ids[mask].tolist()
        score_list = [seq for seq in pred[mask]]
        label_list = [seq for seq in padded_output_ids[mask]]
        mask_list = [seq for seq in attention_mask[mask]]
        ids_list = [seq for seq in input_ids[mask]]
        psudo_label.extend(label_list)
        psudo_mask.extend(mask_list)
        psudo_ids.extend(ids_list)
        summary_score.extend(score_list)
        progress_bar.update(1)
        #test_flag += 1
        #if test_flag == 80:
          #break
    print("\n========================================end generate pseudo label==================================")
    print("number of labels generated: ", len(psudo_ids))
    combined = list(zip(summary_score, psudo_label, psudo_ids, psudo_mask))
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    if len(combined_sorted) > 800:
        combined_sorted = combined_sorted[:800]
    random.shuffle(combined_sorted)
    summary_score, psudo_label, psudo_ids, psudo_mask = zip(*combined_sorted)
    summary_score = list(summary_score)
    psudo_label = list(psudo_label)
    psudo_ids = list(psudo_ids)
    psudo_mask = list(psudo_mask)
    pseudo_data = pseudoDataset(psudo_label,psudo_mask,psudo_ids)
    p_dataset = DataLoader(pseudo_data, shuffle=False, batch_size=int(minibatch_sizes/2), worker_init_fn=lambda worker_id: np.random.seed(seed))
    for param in NetG.parameters():
        param.requires_grad = True
    for param in NetD.parameters():
        param.requires_grad = True
    return p_dataset


def combine_dataset_training(epochs, train_dataloader, pseudo_dataloader, eval_dataloader, minibatch_sizes, NetG, NetD, optimizerD, optimizerG, RBE_tokenizer, BA_tokenizer, is_save, rouge):
    print("\n=============================================start training==================================")
    num_training_steps = len(pseudo_dataloader)
    print(f"\nNum_Epochs:{epochs}, Batch_size:{minibatch_sizes}")

    progress_bar = tqdm(range(num_training_steps))
    #num_valid_steps = len(eval_dataloader)
    #progress_bar2 = tqdm(range(num_valid_steps))
    A_t = []
    A_f = []
    batches = 0
    NetD.train()
    NetG.train()
    for batch, psed_batch in zip(train_dataloader, pseudo_dataloader):
        current_size = batch['input_ids'].shape[0]
        real = 1.0
        fake  = 0.0
        reallabel  = torch.full((current_size,), real).unsqueeze(1).to(device) 
        fakelabel  = torch.full((current_size,), fake).unsqueeze(1).to(device)

        input_ids = batch['input_ids'].to(device)
        attention_mask =batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        discrimnator_input_id = batch['bert_input_id'].to(device)
        discrimnator_mask =batch['bert_mask'].to(device)

        batch_knoun = batch['bert_noun'].to(device)
        batch_knoun_mask = batch['mask_noun'].to(device)

        pseudo_ids = psed_batch['input_ids'].to(device)
        pseudo_attention_mask = psed_batch['attention_mask'].to(device)
        pseudo_labels = psed_batch['labels'].to(device)

        optimizerD.zero_grad()

        genrated = NetG.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=False, 
            num_beams=4,
            early_stopping=True
            )   
        #s_token_id = torch.tensor(0, device=device)
        genrated = genrated[:, 1:]

        genrated = genrated.detach()

        padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1)
        roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()

        roberta_finput_ids, roberta_fattention_masks = concatenate_summary_keyword(padded_input_ids, roberta_attention_masks,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)
        roberta_tinput_ids, roberta_tattention_masks = concatenate_summary_keyword(discrimnator_input_id, discrimnator_mask,batch_knoun,batch_knoun_mask,device,RBE_tokenizer)


        #attention_mask_hidden = torch.ones(int(minibatch_sizes/2), 256).long()
        flogits = NetD(roberta_finput_ids, attention_mask=roberta_fattention_masks)
        tlogits = NetD(roberta_tinput_ids, attention_mask=roberta_tattention_masks)

        loss1 = classification_loss(tlogits.logits, reallabel)
        loss2 = classification_loss(flogits.logits, fakelabel)

        preds = (torch.sigmoid(tlogits.logits.detach()) >= 0.5).long()
        accuracyT = (preds == labels).float().mean()
        A_t.append(accuracyT)
        preds = (torch.sigmoid(flogits.logits.detach()) <= 0.5).long()
        accuracyF = (preds == labels).float().mean()
        A_f.append(accuracyF)
        alpha = 0.5
        Dloss = alpha*loss1 + (1-alpha)*loss2
        Dloss.backward()
        #clip_grad_norm_(NetD.parameters(), max_norm=1.0)
        optimizerD.step()
        #lr_schedulerD.step()

            #for p in NetG.parameters():
                #p.requires_grad = True
        for p in NetD.parameters():
            p.requires_grad = False

        optimizerG.zero_grad()

        output1_g = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        supervised_loss = output1_g.loss
        output2_g = NetG(input_ids=pseudo_ids, attention_mask=pseudo_attention_mask, labels = pseudo_labels)
        unsupervised_loss = output2_g.loss
        Gloss = supervised_loss + 0.2 * unsupervised_loss
        Gloss.backward()
        #clip_grad_norm_(NetG.parameters(), max_norm=1.0)
        optimizerG.step()
        #lr_schedulerG.step()
        for p in NetD.parameters():
            p.requires_grad = True
        if batches % 20 == 0:
          print("\nEpoch:{: <5}| Batch:{: <5}| Gtrain_loss:{: <5.4f}| Dtrain_loss:{: <5.4f}|{: <5.4f}".format(epochs, batches, Gloss.item() , loss1, loss2))
          print("\n discrimnator prediction:{: .2%}|{: .2%}".format(sum(A_t)/len(A_t), sum(A_f)/len(A_f)))
          A_t = []
          A_f = []
        progress_bar.update(1)
        batches += 1
    print(f"\n======================================Start Validation for Epoch: {epochs}==================================")
    NetG.eval()

    pred_list = []
    ref_list = []
    loss_list = []

    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = NetG(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            gen_tokens = NetG.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )

        loss_list.append(outputs.loss.item())

            # decode
        for i in range(len(gen_tokens)):
            pred_list.append(
                    BA_tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
                )
            ref_list.append(
                    BA_tokenizer.decode(labels[i], skip_special_tokens=True)
                )
        #progress_bar2.update(1)
    # compute ROUGE once
    r_score = rouge.compute(predictions=pred_list, references=ref_list)
    average_loss = sum(loss_list) / len(loss_list)

    print("\nEpoch:{: <5}| validation_loss:{: <5.4f}".format(epochs, average_loss))
    print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']))
    print(f"\n======================================End Validation for Epoch: {epochs}==================================")

    print(f"\n======================================saving model for : {epochs}==================================")
    G_path = f"./SaveModel/lora_bartGAN_G_epoch{epochs}_{minibatch_sizes}_pesudo.pt"
    D_path = f"./SaveModel/lora_bartGAN_D_epoch{epochs}_{minibatch_sizes}_pesudo.pt"
    if is_save:
      torch.save({
          "model_state": NetG.state_dict(),
          "optimizer_state": optimizerG.state_dict(),
          #"lr_scheduler": lr_schedulerG.state_dict(),
          "epoch": epochs
      }, G_path)

      torch.save({
          "model_state": NetD.state_dict(),
          "optimizer_state": optimizerD.state_dict(),
          #"lr_scheduler": lr_schedulerD.state_dict(),
          "epoch": epochs
      }, D_path)
    print("\n=============================================end training==================================")









def model_predict(input_texts_file, pathG):
  model_ckpt = 'facebook/bart-large-cnn'
  tokenizer = BartTokenizer.from_pretrained(model_ckpt)
  model = torch.load(pathG, weights_only=False)

  model.eval()
  file_path = input_texts_file

  # Open the file in read mode and read the entire content
  with open(file_path, 'r') as file:
      content = file.read()

  print(f"Lecture:\n{content}")
  input_ids = tokenizer(content, truncation=True, padding='max_length', max_length=512, return_tensors= "pt")
  input_ids.to(device)
  output_ids = model.generate(**input_ids, max_length = 256)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  print(f"Summary:\n{output_text}")
  base_file_name = os.path.basename(input_texts_file)
  with open(f"./Summary/Summary_of_{base_file_name}", "w") as file:
    # Write the string to the file
    file.write(output_text)

  print(f"Summary of {base_file_name} created successfully!")



