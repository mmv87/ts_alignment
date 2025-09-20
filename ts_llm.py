
###!pip install transformers

import os
import sys
import numpy as np

from TS_encoder import PatchTSTEncoder
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import ts_multimodal_text,collate_func
from torch.utils.data import Dataset,DataLoader

##_json_file = os.path.join(os.environ["SLURM_TMPDIR"], "ift.json")

model_name='microsoft/Phi-4-mini-reasoning'
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
tokenizer =AutoTokenizer.from_pretrained(model_name,local_files_only=True)

device ='cuda' if torch.cuda.is_available() else 'cpu'
model_dtype=next(model.parameters()).dtype

## to expand the tokenizer to add the special tokens <ts> <ts/>
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))

##dataset fetching
import json
_json_file = os.path.join(os.environ["SLURM_TMPDIR"], "ift.jsonl")

###_json_file='./ift.jsonl'
def dataset_align(file):
    list_data=[]
    with open(file,'r',encoding='utf-8') as file:
        for idx,line in enumerate(file):
            if idx<5500:
                obj= json.loads(line)
                list_data.append(obj)
            else:
                break

    return list_data

data= dataset_align(_json_file)

## to check the batch of samples returned
dataset=ts_multimodal_text(256,256,data,tokenizer,device=device,model_dtype=model_dtype)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))


import torch
import torch.nn as nn

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,max_patches,patch_len,llm_model,device=device):
      super().__init__()
      self.tokenizer=tokenizer
      self.llm_model=llm_model
      self.max_patches=max_patches
      self.P=patch_len
      self.device=device

      ##initialise the ts_encoder
      self.ts_encoder=PatchTSTEncoder(c_in=1,n_vars=1,num_patch=self.max_patches,patch_len=self.P,n_layers=6,d_model=llm_model.config.hidden_size,n_heads=4,shared_embedding=False,d_ff=2*256,
                norm='BatchNorm',attn_dropout=0.1,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=True)

    ##purpose is to assemble the input_embeddings and has to return the assembled input_embedding tensor
    def assemble_input_embeds(self,input_ids,ts_embeddings:torch.tensor,attention_mask,labels):

        assemb_embed_list=[]
        attention_assem=[] ##batch,seq_len
        labels_assem=[] ##batch,seq_len
        
        self.llm_model.to(self.device)
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)  ##(batch,seq_len,emb_dim)
        input_embeds=input_embeds.to(self.device)
        ##print(f'input_embeds : {input_embeds.shape}')
        batch_size=input_embeds.shape[0]
        ##ts_input=self.batch['time_series']

        for i in range(batch_size):
          
            ts_embeddings=ts_embeddings.squeeze(dim=1) ### for the univariate case (bs,1,N,P)
            
            ts_start_mask = (input_ids[i,:]==self.tokenizer.convert_tokens_to_ids('<ts>')) ##boolean logic
            ts_end_mask = (input_ids[i,:]==self.tokenizer.convert_tokens_to_ids('<ts/>'))
            start_index = torch.where(ts_start_mask)[0]
            end_index = torch.where(ts_end_mask)[0]

            assem_embed = torch.cat([input_embeds[i,:start_index.item()+1,:],ts_embeddings[i,:,:],input_embeds[i,end_index.item():,:]],dim=0)
            assem_attention_mask = torch.cat([attention_mask[i,:start_index.item()+1],torch.ones((ts_embeddings.shape[1],),dtype=torch.long,device=self.device),attention_mask[i,end_index.item():]])
            assem_labels = torch.cat([labels[i,:start_index.item()+1],torch.full((ts_embeddings.shape[1],),-100.0,dtype=torch.long,device=self.device),labels[i,end_index.item():]])

            assemb_embed_list.append(assem_embed)
            attention_assem.append(assem_attention_mask.to(self.device))
            labels_assem.append(assem_labels.to(self.device))

        return torch.stack(assemb_embed_list),torch.stack(attention_assem,dim=0),torch.stack(labels_assem,dim=0)


    def forward(self,input_ids=None,ts_input=None,attention_mask=None,labels=None,):
      ##convert the ts_pathces into ts_embeddings
      ts_tensor = ts_input.view(-1,self.max_patches,1,self.P).to(self.device)  ## (bs,N,n_var,P)
      ts_embedding = self.ts_encoder(ts_tensor) ## (bs,n_vars,num_patch,d_model)

      input_embeddings,attentionmask_batch,lable_batch = self.assemble_input_embeds(input_ids,ts_embedding,attention_mask,labels)
        ##input_embeddings=input_embeddings.squeeze(0)

      attention_mask = attention_mask.to(self.device)
      labels = labels.to(self.device)

      return self.llm_model(inputs_embeds=input_embeddings,attention_mask=attentionmask_batch,labels=lable_batch)

##instantiate the llm_wrapper with max_ts_patches and patch_length
model_wrapper=LLM_wrapper(tokenizer,10,256,model,device=device)
model_wrapper.to(device)

##optimizer
all_params = (list(model_wrapper.ts_encoder.parameters()) +list(model_wrapper.llm_model.get_input_embeddings().parameters()))
optimizer = torch.optim.AdamW(all_params, lr=1e-5)

from tqdm import tqdm

epoch_losses=[]
##model.train()
##ts_encoder.train()
model_wrapper.train()
for p in model_wrapper.llm_model.parameters():
  p.requires_grad=False
# Unfreeze only input embedding layer
for p in model_wrapper.llm_model.get_input_embeddings().parameters():
  p.requires_grad = True

for p in model_wrapper.ts_encoder.parameters():
  p.requires_grad=True


for epoch in range(3):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    running_loss=0
    epoch_loss=0
    for batch in pbar:
      input_ids=batch['input_ids'].to(device) ## input and output
      ts_input=batch['time_series'] ### batch of patchified ts_inputs (bs,n,p)
      ##ts_input
      attention_mask_batch =batch['attention_mask'].to(device)  ## the causal attention mask for the input labels
      labels_batch=batch['labels'].to(device)  ## the output text , natural language description of the sample

      ##model_wrapper=LLM_wrapper(tokenizer,ts_input,model,device=device)
      outputs = model_wrapper(input_ids=input_ids,ts_input=ts_input,attention_mask=attention_mask_batch,labels=labels_batch)
      loss=outputs.loss
      loss.backward()  ##gradient calculation
      ###track_gradients(ts_encoder)
      running_loss+=loss.item()
      num_batches+=1
      """
      for name, param in ts_encoder.named_parameters():
          print(name, param.grad is not None)"""
      ##to track the gradients of the TS-Encoder
      optimizer.step()
      optimizer.zero_grad()

      pbar.set_postfix(loss=loss.item())

    epoch_loss=running_loss/num_batches
    epoch_losses.append(epoch_loss)

### save the plot
out_path = os.path.join(os.environ["SLURM_TMPDIR"], "training_loss.png")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(8, 5))
plt.plot(range(0, 3), epoch_losses, marker='o')
plt.title("Training Loss Trend Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(out_path)