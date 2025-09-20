
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
            if idx<5000:
                obj= json.loads(line)
                list_data.append(obj)
            else:
                break

    return list_data

data= dataset_align(_json_file)

##prepare the datasample and batches
##device ='cuda' if torch.cuda.is_available() else 'cpu'

"""## Dataset class to preprocess on sample , sam;ple of ts_input to patchify based on the window size and stride
class ts_multimodal_text(Dataset):
    def __init__(self,patch_len,stride,data,tokenizer,device=device,model_dtype=model_dtype):
        self.data = data
        self.tokenizer = tokenizer
        self.mode_dtype=model_dtype
        self.device=device
        self.p=patch_len
        self.s=stride

    def __len__(self):
      return len(self.data)

    ## to patchify/sliding window operation of the ts_input
    def padding_stride(self,x,p=256,s=256):
      x=x.view(1,-1)
      ##print(x.shape)
      l=x.shape[1]
      ##print(l)
      r =(l-p)%s
      if (r==0):
        num_windows=(l-p)//s+1
          #print(f'num_windows: {num_windows}')
        pad_width=0
        x_unfolded = x.unfold(1,p,s)
        return x_unfolded  ## (bs,num_windows,patch_len)

      else:
        num_windows=(l-p)//s+2
          #print(f'num_windows: {num_windows}')
        pad_width=s-r
        pattern = torch.tensor([0.0,1.0],device=self.device)
        num_repeats = pad_width // 2
        pad = pattern.repeat(num_repeats).view(1,-1)  ## (1,pad_width)
        x_padded = torch.cat([x,pad],axis=1)
        x_unfolded = x_padded.unfold(1,p,s)
        return x_unfolded


    def parse_extract_ts_boundary(self,prompt):
        tokenized= self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        input_ids= tokenized['input_ids'][0]

        ts_start_token=self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token=self.tokenizer.convert_tokens_to_ids('<ts/>')
        ts_position=[]

        for i,token_id in enumerate(input_ids):
            if (token_id==ts_start_token):
                ts_position.append(('start',i))
            elif (token_id==ts_end_token):
                ts_position.append(('end',i))

        stack =[]
        ts_pairs=[]

        for j in range(len(ts_position)):
            pos,idx = ts_position[j]
            if pos=='start':
                stack.append(idx)
            elif stack and pos=='end':
                start=stack.pop(0)
                ts_pairs.append((start,idx))

        return ts_pairs,input_ids

    def __getitem__(self,idx):
      sample=self.data[idx]
      prompt=sample['input']
      output=sample['output']
      timeseries=sample['timeseries']
      ts_inputs=[]

      output_ids=self.tokenizer(output,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
      ts_pairs,prompt_ids=self.parse_extract_ts_boundary(prompt)
      ts_metrics=len(ts_pairs)

      ## to account for multi-variate timeseries
      for metric in timeseries:
          ts_tensor=torch.tensor(metric,dtype=self.mode_dtype,device=self.device).view(-1,1)
          # Call padding_stride with p and s
          ts_tensor=self.padding_stride(ts_tensor,p=self.p,s=self.s)##patchify operation as preprocessing
          ts_inputs.append(ts_tensor.to(self.device))

      ##print(ts_inputs


      combined_input_ids=torch.cat([prompt_ids,output_ids],dim=0) ## input + output tokens

      return{
          'input_ids':combined_input_ids,
          'output_ids':output_ids,
          'ts_inputs':ts_inputs} ## list of ts_data(tensors) of size (1,N_i,P)


###collate function for the batch
## padding at the ts_input level and the input_ids (textual tokens)
def collate_func(batch):
  input_ids = [x['input_ids'] for x in batch]
  output_ids=[x['output_ids'] for x in batch]
  ts_data=[x['ts_inputs'][0] for x in batch] ##list of univariate ts_tensor of size (batch_size,N_i,P)
  ##print(len(ts_data))

  ts_patches_len=[x['ts_inputs'][0].shape[1] for x in batch] # Accessing shape from the actual tensor for univariate case
  ## setting the max_patch_length for ts_tokens
  max_n_per_batch=10
  padded_ts_data=[]

  ##padding the times series input in the batch
  for i,ts_input_sample in enumerate(ts_data):
    padded_patch_len=max_n_per_batch-ts_input_sample.shape[1]
    patch_len=ts_input_sample.shape[2]
    ts_padding_len= padded_patch_len*patch_len
    pattern = torch.tensor([0.0,1.0]).to(device)
    num_repeats = ts_padding_len // 2
    pad = pattern.repeat(num_repeats)

    ##converting the ts_input = <ts_tokens>+<padded_ts_token>
    padded_ts_token=pad.view(-1,patch_len).unsqueeze(0)
    padded_ts_data.append(torch.cat([ts_input_sample.to(device),padded_ts_token.to(device)],dim=1)) ## the ts_tokens are right padded

  ## N_i of batch of samples after padding
  ts_patch_padded_len=[x.shape[1] for x in padded_ts_data]
  max_text_len=max([x.size(0) for x in input_ids])
  max_ts_len=max(ts_patches_len)
  ts_seq_len = [seq.size(0) for seq in input_ids]
  tot_len=[(x+y) for x,y in zip(ts_patch_padded_len,ts_seq_len)]
  max_len_batch=max(tot_len)

  ##print([(max_text_len-seq.size(0))+(max_ts_len-p_len) for seq,p_len in zip(input_ids,ts_patches_len)])

  ##input_ids_padded= pad_sequence(input_ids,batch_first=True,padding_value=tokenizer.pad_token_id,padding_side='left')
  input_ids_padded= torch.stack([torch.cat([torch.full(((max_len_batch-seq.size(0)),),tokenizer.pad_token_id,dtype=seq.dtype),seq]) for seq in input_ids])

  ##max_len_batch=input_ids_padded.shape[1] # Corrected to use shape[1] for sequence length
  ###max_N_per_batch=max(ts_data[])


  labels_batch=[]
  attention_mask_batch=[]

  for i,sample in enumerate(batch):
    labels = torch.full((max_len_batch,),-100,dtype=torch.long,device=device)
    # Calculate pad_len based on combined_input_ids length and max_len_batch
    # The combined length includes input_ids and ts_inputs length
    combined_len = sample['input_ids'].shape[0] + sample['ts_inputs'][0].shape[1] # Assuming one ts input per sample for simplicity
    pad_len = max_len_batch - combined_len

    seq_len=sample['input_ids'].shape[0]
    output_len=sample['output_ids'].shape[0]

    # Adjust label assignment based on padding at the beginning and TS embeddings
    # The labels correspond to the output_ids, which are at the end of the combined sequence
    # Calculate the starting index for output_ids in the padded label tensor
    output_start_index = max_len_batch - output_len
    labels[output_start_index:] = sample['output_ids']
    labels_batch.append(labels)

    # Adjust attention mask based on padding and TS embeddings
    attention_mask=torch.cat([torch.zeros(pad_len,dtype=torch.long,device=device),
                                torch.ones(seq_len,dtype=torch.long,device=device),
                                torch.ones(sample['ts_inputs'][0].shape[1],dtype=torch.long,device=device)]) # Assuming one ts input
    attention_mask_batch.append(attention_mask)

  ##return the batch of input_ids , labels and timeseries
  return{
      'input_ids':input_ids_padded,
      "labels":torch.stack(labels_batch),
      'attention_mask':torch.stack(attention_mask_batch),
      "time_series":torch.cat(padded_ts_data,dim=0)} ##list of tensor (bs,max_N,Patch_len)"""

## to check the batch of samples returned
dataset=ts_multimodal_text(256,256,data,tokenizer,device=device,model_dtype=model_dtype)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))

"""
for batch in dataloader:
    print(f'padded_input_ids' ,batch['input_ids'].shape)
    print(batch['labels'].shape)
    print(batch['attention_mask'].shape)
    print(f'time_series:{batch['time_series'].shape}')
    ##print(model.get_input_embeddings()(batch['input_ids']).shape)
    break"""
    ##print(batch['outpu""t_ids'][1,:])


##create ts_encoder instance to be passed
##ts_encoder = TS_encoder(config)
##ts_encoder.load_state_dict(torch.load(ts_encoder_cp,map_location=device))
##ts_encoder.to(device)

## nn.Module to wrap the get_input_embeddings_uperating
## calling .get_input_embeddings being part of computational graph
## calling this model should perform single forward pass
## loss.backward should perform gradient calc

## this model will be called inside for each batch iterated over the entire dataset
## the wrapper model expects a 'batch' of input_ids,lables ,raw_timeseries

import torch
import torch.nn as nn

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,max_patches,patch_len,llm_model,device=device):
        super().__init__()
        ##self.ts_encoder=ts_encoder
        self.tokenizer=tokenizer
        ##self.ts_input=ts_input
        self.llm_model=llm_model
        ##self.ts_input=ts_input
        ##self.N=ts_input.shape[1]
        self.max_patches=max_patches
        self.P=patch_len
        self.device=device

        ##initialise the ts_encoder
        self.ts_encoder=PatchTSTEncoder(c_in=1,n_vars=1,num_patch=self.max_patches,patch_len=self.P,n_layers=6,d_model=llm_model.config.hidden_size,n_heads=4,shared_embedding=True,d_ff=2*256,
                 norm='BatchNorm',attn_dropout=0.1,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=False,pe='zeros',learn_pe=True,verbose=True)


        """self.input_ids=input_ids
        self.output_ids=output_ids"""

    ##purpose is to assemble the input_embeddings and has to return the assembled input_embedding tensor
    # return assembled input embedding tensor for the
    def assemble_input_embeds(self,input_ids,ts_embeddings:torch.tensor,attention_mask,labels):

        assemb_embed_list=[]
        attention_assem=[] ##batch,seq_len
        labels_assem=[] ##batch,seq_len
        ##input_ids = self.batch['input_ids'].to(self.device)## batch,seq_len
        ##print(f'input_ids :{input_ids.shape}')
        self.llm_model.to(self.device)
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)  ##(batch,seq_len,emb_dim)
        input_embeds=input_embeds.to(self.device)
        ##print(f'input_embeds : {input_embeds.shape}')
        batch_size=input_embeds.shape[0]
        ##ts_input=self.batch['time_series']

        for i in range(batch_size):
            ##logic to assemble
            ##ts_tensor = ts_input.view(-1,self.max_patches,1,self.P).to(self.device) # Ensure tensor is on the correct device
            ##ts_tensor = ts_tensor.view(-1, 1) ## need the
            ##print(ts_tensor.shape) ## to check device
            ##patched_ts = padding_stride(ts_tensor,p=256,s=256)
            ##patched_ts=ts_tensor.unsqueeze(0).permute(0,2,1,3)
            ##print(patched_ts.shape)
            ##patched_ts = patched_ts.view(1,N,1,patch_len) ## re-arrange as that is required shape of tensor to model
            ##print(patched_ts.shape)
            ##ts_embeddings = self.ts_encoder(ts_tensor)
            ##print(ts_embeddings.shape)
            ts_embeddings=ts_embeddings.squeeze(dim=1) ### for the univariate case (bs,1,N,P)
            ##print(ts_embeddings.shape)
            ##ts_embeddings=ts_embeddings.squeeze(dim=(0,1)) ##(N_i,d_embed)
            ##print(f'ts_embedding_shape : {ts_embeddings.shape}')
            ##ts_embeddings = self.ts_encoder(ts_input[i][0]).to(self.device)
            ts_start_mask = (input_ids[i,:]==self.tokenizer.convert_tokens_to_ids('<ts>')) ##boolean logic
            ts_end_mask = (input_ids[i,:]==self.tokenizer.convert_tokens_to_ids('<ts/>'))
            start_index = torch.where(ts_start_mask)[0]
            end_index = torch.where(ts_end_mask)[0]
            ##print(start_index,end_index)

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
      ##print(f'after ts_encoder : {ts_embedding.shape}')

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

"""for param in ts_encoder.parameters():
    param.requires_grad=True"""

for epoch in range(5):
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
        ##model_wrapper.train()
        ##all_params = (list(model_wrapper.ts_encoder.parameters()) +list(model_wrapper.llm_model.get_input_embeddings().parameters()))
        ##optimizer = torch.optim.AdamW(all_params, lr=1e-5)

        """outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
        labels=labels)"""
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
plt.plot(range(0, 5), epoch_losses, marker='o')
plt.title("Training Loss Trend Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(out_path)