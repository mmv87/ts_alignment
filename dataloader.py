
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

device ='cuda' if torch.cuda.is_available() else 'cpu'

## Dataset class to preprocess on sample , sam;ple of ts_input to patchify based on the window size and stride
class ts_multimodal_text(Dataset):
    def __init__(self,patch_len,stride,data,tokenizer,device=device,model_dtype=None):
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
        ##x_padded = torch.nn.functional.pad(x,(0,pad_width),mode='constant')
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
          ##print(ts_tensor.shape)
          ts_inputs.append(ts_tensor.to(self.device))

      ##print(ts_inputs

      combined_input_ids=torch.cat([prompt_ids,output_ids],dim=0) ## input + output tokens

      return{
          'input_ids':combined_input_ids,
          'output_ids':output_ids,
          'ts_inputs':ts_inputs} ## list of ts_data(tensors) of size (1,N_i,P)

## padding at the ts_input level and the input_ids,attention_mask,textual tokens)
def collate_func(batch,tokenizer=None,device=device):
    input_ids = [x['input_ids'] for x in batch]
    output_ids=[x['output_ids'] for x in batch]
    ts_data=[x['ts_inputs'][0] for x in batch] 
  
    ts_patches_len=[x['ts_inputs'][0].shape[1] for x in batch] # Accessing shape from the actual tensor for univariate case
  ## setting the max_patch_length for ts_tokens
    max_n_per_batch=10
    padded_ts_data=[]
    labels_batch=[]
    attention_mask_batch=[]

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
    ##print(ts_seq_len)
    tot_len=[(x+y) for x,y in zip(ts_patch_padded_len,ts_seq_len)]
    max_len_batch=max(tot_len)

    if len(batch)==1:
        input_ids_padded=input_ids[0].unsqueeze(0)    ##print(f'textual_shape {input_ids_padded.shape}')

        output_len=output_ids[0].shape[0]
        output_start_index = input_ids[0].shape[0] - output_len

        labels = torch.full((ts_seq_len[0],),-100,dtype=torch.long,device=device)
        labels[-output_len:] = output_ids[0]
        labels_batch.append(labels)
        ##print(labels.shape)
        attention_mask=torch.cat([torch.ones(ts_data[0].shape[1],dtype=torch.long,device=device),torch.ones(ts_seq_len[0],dtype=torch.long,device=device)])
        attention_mask_batch.append(attention_mask)
        ##print(attention_mask.shape)

        return {
            'input_ids':input_ids_padded,
            "labels":torch.stack(labels_batch),
            'attention_mask':torch.stack(attention_mask_batch),
            "time_series":padded_ts_data[0]} ##list of tensor (bs,max_N,Patch_len)}


    else:
        input_ids_padded= torch.stack([torch.cat([torch.full(((max_len_batch-seq.size(0)),),tokenizer.pad_token_id,dtype=seq.dtype),seq]) for seq in input_ids])

  ##max_len_batch=input_ids_padded.shape[1] # Correctepl=d to use shape[1] for sequence length
  ###max_N_per_batch=max(ts_data[])
  
    for i,sample in enumerate(batch):
        labels = torch.full((max_len_batch,),-100,dtype=torch.long,device=device)
        combined_len = sample['input_ids'].shape[0] + sample['ts_inputs'][0].shape[1] # Assuming one ts input per sample for simplicity
        pad_len = max_len_batch - combined_len

        seq_len=sample['input_ids'].shape[0]
        output_len=sample['output_ids'].shape[0]

        # Adjust label assignment based on padding at the beginning and TS embeddings
        # The labels correspond to the output_ids, which are at the end of the combined sequence
        # Calculate the starting index for output_ids in the padded label tensor
        output_start_index = max_len_batch - output_len
        labels[-output_len:] = sample['output_ids']
        labels_batch.append(labels)

        # Adjust attention mask based on padding and TS embeddings
        attention_mask=torch.cat([torch.zeros(pad_len,dtype=torch.long,device=device),torch.ones(sample['ts_inputs'][0].shape[1],dtype=torch.long,device=device),
                                torch.ones(seq_len,dtype=torch.long,device=device)
                                    ]) # Assuming one ts input
        attention_mask_batch.append(attention_mask)

  ##return the batch of input_ids , labels and timeseries
    return{
        'input_ids':input_ids_padded,
        "labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.cat(padded_ts_data,dim=0)} ##list of tensor (bs,max_N,Patch_len)