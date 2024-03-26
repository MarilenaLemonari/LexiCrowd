# @misc{https://doi.org/10.48550/arxiv.2210.11416,
#   doi = {10.48550/ARXIV.2210.11416}, 
#   url = {https://arxiv.org/abs/2210.11416},
#   author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
#   keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
#   title = {Scaling Instruction-Finetuned Language Models},
#   publisher = {arXiv},
#   year = {2022},
#   copyright = {Creative Commons Attribution 4.0 International}
# }

# IMPORTS
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
import torch
from data_preprocessing import *

# LOAD PRE-TRAINED MODELS
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
# print(model)

# DEFINE AUTOENCODER
for param in model.parameters():
    param.requires_grad = False

class Net(nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.fc_mean = nn.Linear(1024, 3)
        self.fc_var = nn.Linear(1024, 3)
        self.fc = nn.Linear(max_seq_len * 1024, 1024)
        self.encoder = model.encoder
    def forward(self, x, embeds = None):
        if embeds == None:
          embeddings = self.encoder(input_ids = x)
        else:
          embeddings = self.encoder(inputs_embeds = embeds)
        embeddings = embeddings.last_hidden_state
        x = torch.flatten(embeddings, start_dim = 1, end_dim = -1)
        x = F.relu(self.fc(x))
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        return mean, var, embeddings
        
model_2 = Net(model,max_seq_len)

for name, param in model_2.named_parameters():
  if param.requires_grad == True:
      print(name)

class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def sample_weight(self,inputs):
      z_mean, z_log_var = inputs
      batch = z_mean.size(0)
      dim = z_mean.size(1)
      epsilon = torch.randn(batch, dim, device=z_mean.device)
      return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    def forward(self, inputs, labels):
      batch = inputs[0].shape[0]
      w = torch.zeros(batch, 100, 3)
      labels_ext = torch.zeros(batch,100,1)
      for i in (range(100)):
        w_i = self.sample_weight(inputs) # (batch, 3)
        w[:, i,:] = w_i
        labels_ext[:, i, :] = labels.unsqueeze(1)
      w = w.view(-1, w.shape[-1])
      labels_ext = labels_ext.view(-1, labels_ext.shape[-1]).squeeze(-1).long()
      return w , labels_ext
sample = Sampling()

print("Pretrained Language Model (T5) Loaded.")
print("Autoencoder Defined.")