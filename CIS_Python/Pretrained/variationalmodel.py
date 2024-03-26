# IMPORTS
from autoencoder import *
from data_preprocessing import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD TRAINED AUTOENCODER AND FREEZE WEIGHTS
model_2.load_state_dict(torch.load('C:\\PROJECTS\\CrowdsInSentences\\CIS_Python\\Pretrained\\Saved_Models\\autoencoderv4.pth')) # TODO
# for param in model_2.parameters():
#     param.requires_grad = False
print("Trained Autoencoder Loaded.")

vocab_size = tokenizer.vocab_size

class VAE_Net(nn.Module):
    def __init__(self, model_2, max_seq_len):
        super().__init__()
        self.pretrained_encoder = model_2
        self.decoder_emb = nn.Sequential(
            nn.Linear(3, 1024), # latent_dim 
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024 * max_seq_len) # hidden_dim
            )
        # self.3decoder_distr = nn.Sequential(
        #     nn.Linear(max_seq_len * 1024, 1024), # latent_dim
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2)
        #     )
        # self.decoder_mean = nn.Linear(512, 3)
        # self.decoder_var = nn.Linear(512, 3)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z
        
    def encode(self, x):
      gt_mean, gt_var, embeds_gt = self.pretrained_encoder(x)
      return gt_mean, gt_var, embeds_gt

    def decoder(self, w):
        x = self.decoder_emb(w)
        embeds = torch.reshape(x, (x.shape[0], max_seq_len, 1024))
        return embeds

    def forward(self, x, mode):
        init_shape = x.shape
        gt_mean, gt_var, embeds_gt = self.pretrained_encoder(x)
        # embeds_gt = self.pretrained_encoder.encoder(x).last_hidden_state
        x = torch.flatten(embeds_gt, start_dim = 1, end_dim = -1)
        # x = self.decoder_distr(x)
        # gt_mean = self.decoder_mean(x)
        # gt_var = self.decoder_var(x)
        gt = (gt_mean, gt_var)
        w = sample.sample_weight(gt) 
        x = self.decoder_emb(w)
        embeds = torch.reshape(x, (x.shape[0], max_seq_len, 1024))
        if mode == "Embeddings":
          # mean, var = self.new_encoder(torch.zeros(init_shape),embeds = embeds)
          return [gt_mean, gt_var], w, [embeds_gt, embeds]

        elif mode == "Ids":
          x = self.decoder_distr(x)
          mean = self.decoder_mean(x)
          var = self.decoder_var(x)
          return [gt_mean, gt_var], [mean, var] ,w, [embeds_gt, embeds]

        else:
          print("Wrong mode.")
          return "Wrong mode."

model_3 = VAE_Net(model_2,max_seq_len).to(device)
# [gt_mean, gt_var], [mean, var], w = model_3(inputs,"Embeddings")
# [gt_mean, gt_var], [mean, var], w, s = model_3(inputs,"Ids")

for name, param in model_3.named_parameters():
  if param.requires_grad == True:
      print(name)

print("Variational Model Defined.")