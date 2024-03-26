# IMPORTS
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm 
from preprocess_user_data import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..\Pretrained')
from variationalmodel import *

# LOAD TRAINED MODEL & SIMILARITY MODEL:
c_encoder_path = "C:\PROJECTS\CrowdsInSentences\CIS_Python\Pretrained\Saved_Models"
model_3.load_state_dict(torch.load(f'{c_encoder_path}\\VAEv1.pth'))
# sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# LOAD DATA:
# text, _ = load_data()
# group1 = text # TODO: text[:10000]
user_data = UserData("descriptions")
# group1 = user_data.get_all_videos()
# input_ids = user_data.tokenize_sentences(group1, max_seq_len)
# # input_ids = preprocess_data(group1, tokenizer)
# [gt_mean, gt_var], w, [embeds_1, embeds] = model_3(input_ids,"Embeddings")
# print(embeds_1.shape, embeds.shape)
# user_data = UserData("descriptions")
# group2 = user_data.get_all_videos()
# print(len(group1),len(group2))

def calculate_similarity(group1, group2):  
  similarities = np.zeros((len(group1),len(group2)))

  for r in tqdm(range(len(group1))):
    for c in range(len(group2)):
      embedding_1= sim_model.encode(group1[r], convert_to_tensor=True)
      embedding_2 = sim_model.encode(group2[c], convert_to_tensor=True)

      similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
      similarities[r,c] = similarity
  return similarities

def normalize_weights(z):
        exp_z = torch.exp(z)
        sum_exp_z = torch.sum(exp_z, dim=1, keepdim=True)
        softmax_probs = exp_z / sum_exp_z
        return softmax_probs

def apply_model(input_ids):
    with torch.no_grad():
            [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
            # w = sample.sample_weight(outputs)
            # z = outputs[0]
            z = normalize_weights(w)
            predictions = torch.argmax(z, dim = 1)
    return z.numpy(), predictions.numpy()

def find_similar_sentences(group1, group2, similarities):
  k = 5
  sorted_indices = np.argsort(similarities.flatten())[::-1]
  topk_indices = np.unravel_index(sorted_indices[:k], similarities.shape)
  indices_list = []
  for value, indices in zip(similarities[topk_indices], zip(*topk_indices)):
      indices_list.append(indices)
      sentence1 = group1[indices[0]]
      sentence2 = group2[indices[1]]
      input_ids = tokenizer(sentence2.strip('""'), return_tensors="pt").input_ids
      inputs = torch.zeros(1,max_seq_len).long()
      inputs[:,:input_ids.shape[1]] = input_ids
      _, label2 = apply_model(inputs)
      print(sentence1, "LABEL:", 0)
      print(sentence2, "LABEL:", label2)
      print('==============')

def calculate_emb_similarty(emb1, emb2, len_emb):
    similarities = np.zeros((emb1.shape[0],len_emb))
    sim = nn.CosineSimilarity(dim = 1)
    for r in tqdm(range(emb1.shape[0])):
      for c in range(len_emb):
        similarity = torch.norm(sim(emb1[r, :, :], emb2[c,:,:]))
        similarities[r,c] = similarity.item()
    return similarities

def find_similar_sentences_emb(group1, similarities):
    k = 5
    ind = np.argmax(similarities)
    print(similarities)
    print(ind)
    exit()
    sorted_indices = np.argsort(similarities.flatten())[::-1]
    topk_indices = np.unravel_index(sorted_indices[:k], similarities.shape)
    indices_list = []
    for value, indices in zip(similarities[topk_indices], zip(*topk_indices)):
        indices_list.append(indices)
        sentence1 = group1[indices[0]]
        print("Sentence: ", sentence1)
        # sentence2 = group2[indices[1]]
        # input_ids = tokenizer(sentence2.strip('""'), return_tensors="pt").input_ids
        # inputs = torch.zeros(1,max_seq_len).long()
        # inputs[:,:input_ids.shape[1]] = input_ids
        # _, label2 = apply_model(inputs)
        # print(sentence1, "LABEL:", 0)
        # print(sentence2, "LABEL:", label2)
        # print('==============')
    
# similarities = calculate_similarity(group1, group2)
# find_similar_sentences(group1, group2, similarities)
# https://huggingface.co/tasks/sentence-similarity

# Input of encoder
sets_arx = user_data.get_image_data()["arx"]
enc_inputs = user_data.tokenize_sentences(["A small crowd walking by a church. All groups are walking in the same direction except one individual.",
                        "A group of people walking in front of a church during a tour.",
                        "Walking near the church"], max_seq_len)
# Candidate decoder inputs
img_si = user_data.get_img_si()
si_dict ={}
si_sent = {}
counter = 0
for env in range(len(img_si)):
    si_list = img_si[env]
    for si in si_list:
        si_dict[counter] = env #0: arx, 1: uni, 2: eth
        counter += 1
        si_sent[counter] = si
dec_input = img_si[0] + img_si[1] + img_si[2]
# dec_input = img_si[1] + img_si[2]
dec_input = user_data.tokenize_sentences(dec_input, max_seq_len)

#Sample latent space 
[gt_mean, gt_var], w, [embeds_gt, embeds_in] = model_3(enc_inputs,"Embeddings")
[gt_mean, gt_var], w, [embeds_cand, embeds] = model_3(dec_input,"Embeddings")

#Find 'decoded' sentence.
pred = []
for i in range(embeds_in.shape[0]):
  similarities = calculate_emb_similarty(embeds_cand, embeds_in[i:(i+1),:,:], 1)
  idx = np.argmax(similarities)
  pred.append(si_dict[idx])
# print(idx,si_dict[idx],si_sent[idx] )
print(len(pred), pred.count(0), pred.count(1), pred.count(2))