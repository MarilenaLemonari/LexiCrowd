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
c_encoder_path = "...\CIS_Python\Pretrained\Saved_Models"
model_3.load_state_dict(torch.load(f'{c_encoder_path}\\VAEv1.pth'))

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
enc_inputs = user_data.tokenize_sentences(["People walking rapidly towards a concert."], max_seq_len)
# enc_inputs = user_data.tokenize_sentences(["People walk towards a concert."], max_seq_len)
# enc_inputs = user_data.tokenize_sentences(["People attending a concert."], max_seq_len)
# enc_inputs = user_data.tokenize_sentences(["People walking in a park."], max_seq_len)
# Candidate decoder inputs
dec_input = user_data.tokenize_sentences(["People are friendly.",
                                          "People are distant.",
                                          "People are aggressive."], max_seq_len)
# dec_input = user_data.tokenize_sentences(["Business professionals walk towards a concert.",
#                                           "Music enthusiasts walk towards a concert.",
#                                           "An angry mob walks towards a concert"], max_seq_len)
# dec_input = user_data.tokenize_sentences(["Students going to class.",
#                                           "The students have been going to class.",
#                                           "The students were heading towards the classrooms.",
#                                           "The classroom would soon be filled with students."], max_seq_len)
# dec_input = user_data.tokenize_sentences(["People standing at the park.",
#                                           "People eating near the park kiosk.",
#                                           "People walking in a mall.",
#                                           "People walking in a market."], max_seq_len)

#Sample latent space 
[gt_mean, gt_var], w, [embeds_gt, embeds_in] = model_3(enc_inputs,"Embeddings")
# [gt_mean, gt_var], w, [embeds_gt, embeds_cand] = model_3(dec_input,"Embeddings")
[gt_mean, gt_var], w, [embeds_cand_gt, embeds_cand] = model_3(dec_input,"Embeddings")
print(w)
#Find 'decoded' sentence.
similarities = calculate_emb_similarty(embeds_cand, embeds_in, 1)
print(similarities)
exit()
find_similar_sentences_emb(group1, similarities)

# print(group1[0])