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

def load_custom_data(path, label):
    sentences = []
    with open(path, 'r') as file:
        sentences = file.readlines()
    for sentence in sentences[:100]:
        sentences.append(sentence.strip('""'))  
    
    labels = torch.ones(len(sentences)) * label
    return sentences, labels

# LOAD DATA:
# text, _ = load_data()
text, _ = load_custom_data("C:\PROJECTS\CrowdsInSentences\Data\goalSentencesV2.txt", 0)
inputs = preprocess_data(text, tokenizer)

def normalize_weights(z):
        exp_z = torch.exp(z)
        sum_exp_z = torch.sum(exp_z, dim=1, keepdim=True)
        softmax_probs = exp_z / sum_exp_z
        return softmax_probs

def apply_model(input_ids, points = None):
    with torch.no_grad():
            [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
            z = normalize_weights(w)
            predictions = torch.argmax(z, dim = 1)
            print("=========================")
            print("Sampled point: ", w)
            print("Normalised weight: ", z)
            print("Predicted labels: ", predictions)
            print("Distribution: ", gt_mean, normalize_weights(gt_mean))
            # w = sample.sample_weight(outputs)
            weights = []
            norm_weights = []
            if points != None:
                  for p in range(points):
                    w = sample.sample_weight((gt_mean, gt_var))
                    z = normalize_weights(w)
                    weights.append(w)
                    norm_weights.append(z)
                    print(f"w_{p}", w)
                    print(f"z_{p}", z)
            mean_weights = torch.mean(torch.stack(weights), dim=0)
            mean_z = torch.mean(torch.stack(norm_weights), dim=0)
            print("Mean w: ", mean_weights)
            print("Mean z: ", mean_z)
    return z.numpy(), predictions.numpy(), w.numpy(), embeds

def generate_W(custom_sentence, points):
      inputs = preprocess_data([custom_sentence], tokenizer)
      z , p, w , embeds = apply_model(inputs, points)
      return w, z, p, embeds
      

# SANITY CHECK: 
# n = 0
# print(len(text))
# for i in range(int(100/100)):
#   z , p, w = apply_model(inputs[100*i: 100*(i+1)]) # Goal Sanity
#   # print(z) # - Sanity check
#   # print(np.min(p),np.max(p))
#   n += len(p[p == 0])
# print(n)
      
# Controlled generation:
input_sentence_goal = "Individuals rush through a road."
input_sentence_group = "Students talking during a break in a school yard." 
input_sentence_interaction = "Park visitors enjoying some snacks at the kiosk."
# 
input_arx_vid = "Groups of students are walking into a historical monument's yard in the same direction." # keep for video
input_arx = "High school students walking in a Greek orthodox church yard."
input_zara = "People walking in the street by a shop window, with bags. Some show interest in the shop's contents."
input_uni_vid = "A crowd of college students were sitting, walking on a wide street. A small group of people were standing and talking in the middle of the street."
input_uni = "Top down video of many groups of people of 2 to 6 people walking, talking and sitting, in a wide pedestrian street."
# input_uni = "People just walking around at a uni campus."
input_eth = "People waiting and walking down a tramway or train station."
input_teaser = "People walking on a sidewalk in front of a fashion stroe."
w,z,p, embeds = generate_W(input_arx_vid,5)


# 
# input_ids = tokenizer(input_sentence_interaction, return_tensors="pt").input_ids
# a = model.encoder(input_ids)
# a.last_hidden_state = embeds
# decoder_ids = tokenizer("Translate to English: Visitors of a museum admire the exhibits.", return_tensors="pt").input_ids
# outputs = model.generate(decoder_input_ids= decoder_ids, encoder_outputs= a)
# print(tokenizer.decode(outputs[0]))

