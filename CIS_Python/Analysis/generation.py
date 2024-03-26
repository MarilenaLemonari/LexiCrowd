# IMPORTS
from preprocess_user_data import *
import torch
import sys
sys.path.append('..\Pretrained')
from variationalmodel import *

# LOAD TRAINED MODEL & SIMILARITY MODEL:
c_encoder_path = "...\CIS_Python\Pretrained\Saved_Models"
model_3.load_state_dict(torch.load(f'{c_encoder_path}\\VAEv1.pth'))

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
text, _ = load_custom_data("...\Data\goalSentencesV2.txt", 0)
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
      
# Controlled generation:
input_arx_vid = "Groups of students are walking into a historical monument's yard in the same direction." 
w,z,p, embeds = generate_W(input_arx_vid,5)