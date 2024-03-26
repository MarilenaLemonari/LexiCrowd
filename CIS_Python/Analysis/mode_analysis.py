# IMPORTS:
from preprocess_user_data import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..\Pretrained')
from variationalmodel import *
from tqdm import tqdm

# HELPER FUNCTIONS
def normalize_weights(z):
        exp_z = torch.exp(z)
        sum_exp_z = torch.sum(exp_z, dim=1, keepdim=True)
        softmax_probs = exp_z / sum_exp_z
        return softmax_probs

# LOAD TRAINED MODEL:
c_encoder_path = "...\CIS_Python\Pretrained\Saved_Models"
model_3.load_state_dict(torch.load(f'{c_encoder_path}\\VAEv1.pth'))

# LOAD DATA:
user_data = UserData("descriptions")
img_dict = user_data.get_image_data()
vid_dict = user_data.get_video_data()

arx_image = img_dict["arx"] 
arx_video_0 = vid_dict["arx"][0]
arx_video_1 = vid_dict["arx"][1] 
responses = [arx_image, arx_video_0, arx_video_1]
total = 3

# APPLY MODEL:
def apply_model(input_ids, idx = 0):
    with torch.no_grad():
            [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
            # w = sample.sample_weight(outputs)
            # z = outputs[0]
            z = normalize_weights(w)
            predictions = torch.argmax(z, dim = 1)
    group_id = [idx] * z.shape[0]
    return z.numpy(), predictions.numpy(), group_id

# ANALYSE:
model_runs = 50
count_dict = {}
for run in tqdm(range(model_runs)):

    for idx in range(total):
        tokens = user_data.tokenize_sentences(responses[idx], 47)
        input_ids = tokens
        z_mean, prediction, group_id = apply_model(input_ids, idx)
        count_goal = len(prediction[prediction == 0])
        count_group = len(prediction[prediction == 1])
        count_inter = len(prediction[prediction == 2])

        total_count = len(prediction)
        count_goal *= (1/total_count)*100
        count_group *= (1/total_count)*100
        count_inter *= (1/total_count)*100

        if run == 0:
            count_dict[f'{idx}_0'] = [count_goal]
            count_dict[f'{idx}_1'] = [count_group]
            count_dict[f'{idx}_2'] = [count_inter]
        else:
            count_dict[f'{idx}_0'].append(count_goal)
            count_dict[f'{idx}_1'].append(count_group)
            count_dict[f'{idx}_2'].append(count_inter)

perc_dict = {}
dataset = "Church"
names = ["image", "video_1","video_2"]
for idx in range(total):
    mean_0 = np.mean(count_dict[f'{idx}_0'])
    mean_1 = np.mean(count_dict[f'{idx}_1'])
    mean_2 = np.mean(count_dict[f'{idx}_2'])
    means = [mean_0, mean_1, mean_2]
    perc_dict[f'{dataset}_{names[idx]}'] = means

print(perc_dict)