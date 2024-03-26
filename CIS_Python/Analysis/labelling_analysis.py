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
# vid_dict = user_data.get_video_data()
# arx_responses = vid_dict["arx"]
# uni_responses = vid_dict["uni"]
# eth_responses = vid_dict["eth"]
# zara_responses = vid_dict["zara"]
# total = len(arx_responses) + len(uni_responses) + len(eth_responses) + len(zara_responses)
# responses = [arx_responses[0], arx_responses[1], uni_responses[0], uni_responses[1], eth_responses[0], eth_responses[1], zara_responses[0],
#              zara_responses[1], zara_responses[2]]
# img_dict = user_data.get_image_data()
# set_dict = user_data.get_set_data()
# vid_dict = user_data.get_video_data()
# synth_dict = user_data.get_synth_data()
# dataset = "eth"
# img_responses = img_dict[f"{dataset}"]
# set_responses = set_dict[f"{dataset}"]
# vid_responses = vid_dict[f"{dataset}"]
# # synth_responses = synth_dict[f"{dataset}"]
# total = 4
# responses = [img_responses, set_responses, vid_responses[0], vid_responses[1]]
img_dict = user_data.get_image_data()
set_dict = user_data.get_set_data()
vid_dict = user_data.get_video_data()
synth_dict = user_data.get_synth_data()
dataset = "arx"
img_responses = img_dict[f"{dataset}"]
set_responses = set_dict[f"{dataset}"]
vid_responses = vid_dict[f"{dataset}"]
synth_responses = synth_dict[f"{dataset}"]
total = 8
responses = [img_responses, set_responses, vid_responses[0], vid_responses[1],synth_responses[0],synth_responses[1],synth_responses[2], synth_responses[3]]
# img_dict = user_data.get_synth_data()
# arx_responses = img_dict["arx"]
# zara_response = img_dict["zara"]
# total = 6
# responses = [arx_responses[0], arx_responses[1], arx_responses[2], arx_responses[3], zara_response[0], zara_response[1]]
# img_dict = user_data.get_set_data()
# arx_responses = img_dict["arx"]
# uni_responses = img_dict["uni"]
# eth_responses = img_dict["eth"]
# total = 3
# responses = [arx_responses, uni_responses, eth_responses]

# APPLY MODEL:
def apply_model(input_ids, idx):
    with torch.no_grad():
            # outputs = model_2(input_ids)
            [gt_mean, gt_var], w, [embeds_gt, embeds] = model_3(input_ids,"Embeddings")
            # w = sample.sample_weight((gt_mean, gt_var))
            # z = outputs[0]
            z = normalize_weights(gt_mean) #TODO
            predictions = torch.argmax(z, dim = 1)
    group_id = [idx] * z.shape[0]
    return z.numpy(), predictions.numpy(), group_id

# ANALYSE:
def plot_z_mean(z_mean, predictions, group_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    marker_dict = {0: ".", 1: "*", 2: "x"}
    marker_dict_2 = {"Goal Label": ".", "Group Label": "*", "Interaction Label": "x"}
    # markers=[".", "*", "x", "^", ">"]
    markers=["x", "+",".", "*",   ">", "^", "<", "v"]
    x_p = z_mean[:,0]
    y_p = z_mean[:,1]
    z_p = z_mean[:,2]
 
    # labels = ['Vid. 1', 'Vid. 2', 'Vid. 3', 'Vid. 4', 'Vid. 5', 'Vid. 6', 'Vid. 7', 'Vid. 8', 'Vid. 9']
    # colors = ['olivedrab', 'forestgreen', 'indianred', 'firebrick', 'darkorange' , 'darkgoldenrod', 'royalblue', 'navy', 'slategray']
    # labels = ['Zara Image', 'Church Image Set','Church Clip. 1', 'Church Clip. 2']
    # labels = ['Clip. 1', 'Clip. 2', 'Clip. 3', 'Synth. Top', 'Synth. Corner']
    labels = ['Image', 'Image Set', 'Clip 1', 'Clip 2', 'Synth. Corner 1', 'Synth. Top 1', 'Synth. Corner 2', 'Synth. Top 2']
    color_dict={0: 'firebrick', 1: 'royalblue', 2: 'forestgreen'}
    color_dict_2={"Goal": 'firebrick', "Group" : 'royalblue', "Interaction" : 'forestgreen'}
    # colors = ['forestgreen', 'indianred', 'darkorange' ]
    # labels = ['Set 1', 'Set 2', 'Set 3']
    # colors = ['olivedrab', 'forestgreen', 'limegreen', 'lightseagreen', 'royalblue', 'navy']
    # labels = ['Syn.Vid. 1', 'Syn.Vid. 2', 'Syn.Vid 3','Syn.Vid. 4', 'Syn.Vid. 5', 'Syn.Vid 6']

    # for i in range(z_mean.shape[0]):
    #     ax.scatter(x_p[i], y_p[i], z_p[i], c=colors[group_id[i]], marker = marker_dict[int(predictions[i])])
    for i in range(z_mean.shape[0]):
        ax.scatter(x_p[i], y_p[i], z_p[i], c=color_dict[int(predictions[i])], marker = markers[group_id[i]], s=20)

    ax.set_xlabel('Goal')
    ax.set_ylabel('Group') 
    ax.set_zlabel('Interaction')
    ax.set_title('Church Descriptions: Distribution Labels')

    # legend_proxies = [plt.Line2D([0], [0], linestyle="none", marker='o', markersize=8, markerfacecolor=color, markeredgewidth=2,
    #                               markeredgecolor=color) for color in colors]
    # ax.legend(legend_proxies, labels, loc = 'upper right')
    legend_proxies = [plt.Line2D([0], [0], linestyle="none", marker=marker, markersize=6, markerfacecolor='k', markeredgewidth=2,
                                  markeredgecolor='k') for marker in markers]
    ax.legend(legend_proxies, labels, loc = 'upper right')

    # marker_legend_proxies = [
    # plt.Line2D([], [], linestyle="none", marker=marker, markersize=10, color='k', label=label)
    # for label, marker in marker_dict_2.items()]
    # ax.legend(handles=marker_legend_proxies, loc='lower left')
    # marker_legend_proxies = [
    # plt.Line2D([], [], linestyle="none", marker='o', markersize=6, color=color, label=label)
    # for label, color in color_dict_2.items()]
    # ax.legend(handles=marker_legend_proxies, loc='lower left')


    # plt.gca().add_artist(ax.legend(handles=marker_legend_proxies))

    plt.show()

model_runs = 1
run_means = []
run_predictions = []
means_dict = {}
stds_dict = {}
pred_dict = {}
for run in tqdm(range(model_runs)):
    z_means = []
    predictions = []
    group_ids = []
    for idx in range(total):
        tokens = user_data.tokenize_sentences(responses[idx], 47)
        input_ids = tokens
        z_mean, prediction, group_id = apply_model(input_ids, idx)
        z_means.append(z_mean)
        predictions.append(prediction)
        group_ids += group_id
        if run == 0:
            means_dict[str(idx)] = [np.mean(z_mean, axis = 0)]
            stds_dict[str(idx)] = [np.std(z_mean, axis = 0)]
            pred_dict[str(idx)] = [prediction]
        else:
            means_dict[str(idx)].append(np.mean(z_mean, axis = 0))
            stds_dict[str(idx)].append(np.std(z_mean, axis = 0))
            pred_dict[str(idx)].append(prediction)
    z_m = np.vstack(z_means)
    preds = np.hstack(predictions)
    run_means.append(z_m)
    run_predictions.append(preds)

for idx in range(total):
    idx_means = np.mean(np.stack(means_dict[str(idx)]), axis = 0)
    idx_stds = np.mean(np.stack(stds_dict[str(idx)]), axis = 0)
    idx_pred = np.mean(np.stack(pred_dict[str(idx)]), axis = 0)
    goal_preds = np.sum(idx_pred == 0)
    group_pred = np.sum(idx_pred == 1)
    inter_pred = np.sum(idx_pred == 2)
    print("idx: ", idx, " Number of Points: ", len(responses[idx]), "with (mean, std): ", idx_means, idx_stds,
           "and Labels: ",goal_preds, group_pred, inter_pred) 

z_m = np.mean(np.stack(run_means), axis = 0)
preds = np.mean(np.stack(run_predictions), axis = 0)
plot_z_mean(z_m, preds, group_ids)