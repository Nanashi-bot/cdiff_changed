import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# TAKING AMAZON DATA


# THE FOLLOWING ARE FOR WHEN SEQUENCE IS STARTED FROM SECOND VALUE NOT FIRST
#base_path = './log/flow/amazon/cross_diffusion_discrete_boxcox_200_tgt_len_20/cosanneal/original1000/samples/sample_ep900_s1_num_s_7_num_steps_200/'
base_path  = './nullsamples/random88'
gt_dt1 = torch.load(os.path.join(base_path, 'gt_dt.pt'))
gt_type1 = torch.load(os.path.join(base_path, 'gt_type.pt'))
samples_dt1 = torch.load(os.path.join(base_path, 'samples_dt.pt'))
samples_type1 = torch.load(os.path.join(base_path, 'samples_type.pt'))


# THE FOLLOWING ARE FOR WHEN SEQUENCE IS STARTED FROM FIRST VALUE ITSELF
#base_path = './log/flow/amazon/cross_diffusion_discrete_boxcox_200_tgt_len_20/cosanneal/changed1000/samples/sample_ep950_s1_num_s_7_num_steps_200'
gt_dt2 = torch.load(os.path.join(base_path, 'gt_dt.pt'))
gt_type2 = torch.load(os.path.join(base_path, 'gt_type.pt'))
samples_dt2 = torch.load(os.path.join(base_path, 'samples_dt.pt'))
samples_type2 = torch.load(os.path.join(base_path, 'samples_type.pt'))


# 500 length
#print(len(gt_dt))
#print(len(samples_dt))
#print(len(gt_type))
#print(len(samples_type))


def first_repeated_index(lst):
    seen = {}
    for idx, val in enumerate(lst):
        if val in seen:
            return seen[val]
        seen[val] = idx
    return len(lst)-2

pred_type1 = torch.mode(samples_type1, dim=-1).values.long()
pred_x1 = samples_dt1.mean(dim=-1).squeeze(-1)

pred_type2 = torch.mode(samples_type2, dim=-1).values.long()
pred_x2 = samples_dt2.mean(dim=-1).squeeze(-1)

gt_seq_len = []
pred_seq_len = []
gt_types = []
pred_types = []
# Total 500
for i in range(500):

    print(f"###### FOR SEQUENCE {i+1}: ######\n")

    index = first_repeated_index(torch.cumsum(gt_dt1[i], dim=0).tolist()) + 1
    print("Number of events in original sequence:", index-1)
    threshold = torch.cumsum(gt_dt1[i], dim=0).tolist()[index]
    print("Last time stamp: ", threshold)
    count = (torch.cumsum(pred_x1[i], dim=0) < threshold).sum().item()
    print("Number of events in same time frame as ground truth: ", count)
    gt_seq_len.append(index-1)
    pred_seq_len.append(count)
    gt_types.append(gt_type1[i].tolist()[:index])
    pred_types.append(pred_type1[i].tolist()[:count])

    print("Ground truth type:\n",' '.join(str(v) for v in gt_type1[i].tolist()[:index]))
    print("Mode of predicted types in original:\n",' '.join(str(v) for v in pred_type1[i].tolist()[:count]))
#    print("Mode of predicted types in changed:\n",' '.join(str(v) for v in pred_type2[i].tolist()),"\n\n")

    print("Ground truth interarrival time:\n",' '.join(f'{v:.4f}' for v in gt_dt1[i].tolist()[:index]))
    print("Mean of predicted interarrival time in original:\n",' '.join(f'{v:.4f}' for v in pred_x1[i].tolist()[:count]))
#    print("Mean of predicted interarrival time in changed:\n",' '.join(f'{v:.4f}' for v in pred_x2[i].tolist()),"\n\n")

    print("Ground truth arrival time:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(gt_dt1[i], dim=0).tolist()[:index]))
    print("Mean of predicted arrival time in original:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x1[i], dim=0).tolist()[:count]))
#    print("Mean of predicted arrival time in changed:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x2[i], dim=0).tolist()),"\n\n")


    print("\n\n")


#print(gt_seq_len)
#print(pred_seq_len)

 
#print(gt_types)
#print(pred_types)


def arrival_time_visual(ax=None):
    gt_arrival = torch.cumsum(gt_dt1, dim=1).flatten().tolist()
    pred_arrival = torch.cumsum(pred_x1, dim=1).flatten().tolist()
    
    if ax is None:
        ax = plt.gca()
    
    sns.kdeplot(gt_arrival, label="Ground Truth", bw_adjust=1, ax=ax)
    sns.kdeplot(pred_arrival, label="Predicted", bw_adjust=1, ax=ax)
    
    ax.set_xlabel("Arrival Time")
    ax.set_ylabel("Density")
    ax.set_title("Arrival Time Distributions")
    ax.legend()


def seq_len_visual(ax=None):
    freq_gt = Counter(gt_seq_len)
    freq_pred = Counter(pred_seq_len)
    
    keys = sorted(set(freq_gt.keys()).union(freq_pred.keys()))
    values_gt = [freq_gt.get(k, 0) for k in keys]
    values_pred = [freq_pred.get(k, 0) for k in keys]
    
    x = range(len(keys))
    bar_width = 0.35

    if ax is None:
        ax = plt.gca()
    
    ax.bar([i - bar_width/2 for i in x], values_gt, width=bar_width, label='Ground truth', color='blue')
    ax.bar([i + bar_width/2 for i in x], values_pred, width=bar_width, label='Generated', color='orange')
    
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Sequence Length Distribution')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)


def type_visualise(ax=None):
    flat1 = np.concatenate(gt_types)
    flat2 = np.concatenate(pred_types)

    count1 = Counter(flat1)
    count2 = Counter(flat2)
    
    all_keys = sorted(set(count1.keys()) | set(count2.keys()))
    freq1 = [count1.get(k, 0) for k in all_keys]
    freq2 = [count2.get(k, 0) for k in all_keys]
    
    x = np.arange(len(all_keys))
    width = 0.4

    if ax is None:
        ax = plt.gca()
    
    ax.bar(x - width/2, freq1, width=width, label='Ground Truth')
    ax.bar(x + width/2, freq2, width=width, label='Generated')
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=90)
    ax.set_xlabel('Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Type Frequency Distribution')
    ax.legend()


# Function to display all three
def plot_all(arrival=True, seq_len=True, types=True):
    n = sum([arrival, seq_len, types])
    fig, axs = plt.subplots(n, 1, figsize=(8, 5 * n))
    
    if n == 1:
        axs = [axs]
    
    i = 0
    if arrival:
        arrival_time_visual(ax=axs[i])
        i += 1
    if seq_len:
        seq_len_visual(ax=axs[i])
        i += 1
    if types:
        type_visualise(ax=axs[i])
    
    plt.tight_layout()
    plt.show()

plot_all(arrival=True, seq_len=True, types=True)
