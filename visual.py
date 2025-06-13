import pickle
import torch
import os

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

    print("Ground truth type:\n",' '.join(str(v) for v in gt_type1[i].tolist()))
    print("Mode of predicted types in original:\n",' '.join(str(v) for v in pred_type1[i].tolist()))
#    print("Mode of predicted types in changed:\n",' '.join(str(v) for v in pred_type2[i].tolist()),"\n\n")

    print("Ground truth interarrival time:\n",' '.join(f'{v:.4f}' for v in gt_dt1[i].tolist()[:index]))
    print("Mean of predicted interarrival time in original:\n",' '.join(f'{v:.4f}' for v in pred_x1[i].tolist()[:count]))
#    print("Mean of predicted interarrival time in changed:\n",' '.join(f'{v:.4f}' for v in pred_x2[i].tolist()),"\n\n")

    print("Ground truth arrival time:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(gt_dt1[i], dim=0).tolist()[:index]))
    print("Mean of predicted arrival time in original:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x1[i], dim=0).tolist()[:count]))
#    print("Mean of predicted arrival time in changed:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x2[i], dim=0).tolist()),"\n\n")


    print("\n\n")


print(gt_seq_len)
print(pred_seq_len)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gt_arrival = torch.cumsum(gt_dt1, dim=1).flatten().tolist()
pred_arrival = torch.cumsum(pred_x1, dim=1).flatten().tolist()
#sns.kdeplot(gt_arrival, label="Ground Truth", bw_adjust=1)
#sns.kdeplot(pred_arrival, label="Predicted", bw_adjust=1)

#plt.xlabel("Arrival Time")
#plt.ylabel("Density")
#plt.title("Arrival Time Distributions")
#plt.legend()
#plt.show()

from collections import Counter

freq_gt = Counter(gt_seq_len)
freq_pred = Counter(pred_seq_len)

keys = sorted(set(freq_gt.keys()).union(freq_pred.keys()))
values_gt = [freq_gt.get(k, 0) for k in keys]
values_pred = [freq_pred.get(k, 0) for k in keys]

x = range(len(keys))
bar_width = 0.35

plt.bar([i - bar_width/2 for i in x], values_gt, width=bar_width, label='Ground truth', color='blue')
plt.bar([i + bar_width/2 for i in x], values_pred, width=bar_width, label='Generated', color='orange')

plt.xticks(x, keys)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Distribution')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


