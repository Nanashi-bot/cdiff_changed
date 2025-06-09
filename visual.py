import pickle
import torch
import os

# TAKING AMAZON DATA


# THE FOLLOWING ARE FOR WHEN SEQUENCE IS STARTED FROM SECOND VALUE NOT FIRST
base_path = './log/flow/amazon/cross_diffusion_discrete_boxcox_200_tgt_len_20/cosanneal/original1000/samples/sample_ep900_s1_num_s_7_num_steps_200/'
gt_dt1 = torch.load(os.path.join(base_path, 'gt_dt.pt'))
gt_type1 = torch.load(os.path.join(base_path, 'gt_type.pt'))
samples_dt1 = torch.load(os.path.join(base_path, 'samples_dt.pt'))
samples_type1 = torch.load(os.path.join(base_path, 'samples_type.pt'))


# THE FOLLOWING ARE FOR WHEN SEQUENCE IS STARTED FROM FIRST VALUE ITSELF
base_path = './log/flow/amazon/cross_diffusion_discrete_boxcox_200_tgt_len_20/cosanneal/changed1000/samples/sample_ep950_s1_num_s_7_num_steps_200'
gt_dt2 = torch.load(os.path.join(base_path, 'gt_dt.pt'))
gt_type2 = torch.load(os.path.join(base_path, 'gt_type.pt'))
samples_dt2 = torch.load(os.path.join(base_path, 'samples_dt.pt'))
samples_type2 = torch.load(os.path.join(base_path, 'samples_type.pt'))


# 500 length
#print(len(gt_dt))
#print(len(samples_dt))
#print(len(gt_type))
#print(len(samples_type))

pred_type1 = torch.mode(samples_type1, dim=-1).values.long()
pred_x1 = samples_dt1.mean(dim=-1).squeeze(-1)

pred_type2 = torch.mode(samples_type2, dim=-1).values.long()
pred_x2 = samples_dt2.mean(dim=-1).squeeze(-1)


# Total 500
for i in range(10):

    print(f"###### FOR SEQUENCE {i+1}: ######\n")
    print("Ground truth type:\n",' '.join(str(v) for v in gt_type1[i].tolist()))
    print("Mode of predicted types in original:\n",' '.join(str(v) for v in pred_type1[i].tolist()))
    print("Mode of predicted types in changed:\n",' '.join(str(v) for v in pred_type2[i].tolist()),"\n\n")

    print("Ground truth interarrival time:\n",' '.join(f'{v:.4f}' for v in gt_dt1[i].tolist()))
    print("Mean of predicted interarrival time in original:\n",' '.join(f'{v:.4f}' for v in pred_x1[i].tolist()))
    print("Mean of predicted interarrival time in changed:\n",' '.join(f'{v:.4f}' for v in pred_x2[i].tolist()),"\n\n")

    print("Ground truth arrival time:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(gt_dt1[i], dim=0).tolist()))
    print("Mean of predicted arrival time in original:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x1[i], dim=0).tolist()))
    print("Mean of predicted arrival time in changed:\n",' '.join(f'{v:.4f}' for v in torch.cumsum(pred_x2[i], dim=0).tolist()),"\n\n")


    print("\n\n")

