import torch
import matplotlib.pyplot as plt

confidence = torch.tensor([0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])
correct_label = torch.tensor([True, True, False, True, True, False, True, True, False, False])
num_pos = torch.sum(correct_label == True).item()
num_neg = torch.sum(correct_label == False).item()

tp = 0
fp = 0
last_tp = 0

fpr_list = [0]
tpr_list = [0]

for i in range(len(confidence)):
    # check this is currect
    # tp = torch.sum(correct_label[:i+1] == True).item()
    # fp = torch.sum(correct_label[:i+1] == True).item()
    if (i > 0) and (correct_label[i] != correct_label[i-1]) and (correct_label[i] == False) \
        and tp > last_tp:
            last_tp = tp
            tpr = tp / num_pos
            fpr = fp / num_neg
            tpr_list.append(tpr)
            fpr_list.append(fpr)
    if correct_label[i] == True:
        tp += 1
    else:
        fp += 1

tpr = tp / num_pos
fpr = fp / num_neg
tpr_list.append(tpr)
fpr_list.append(fpr)

#plt.figure(figsize=(8, 6))
plt.scatter(fpr_list, tpr_list, marker='o', c='b')
plt.plot(fpr_list, tpr_list)
plt.title('roc curve')
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.legend()
plt.grid(True)
plt.show()
