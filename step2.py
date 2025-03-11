import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.mixture import GaussianMixture
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--stct_epoch', type=int, default=0)
parser.add_argument('--k', type=int, default=2000)
parser.add_argument('--sampling_rate', type=float, default=0.5)
parser.add_argument('--eta_num', type=int, default=800)
param = parser.parse_args()


epoch = param.stct_epoch
knn = param.k
num = int(param.eta_num * (epoch + 1))
M = int(num * 0.75)


noisy_labels = np.load('./noisy_labels_{}.npy'.format(epoch))
clean_labels = np.load('./clean_labels.npy')
class_number = len(np.unique(noisy_labels))
init_feat = np.load('./init_feat.npy')

noisy_labels_per_class = np.zeros(shape=(class_number,), dtype=np.int32)
for i in range(class_number):
    idx = np.where(noisy_labels == i)[0]
    noisy_labels_per_class[i] = idx.shape[0]

if epoch == 0:
    cosine_similarity_matrix = cosine_similarity(init_feat, init_feat)

else:
    improved_feat = np.load('./improved_feat.npy')
    cosine_similarity_matrix0 = cosine_similarity(init_feat, init_feat)
    cosine_similarity_matrix1 = cosine_similarity(improved_feat, improved_feat)
    cosine_similarity_matrix = (cosine_similarity_matrix0 + cosine_similarity_matrix1) / 2

    del cosine_similarity_matrix0
    del cosine_similarity_matrix1

sample_size = noisy_labels.shape[0]
for i in range(sample_size):
    cosine_similarity_matrix[i, i] = np.abs(cosine_similarity_matrix[i, i] - 1.0)

noisy_one_hot_labels = np.eye(class_number)[noisy_labels]

selected_KNN_samples_list = []
for i in range(sample_size):
    cosine_similarity_i = cosine_similarity_matrix[i]
    top_K_idx = cosine_similarity_i.argsort()[::-1][0:knn]
    selected_KNN_samples_list.append(top_K_idx.tolist())

aggregation_labels = np.zeros(shape=(sample_size,), dtype=np.int32)
aggregation_prob = np.zeros(shape=(sample_size, class_number))
for i in range(sample_size):
    knn_one_hot_labels = noisy_one_hot_labels[selected_KNN_samples_list[i]]
    aggregation_prob[i] = np.mean(np.array(knn_one_hot_labels), axis=0)
    aggregation_labels[i] = np.argmax(aggregation_prob[i])

aggregation_prob_tensor = torch.from_numpy(aggregation_prob).float()
noisy_labels_tensor = torch.from_numpy(noisy_labels).long()
ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
loss_t = ce_loss(aggregation_prob_tensor, noisy_labels_tensor)
loss = loss_t.detach().cpu().numpy()

loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
input_loss = loss.reshape(-1, 1)
gmm = GaussianMixture(n_components=2, max_iter=1000, tol=1e-3, reg_covar=5e-4)
gmm.fit(input_loss)
prob = gmm.predict_proba(input_loss)

pre_prob = np.argmax(prob, axis=1)
pre_0 = np.where(pre_prob == 0)[0]
pre_1 = np.where(pre_prob == 1)[0]

q = prob[:, np.argmax(np.array([pre_0.shape[0], pre_1.shape[0]]))]

select_list = []
for i in range(class_number):
    list_i = np.where(noisy_labels == i)[0]
    q_i = q[list_i]
    select_idx_inner_class = q_i.argsort()[::-1][0: num]
    list_i_np = np.array(list_i)
    select_idx = list_i_np[select_idx_inner_class]
    select_list.append(select_idx)

select_list_np = np.concatenate(select_list, axis=0)

select_train_labels = noisy_labels[select_list_np]
select_clean_labels = clean_labels[select_list_np]

acc = np.mean(select_train_labels == select_clean_labels)
print('selected noisy labels acc:{}'.format(acc))

selected_noisy_labels = noisy_labels[select_list_np]
sel = []
for i in range(10):
    noisy_idx_i = np.where(selected_noisy_labels == i)[0]
    np.random.shuffle(noisy_idx_i)
    se_i = noisy_idx_i[:M]
    sel.append(select_list_np[se_i])

sel = np.concatenate(sel, axis=0)

select_train_labels = noisy_labels[sel]
select_clean_labels = clean_labels[sel]
acc = np.mean(select_train_labels == select_clean_labels)
print('selected noisy labels acc:{}'.format(acc))
np.save('./select_idx.npy', sel)





