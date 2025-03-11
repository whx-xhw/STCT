import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def meta_correction(param, feat, noisy_labels, clean_labels, current_epoch):
    feat = feat_pre_process(feat=feat)
    acc_list = []
    for st in range(param.sample_times):
        last = noisy_labels
        feat, noisy_labels = meta_correction_single_sample_time(param=param, feat=feat, noisy_labels=noisy_labels,
                                                                clean_labels=clean_labels, current_epoch=current_epoch,
                                                                current_sample_times=st)
        current = noisy_labels
        diff = np.mean(last == current)
        print('epoch diff: {}'.format(diff))
        acc0 = np.mean(current == clean_labels)
        acc_list.append(acc0)
        print(st)
        if diff > 0.9995 or st > 30:
            print(acc_list)
            break
    return noisy_labels


def meta_correction_single_sample_time(param, feat, noisy_labels, clean_labels, current_epoch, current_sample_times):
    class_number = len(np.unique(noisy_labels))
    train_pack, val_pack = uniform_sampling(feat=feat, noisy_labels=noisy_labels, clean_labels=clean_labels,
                                            sampling_rate=param.sampling_rate)

    original_training_labels = train_pack[1].copy()

    device = torch.device('cuda:{}'.format(param.device))
    train_feat_tensor = torch.from_numpy(train_pack[0]).float().to(device)
    train_noisy_labels_tensor = torch.from_numpy(train_pack[1]).long().to(device)
    train_noisy_labels_tensor = F.one_hot(train_noisy_labels_tensor, class_number)

    val_feat_tensor = torch.from_numpy(val_pack[0]).float().to(device)
    val_noisy_labels_tensor = torch.from_numpy(val_pack[1]).long().to(device)
    val_noisy_labels_tensor = F.one_hot(val_noisy_labels_tensor, class_number)
    val_noisy_labels_tensor = val_noisy_labels_tensor.float().detach()

    train_noisy_labels_tensor = Variable(train_noisy_labels_tensor.float(), requires_grad=True)

    optim = torch.optim.Adam([train_noisy_labels_tensor], lr=param.nmc_lr, betas=(0.5, 0.999), weight_decay=3e-4)
    loss_mse = torch.nn.MSELoss(reduce=False, size_average=False)

    loss_list = np.zeros(shape=(param.nmc_epoch, ))
    acc_list = np.zeros(shape=(param.nmc_epoch, ))

    running_loss = 0.0
    marker = 0
    changing_epoch = 0
    epoch_counter = 0
    for epoch in range(param.nmc_epoch):
        batch_number_in_one_epoch = train_pack[0].shape[0] // param.nmc_bs

        index = np.arange(train_pack[0].shape[0])
        np.random.shuffle(index)
        index_list = index.tolist()

        for j in range(batch_number_in_one_epoch):
            if param.nmc_bs * (j + 1) > train_pack[0].shape[0]:
                end = train_pack[0].shape[0]
            else:
                end = param.nmc_bs * (j + 1)

            select_idx = index_list[param.nmc_bs * j: end]
            train_feat_bs = train_feat_tensor[select_idx]
            train_feat_bs = train_feat_bs.to(device)

            w = torch.inverse(torch.mm(train_feat_bs.transpose(1, 0), train_feat_bs))
            w = torch.mm(torch.mm(w, train_feat_bs.transpose(1, 0)), train_noisy_labels_tensor[select_idx])
            lrg_pred_for_noisy_val = torch.mm(val_feat_tensor, w)

            optim.zero_grad()
            loss = torch.sum(loss_mse(lrg_pred_for_noisy_val, val_noisy_labels_tensor))
            running_loss += loss.item()
            loss.backward()
            optim.step()

        loss_average = running_loss / batch_number_in_one_epoch
        corrected_train_labels = train_noisy_labels_tensor.detach().cpu().numpy()
        corrected_train_labels = np.argmax(corrected_train_labels, axis=1)
        train_clean_labels = train_pack[2]

        acc = np.mean(corrected_train_labels == train_clean_labels)

        if marker == 0:
            state = detect_changing_epoch(original_labels=original_training_labels, current_labels=corrected_train_labels)
            if state is True:
                marker = 1
                changing_epoch = epoch


        loss_list[epoch] = loss_average
        acc_list[epoch] = acc
        print(f"\repoch:{current_epoch}, sample times:{current_sample_times}/{param.sample_times}, nmc:{epoch+1}/{param.nmc_epoch}, sampled training label acc:{acc}, loss:{loss_average}...", end='', flush=True)
        if marker == 1:
            epoch_counter += 1

        if epoch_counter > int(1.2 * changing_epoch):
            break

    train_original_index_np = np.array(train_pack[3], dtype=np.int32)
    train_feat_np = train_feat_tensor.detach().cpu().numpy()
    train_labels_np = corrected_train_labels

    val_original_index_np = np.array(val_pack[3], dtype=np.int32)
    val_feat_np = val_feat_tensor.detach().cpu().numpy()
    val_labels_np = np.argmax(val_noisy_labels_tensor.detach().cpu().numpy(), axis=1)

    train_pack_new = [train_feat_np, train_labels_np, train_original_index_np, train_pack[2]]
    val_pack_new = [val_feat_np, val_labels_np, val_original_index_np, val_pack[2]]

    resorted_feat, resorted_labels = resort(train_pack=train_pack_new, val_pack=val_pack_new, clean=clean_labels)

    acc = np.mean(resorted_labels == clean_labels)

    print("epoch:{}, sample times:{}/{}, training label acc:{}".format(current_epoch, current_sample_times, param.sample_times, acc))
    return resorted_feat, resorted_labels


def uniform_sampling(feat, noisy_labels, clean_labels, sampling_rate):
    sample_size = feat.shape[0]
    sample_idx = np.arange(sample_size)
    np.random.shuffle(sample_idx)

    val_idx = sample_idx[:int(np.ceil(sampling_rate * sample_size))]
    train_idx = sample_idx[int(np.ceil(sampling_rate * sample_size)):]

    train_feat = feat[train_idx]
    train_noisy_labels = noisy_labels[train_idx]
    train_clean_labels = clean_labels[train_idx]

    val_feat = feat[val_idx]
    val_noisy_labels = noisy_labels[val_idx]
    val_clean_labels = clean_labels[val_idx]

    train_pack = [train_feat, train_noisy_labels, train_clean_labels, train_idx]
    val_pack = [val_feat, val_noisy_labels, val_clean_labels, val_idx]
    return train_pack, val_pack


def resort(train_pack, val_pack, clean):
    train_feat_np = train_pack[0]
    train_labels_np = train_pack[1]
    train_original_index_np = train_pack[2]

    val_feat_np = val_pack[0]
    val_labels_np = val_pack[1]
    val_original_index_np = val_pack[2]

    all_index = np.concatenate([train_original_index_np, val_original_index_np], axis=0)
    all_feat = np.concatenate([train_feat_np, val_feat_np], axis=0)
    all_labels = np.concatenate([train_labels_np, val_labels_np], axis=0)
    all_clean = np.concatenate([train_pack[3], val_pack[3]], axis=0)
    all_clean1 = clean[all_index]

    print(np.mean(all_labels == all_clean))

    resorted_feat = np.zeros(shape=(all_feat.shape[0], all_feat.shape[1]))
    resorted_labels = np.zeros(shape=(all_labels.shape[0], ), dtype=np.int64)
    resorted_labels1 = np.zeros(shape=(all_labels.shape[0],), dtype=np.int64)
    for i in range(all_feat.shape[0]):
        resorted_feat[all_index[i]] = all_feat[i]
        resorted_labels[all_index[i]] = all_labels[i]
        resorted_labels1[all_index[i]] = all_clean[i]

    return resorted_feat, resorted_labels


def feat_pre_process(feat):
    feat_abs = np.abs(feat)
    feat_abs_mean = np.mean(feat_abs, axis=0)
    effective_idx = np.where(feat_abs_mean > 1e-4)[0]
    effective_feat = feat[:, effective_idx]
    return effective_feat


def detect_changing_epoch(original_labels, current_labels):
    diff = np.mean(original_labels == current_labels)
    change_start = False
    if diff < 1.0:
        change_start = True
    return change_start



