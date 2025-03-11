import numpy as np
import torch
from dataset.data import *
import torch.nn.functional as F
from util.meter import *
from dataset.my_semi_data1 import *
from util.torch_dist_sum import *
import argparse
from util.accuracy import accuracy
from util.dist_init import *
from network.google_wide_resnet import wide_resnet28w2, wide_resnet28w8
from network.head import *
import time
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from network.simmatch import SimMatch



parser = argparse.ArgumentParser()
parser.add_argument('--stct_epoch', type=int, default=0)
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=301)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--bank_m', type=float, default=0.7)
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--c_smooth', type=float, default=0.9)
parser.add_argument('--lambda_in', type=float, default=1)
parser.add_argument('--st', type=float, default=0.1)
parser.add_argument('--tt', type=float, default=0.1)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

epochs = args.epochs


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_x = AverageMeter('X', ':.4e')
    losses_u = AverageMeter('U', ':.4e')
    losses_in = AverageMeter('In', ':.4e')
    progress = ProgressMeter(
        n_iters_per_epoch,
        [batch_time, data_time, losses_x, losses_u, losses_in],
        prefix="Epoch: [{}]".format(epoch)
    )
    end = time.time()

    # dltrain_x.sampler.set_epoch(epoch)
    # dltrain_u.sampler.set_epoch(epoch)
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)

    model.train()
    Loss_U = 0.0
    Loss_X = 0.0
    Loss_In = 0.0
    Loss_All = 0.0
    counter = 0
    for i in range(n_iters_per_epoch):
        counter += 1

        data_time.update(time.time() - end)

        ims_x_weak, lbs_x, index_x = next(dl_x)
        (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)

        lbs_x = lbs_x.to(device)
        lbs_x = lbs_x.long()
        index_x = index_x.to(device)
        lbs_u_real = lbs_u_real.to(device)
        ims_x_weak = ims_x_weak.to(device)
        ims_u_weak = ims_u_weak.to(device)
        ims_u_strong = ims_u_strong.to(device)

        logits_x, pseudo_label, logits_u_s, loss_in = model(
            ims_x_weak, ims_u_weak, ims_u_strong,
            labels=lbs_x, index=index_x, start_unlabel=epoch > 0, args=args
        )
        loss_x = F.cross_entropy(logits_x, lbs_x, reduction='mean')

        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        loss_u = (torch.sum(-F.log_softmax(logits_u_s, dim=1) * pseudo_label.detach(), dim=1) * mask).mean()

        loss_in = loss_in.mean()
        loss = loss_x + loss_u + loss_in * args.lambda_in

        Loss_U += loss_u.item()
        Loss_X += loss_x.item()
        Loss_In += loss_in.item()
        Loss_All += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses_x.update(loss_x.item())
        losses_u.update(loss_u.item())
        losses_in.update(loss_in.item())

        if (i % 340 == 0):
            progress.display(i)

    return Loss_U / counter, Loss_X / counter, Loss_In / counter, Loss_All / counter


@torch.no_grad()
def val(model, val_loader, device, args):
    model.eval()
    # ---------------------- Test --------------------------
    model_acc_counter = 0
    ema_acc_counter = 0

    if args.dataset == 'cifar10':
        ema_feat_nd = np.zeros(shape=(50000, 128))
    elif args.dataset == 'cifar100':
        ema_feat_nd = np.zeros(shape=(50000, 512))

    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            label_np = label.detach().cpu().numpy()

            ema_out, ema_feat = model.ema(image, return_feat=True)
            ema_out_np = ema_out.detach().cpu().numpy()
            ema_pre = np.argmax(ema_out_np, axis=1)

            ema_feat_np = ema_feat.detach().cpu().numpy()

            ema_feat_nd[i * 1000: (i + 1) * 1000] = ema_feat_np

    np.save('./improved_feat.npy', ema_feat_nd)


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    # ---------------------- Test --------------------------
    model_acc_counter = 0
    ema_acc_counter = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            label_np = label.detach().cpu().numpy()

            out = model.encoder_q(image)
            out_np = out.detach().cpu().numpy()
            model_pre = np.argmax(out_np, axis=1)

            ema_out = model.ema(image)
            ema_out_np = ema_out.detach().cpu().numpy()
            ema_pre = np.argmax(ema_out_np, axis=1)

            for uu in range(label_np.shape[0]):
                if label_np[uu] == ema_pre[uu]:
                    ema_acc_counter += 1
            for uu in range(label_np.shape[0]):
                if label_np[uu] == model_pre[uu]:
                    model_acc_counter += 1

    top1_acc = model_acc_counter / 10000
    ema_top1_acc = ema_acc_counter / 10000
    top5_acc = 0.123
    ema_top5_acc = 0.456
    a = 1
    return top1_acc, top5_acc, ema_top1_acc, ema_top5_acc

def main():
    # rank, local_rank, world_size = dist_init(port=args.port)
    device = torch.device('cuda:{}'.format(args.device))
    batch_size = 64
    n_iters_per_epoch = 1024
    lr = 0.03
    mu = 7

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    correct_labels = np.load('./clean_labels.npy'.format(args.dataset))
    noisy_labels = np.load('./noisy_labels_{}.npy'.format(args.stct_epoch))

    dltrain_x, dltrain_u = get_fixmatch_data(
        dataset=args.dataset,
        batch_size=batch_size,
        n_iters_per_epoch=n_iters_per_epoch,
        mu=mu, dist=False, return_index=True,
        noisy_labels=noisy_labels, clean_labels=correct_labels,
        args=args
    )

    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=args.root_path, train=False, download=True,
                                         transform=get_test_augment('cifar100'))
        num_classes = 100

    if args.dataset == 'cifar100':
        weight_decay = 1e-3
        base_model = wide_resnet28w8()
    else:
        weight_decay = 5e-4
        base_model = wide_resnet28w2()

    model = SimMatch(base_encoder=base_model, K=len(dltrain_x.dataset), args=args,
                     device=device)
    print(len(dltrain_x.dataset))
    model.to(device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.SGD(grouped_parameters, lr=lr, momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * n_iters_per_epoch)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    best_acc1 = best_acc5 = 0
    best_ema1 = best_ema5 = 0

    start_epoch = 0
    loss_u_array = np.zeros(shape=(epochs,))
    loss_x_array = np.zeros(shape=(epochs,))
    loss_in_array = np.zeros(shape=(epochs,))
    loss_all_array = np.zeros(shape=(epochs,))
    model_acc = np.zeros(shape=(epochs,))
    ema_acc = np.zeros(shape=(epochs,))

    best_acc_ = 0
    for epoch in range(start_epoch, epochs):
        LU, LX, LIn, LAll = train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch,
                                  device=device)
        loss_u_array[epoch] = np.array(LU)
        loss_x_array[epoch] = np.array(LX)
        loss_in_array[epoch] = np.array(LIn)
        loss_all_array[epoch] = np.array(LAll)
        top1_acc, top5_acc, ema_top1_acc, ema_top5_acc = test(model, test_loader, device=device)
        model_acc[epoch] = top1_acc
        ema_acc[epoch] = ema_top1_acc

        best_acc1 = max(top1_acc, best_acc1)
        best_acc5 = max(top5_acc, best_acc5)
        best_ema1 = max(ema_top1_acc, best_ema1)
        best_ema5 = max(ema_top5_acc, best_ema5)

        if ema_top1_acc > best_acc_:
            best_acc_ = ema_top1_acc

        print(
            'Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(
                epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc1, best_acc5=best_acc5))
        print(
            'Epoch:{} * EMA@1 {top1_acc:.3f} EMA@5 {top5_acc:.3f} Best_EMA@1 {best_acc:.3f} Best_EMA@5 {best_acc5:.3f}'.format(
                epoch, top1_acc=ema_top1_acc, top5_acc=ema_top5_acc, best_acc=best_ema1, best_acc5=best_ema5))

        if epoch % 10 ==0:

            if args.dataset == 'cifar10':
                val_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                               transform=get_test_augment('cifar10'))

            elif args.dataset == 'cifar100':
                val_dataset = datasets.CIFAR100(root='./data', train=True, download=True,
                                                transform=get_test_augment('cifar100'))

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, num_workers=16, pin_memory=True,
                                                     shuffle=False)

            val(model=model, val_loader=val_loader, device=device, args=args)

if __name__ == "__main__":
    main()



