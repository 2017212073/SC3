import os
import shutil
import time
import numpy as np
import torch
import random
import argparse
from modules import contrastive_loss, tensorsDatasetT, signal_transform, signal_network, signal_cnn
from utils import yaml_config_hook, save_model
from torch.utils import data
from utils.save_model import save_model_ori

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train():
    loss_epoch = 0
    loss_instance_epoch = 0
    loss_cluster_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
        loss_instance_epoch += loss_instance.item()
        loss_cluster_epoch += loss_cluster.item()
    return loss_epoch, loss_instance_epoch, loss_cluster_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    shutil.copy("config/config.yaml", os.path.join(args.model_path, 'config.yaml'))

    print(args.seed)
    set_seed(args.seed)
    signal_len = 0

    # prepare data
    if args.dataset == "CBRS":
        class_num = 10
        x = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/tongxin/tongxin_data.npy')
        y = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/tongxin/tongxin_label.npy')
        print(np.mean(x), np.std(x))
        x = (x - np.mean(x)) / np.std(x)
        y = y.astype(int)
        x = torch.tensor(x)
        x = x.type(torch.FloatTensor)
        y = torch.tensor(y)
        signal_len = x.shape[1]
        dataset = tensorsDatasetT.TensorsDatasetT(x, y, signal_transform.Transforms(prob=0.5))
    elif args.dataset == "wifi":
        # 读取16类数据集
        class_num = 10
        X16_train = np.load(r"/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/X_train_16Class.npy")
        Y16_train = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/Y_train_16Class.npy")
        X16_test = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/X_test_16Class.npy")
        Y16_test = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/Y_test_16Class.npy")

        x = np.concatenate((X16_train, X16_test), axis=0)
        y = np.concatenate((Y16_train, Y16_test), axis=0)

        # 切割为10类数据集
        mask_train = y < class_num
        X_train = x[mask_train]
        Y_train = y[mask_train]

        X_train = X_train.transpose((0, 2, 1))
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)

        print(np.mean(X_train), np.std(X_train))
        # -0.2394559293264678 23.719584767780123

        Y_train = Y_train.astype(int)
        X_train = torch.tensor(X_train)
        X_train = X_train.type(torch.FloatTensor)
        signal_len = X_train.shape[1]
        print(X_train.shape)
        Y_train = torch.tensor(Y_train)
        dataset = tensorsDatasetT.TensorsDatasetT(X_train, Y_train, signal_transform.Transforms(prob=0.5))
    elif args.dataset == "XSRP":
        class_num = 10

        x = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC_tongxin/datasets/XSRP_data_10.npy')
        y = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC_tongxin/datasets/XSRP_label_10.npy')

        # 切割为10类数据集
        mask_train = y < class_num
        x = x[mask_train]
        y = y[mask_train]
        print('XSRP:数据集分布', x.shape, np.mean(x), np.std(x))
        x = (x - np.mean(x)) / np.std(x)
        y = y.astype(int)
        x = torch.tensor(x)
        x = x.type(torch.FloatTensor)
        signal_len = x.shape[1]
        y = torch.tensor(y)
        dataset = tensorsDatasetT.TensorsDatasetT(x, y, signal_transform.Transforms(prob=0.5))

    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    if args.dataset == "XSRP" or args.dataset == "CBRS" or args.dataset == "wifi":
        res = signal_cnn.TIMFE(signal_len)
        model = signal_network.Network(res, args.feature_dim, class_num)
        model = model.to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError

    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        begin_time = time.time()
        loss_epoch, loss_instance_epoch, loss_cluster_epoch = train()
        epoch_time = time.time() - begin_time
        print(f"Total time: {epoch_time:.2f}s ({epoch_time / 60:.2f}min)")
        if epoch % 1 == 0:
            save_model_ori(args, model, optimizer, epoch)
        with open(os.path.join(args.model_path, 'loss.txt'), 'a') as f:
            f.write(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)} loss_instance: {loss_instance_epoch / len(data_loader)} loss_cluster: {loss_cluster_epoch / len(data_loader)}\n")
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model_ori(args, model, optimizer, args.epochs)
