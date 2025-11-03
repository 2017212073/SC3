import os
import argparse
import time
import torch
import numpy as np
from utils import yaml_config_hook
from modules import signal_network, signal_cnn
from torch.utils.data import TensorDataset
from evaluation import evaluation
import matplotlib.pyplot as plt
import colorsys
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # color_list = list(colors.cnames)
    # color_list = ['k', 'r', 'y', 'g', 'b']
    color_list = list(map(lambda x: color(tuple(x)), ncolors(len(set(label)))))
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=color_list[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    tsne_feature_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c, c_feature = model.forward_cluster(x)
        c = c.detach()
        c_feature = c_feature.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        tsne_feature_vector.extend(c_feature.cpu().detach().numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector, np.array(tsne_feature_vector)

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
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal_len = 0

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
        dataset = TensorDataset(x, y)

    elif args.dataset == "wifi":
        # 读取16类数据集
        class_num = 10
        X16_train = np.load(r"/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/X_train_16Class.npy")
        Y16_train = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/Y_train_16Class.npy")
        X16_test = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/X_test_16Class.npy")
        Y16_test = np.load("/media/mhb/jxp01/workspace/xzt_/SCSC(Ours)_wifi/datasets/WiFi_ft62/Y_test_16Class.npy")

        # 切割为10类数据集
        mask10_train = Y16_train < 10
        X10_train = X16_train[mask10_train]
        Y10_train = Y16_train[mask10_train]

        mask10_test = Y16_test < 10
        X10_test = X16_test[mask10_test]
        Y10_test = Y16_test[mask10_test]
        x = np.concatenate((X10_train, X10_test), axis=0)
        x = x.transpose((0, 2, 1))
        x = (x - np.mean(x)) / np.std(x)
        print(np.mean(x), np.std(x))

        y = np.concatenate((Y10_train, Y10_test), axis=0)
        y = y.astype(int)
        x = torch.tensor(x)
        x = x.type(torch.FloatTensor)
        signal_len = x.shape[1]
        print(x.shape)
        y = torch.tensor(y)
        dataset = TensorDataset(x, y)

    elif args.dataset == "XSRP":
        class_num = 10
        x = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC_tongxin/datasets/XSRP_data_10.npy')
        y = np.load('/media/mhb/jxp01/workspace/xzt_/SCSC_tongxin/datasets/XSRP_label_10.npy')

        mask_train = y < class_num
        x = x[mask_train]
        y = y[mask_train]
        # x = x.transpose((0, 2, 1))
        print('XSRP:数据集分布', x.shape, np.mean(x), np.std(x))
        x = (x - np.mean(x)) / np.std(x)
        y = y.astype(int)
        x = torch.tensor(x)
        x = x.type(torch.FloatTensor)
        signal_len = x.shape[1]
        y = torch.tensor(y)
        dataset = TensorDataset(x, y)

    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    model_list = os.listdir(args.model_path)
    model_list.sort()
    record_file = open(os.path.join(args.model_path, 'result_record.txt'), 'w')
    nmi_list = list()
    ari_list = list()
    f_list = list()
    acc_list = list()
    for counter, model_load_iter in enumerate(model_list):
        if args.dataset == "XSRP" or args.dataset == "CBRS" or args.dataset == "wifi":
            res = signal_cnn.TIMFE(signal_len)
            model = signal_network.Network(res, args.feature_dim, class_num)
        else:
            raise NotImplementedError

        model_fp = os.path.join(args.model_path, model_load_iter)
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        print("### Creating features from model ###")
        begin_time = time.time()
        X, Y, T = inference(data_loader, model, device)
        ''''''
        # # 画tsne图像
        # tsne = TSNE(n_components=2)
        # XY = np.vstack((X, Y)).transpose((1, 0))
        # train_X, test_X, train_Y, test_Y = train_test_split(T, XY, test_size=500, random_state=42)
        # train_Y, test_Y = train_Y.transpose((1, 0)), test_Y.transpose((1, 0))
        # result = tsne.fit_transform(test_X)
        # plt.figure()
        # fig_pre = plot_embedding(result, test_Y[0], "Contrastive-Clustering-pre")
        # plt.figure()
        # fig_real = plot_embedding(result, test_Y[1], "Contrastive-Clustering-real")
        # plt.show()
        # # ''''''
        nmi, ari, f, acc = evaluation.evaluate(Y, X)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f_list.append(f)
        acc_list.append(acc)
        print('counter = {:d} NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(counter, nmi, ari, f, acc))
        print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc), file=record_file)
        print(time.time()- begin_time)
    record_file.close()
    plt.figure()
    x_axis = np.arange(0, len(nmi_list))
    x_axis = x_axis * 10
    plt.plot(x_axis, nmi_list, color="green", label="nmi")
    plt.plot(x_axis, ari_list, color="red", label="ari")
    plt.plot(x_axis, f_list, color="blue", label="f")
    plt.plot(x_axis, acc_list, color="gray", label="acc")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(args.model_path, "results_plot.jpg"))