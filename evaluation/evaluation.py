import numpy as np
from sklearn import metrics
from munkres import Munkres
import torch
import matplotlib.pyplot as plt
import random
import colorsys
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, save_filename=None, title='Confusion matrix'):
    # 使用蓝色系颜色映射
    cmap = plt.cm.Blues  # 改为蓝色系
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0

    # 先绘制图像获取颜色映射
    if (intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        # 设置颜色条范围为0到1
        plt.clim(0, 1)

    for x_test, y_test in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_test][x_test]
            # 根据背景颜色深浅选择文字颜色
            if c > cm.max() * 0.6:  # 如果背景颜色较深
                text_color = 'white'
            else:  # 如果背景颜色较浅
                text_color = 'black'
            plt.text(x_test, y_test, "%d" % (c,), color=text_color, fontsize=10, va='center', ha='center')

        else:
            c = cm_normalized[y_test][x_test]
            # 根据背景颜色深浅选择文字颜色
            if c > 0.5:  # 如果背景颜色较深（大于0.5）
                text_color = 'white'
            else:  # 如果背景颜色较浅
                text_color = 'black'

            if (c > 0.01):
                plt.text(x_test, y_test, "%0.2f" % (c,), color=text_color, fontsize=10, va='center', ha='center')
            else:
                plt.text(x_test, y_test, "%d" % (0,), color=text_color, fontsize=10, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().set_xlim(-0.5, len(labels) - 0.5)  # 设置x轴范围
    plt.gca().set_ylim(len(labels) - 0.5, -0.5)  # 设置y轴范围（反转）
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-', color='lightsteelblue')  # 网格改为浅钢蓝色
    plt.gcf().subplots_adjust(bottom=0.15)

    # 设置颜色条范围为0到1
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(0, 1.1, 0.1))  # 设置颜色条刻度从0到1，步长为0.1

    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, color='black', fontsize=10)  # x轴标签改为黑色
    plt.yticks(xlocations, labels, rotation=90, color='black', fontsize=10)  # y轴标签改为黑色

    # 设置坐标轴颜色
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_color('black')

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, facecolor='white', dpi=300, bbox_inches='tight')  # 保持背景为白色


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


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
    color_list = list(map(lambda x: color(tuple(x)), ncolors(max(label) + 1)))
    fig = plt.figure(dpi=900)
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if label[i] == -1:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color='black',
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=color_list[label[i]],
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if len(title) != 0:
        plt.title(title)
    return fig


def extract(loader, model, device):
    model.eval()
    feature_vector = []
    tsne_feature_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c, c_feature = model.forward_cluster(x)
        c = c.detach()
        c_feature = c_feature.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        tsne_feature_vector.extend(c_feature.cpu().detach().numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, np.array(tsne_feature_vector)


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


# 计算类别原型
def compute_prototype(feature, label):
    label = np.expand_dims(label, 1)
    prototype_dict = np.zeros((np.max(label) + 1, feature.shape[1]))
    feature = np.concatenate((label, feature), axis=1)
    feature = feature[np.argsort(feature[:, 0])]
    for i in range(np.max(label) + 1):
        prototype_dict[i] = np.expand_dims(np.average(feature[np.where(feature[:, 0] == i), 1:].squeeze(0), axis=0), 0)
    return prototype_dict


def compute_distance(feature, prototype, mode='all'):
    distance = (np.dot(feature, prototype.T)) / \
               (np.dot(np.linalg.norm(feature, ord=2, axis=1, keepdims=True), np.linalg.norm(prototype.T, ord=2, axis=0, keepdims=True)))
    max_distance = np.max(distance, axis=1)
    max_distance_arg = np.argmax(distance, axis=1)
    if mode == 'all':
        return distance, max_distance_arg
    elif mode == 'max':
        return max_distance, max_distance_arg

    else:
        print("wrong mode! please select in 'all' ,'max'.")
        raise NotImplementedError
