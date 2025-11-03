import torch
import torchvision
import numpy as np
import random


def get_piece_list(piece_num, piece_length, single_flag=False):
    if single_flag:
        piece_flag = pow(2, np.random.randint(0, piece_num))
    else:
        piece_flag = np.random.randint(1, pow(2, piece_num))
    piece_list = []
    for ite in range(piece_num):
        if piece_flag % 2 == 1:
            piece_list.append(
                np.arange(ite * piece_length, (ite + 1) * piece_length))
        piece_flag //= 2
    return piece_list


def get_cut_point(piece_list):
    cut_point = []
    for piece_list_ite in range(len(piece_list)):
        if piece_list_ite == 0 or piece_list[piece_list_ite][0] != cut_point[-1]:
            cut_point.append(piece_list[piece_list_ite][0])
        else:
            del cut_point[-1]
        cut_point.append(piece_list[piece_list_ite][-1] + 1)
    return cut_point


class IQPieceExchangePlus:
    def __init__(self, prob=0.5, piece_num=8, single_flag=False):
        self.prob = prob
        self.piece_num = piece_num
        self.single_flag = single_flag
        self.piece_length = -1

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        exchange_flag = np.random.rand() < self.prob
        if exchange_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            piece_list = get_piece_list(self.piece_num, self.piece_length, self.single_flag)
            cut_point = get_cut_point(piece_list)
            for exchange_ite in range(0, len(cut_point), 2):
                data[cut_point[exchange_ite]: cut_point[exchange_ite + 1], :] = \
                    data[cut_point[exchange_ite]: cut_point[exchange_ite + 1], (1, 0)]
            data = torch.Tensor(data)
        return data


class IQAmplitudeAdjust:
    def __init__(self, prob=0.5, piece_num=8, total_flag=True):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1
        self.total_flag = total_flag
        self.scope = [0.9, 1.1]

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        adjust_flag = np.random.rand() < self.prob
        if adjust_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            piece_list = list(np.array(piece_list).reshape(-1))
            if len(piece_list) != 0:
                # 在0.9-1.1范围内抖动
                if self.total_flag:
                    random_list = np.random.rand(len(piece_list)) * (self.scope[1] - self.scope[0]) + self.scope[0]
                    random_list = np.repeat(random_list.reshape(-1, 1), repeats=2, axis=1)
                else:
                    random_list = random.uniform(self.scope[0], self.scope[1])
                data[piece_list, :] = data[piece_list, :] * random_list
                data = torch.Tensor(data)
        return data


class IQAmplitudeNoise:
    def __init__(self, prob=0.5, piece_num=8, total_flag=False):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1
        self.total_flag = total_flag
        self.scope = [0.9, 1.1]

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        adjust_flag = np.random.rand() < self.prob
        if adjust_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            piece_list = list(np.array(piece_list).reshape(-1))
            if len(piece_list) != 0:
                # 在0.9-1.1范围内抖动
                if self.total_flag:
                    random_list = np.random.rand(len(piece_list)) * (self.scope[1] - self.scope[0]) + self.scope[0]
                    random_list = np.repeat(random_list.reshape(-1, 1), repeats=2, axis=1)
                else:
                    random_list = random.uniform(self.scope[0], self.scope[1])
                data[piece_list, :] = data[piece_list, :] * random_list
                data = torch.Tensor(data)
        return data


# 分段抖动，也是用的np.roll()，相当于IQShift的一种演化版本
class IQBudge:
    def __init__(self, prob=0.5, piece_num=8):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        budge_flag = np.random.rand() < self.prob
        if budge_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            if len(piece_list) != 0:
                cut_point = get_cut_point(piece_list)
                for budge_ite in range(0, len(cut_point), 2):
                    data[cut_point[budge_ite]: cut_point[budge_ite + 1]] \
                        = np.roll(data[cut_point[budge_ite]: cut_point[budge_ite + 1]],
                                  int(np.random.rand() * (cut_point[budge_ite + 1] - cut_point[budge_ite])), axis=0)
            data = torch.Tensor(data)
        return data


# 20220409新增 时序翻转
class IQReverse:
    def __init__(self, prob=0.5, piece_num=8):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        budge_flag = np.random.rand() < self.prob
        if budge_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            if len(piece_list) != 0:
                cut_point = get_cut_point(piece_list)
                for rev_ite in range(0, len(cut_point), 2):
                    data[cut_point[rev_ite]: cut_point[rev_ite + 1]] \
                        = np.flipud(data[cut_point[rev_ite]: cut_point[rev_ite + 1]])
            data = torch.Tensor(data)
        return data


class IQCutout:
    def __init__(self, prob=0.5, piece_num=8):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        cutout_flag = np.random.rand() < self.prob
        if cutout_flag:
            piece_list = get_piece_list(self.piece_num, self.piece_length, single_flag=True)
            data[piece_list, :] = 0
        return data


class Dropping:
    def __init__(self, prob=0.5, piece_num=8):
        # 16
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        cutout_flag = np.random.rand() < self.prob
        if cutout_flag:
            piece_list = get_piece_list(self.piece_num, self.piece_length, True)
            data[piece_list, np.random.randint(0, 2)] = 0
        return data


class Pooling:
    def __init__(self, prob=0.5, piece_num=8):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1
        self.pool_factor = 2

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        cutout_flag = np.random.rand() < self.prob
        if cutout_flag:
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            pro_data = data[piece_list, :]
            pro_data[:, 1::self.pool_factor, :] = pro_data[:, ::self.pool_factor, :]
            data[piece_list, :] = pro_data
        return data

# 加噪方法
class IQAddNoise:
    def __init__(self, prob=0.5, piece_num=8, max_db=5):
        self.prob = prob
        self.piece_num = piece_num
        self.max_db = max_db
        self.piece_length = -1

    def help_noise(self, data, snr):
        p_signal = np.sum(abs(data) ** 2) / len(data)
        p_noise = p_signal / 10 ** (snr / 10.0)
        return data + np.repeat(np.random.randn(len(data)) * np.sqrt(p_noise), repeats=2).reshape((-1, 2))

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        add_noise_flag = np.random.rand() < self.prob
        if add_noise_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            data = data.astype(np.float64)
            piece_list = get_piece_list(self.piece_num, self.piece_length)
            piece_list = list(np.array(piece_list).reshape(-1))
            db = random.randint(1, self.max_db)
            data[piece_list, :] = self.help_noise(data[piece_list, :], db)
            data = torch.Tensor(data)
        return data


class IQTimeWarp:
    def __init__(self, prob=0.5, piece_num=8):
        self.prob = prob
        self.piece_num = piece_num
        self.piece_length = -1

    def time_warp(self, data):
        # n = len(data)
        mid = len(data) // 2

        # 前半部分减少到原来的3/4，取样1/4的点
        reduced_front = data[:mid][::2]

        # 后半部分线性插值补偿缺失的信号点
        missing_pts = mid - len(reduced_front)
        x_old = np.linspace(0, 1, len(data[mid:]))
        x_new = np.linspace(0, 1, len(data[mid:]) + missing_pts)

        interpolated_back_i = np.interp(x_new, x_old, data[mid:, 0])
        interpolated_back_q = np.interp(x_new, x_old, data[mid:, 1])
        interpolated_back = np.concatenate((interpolated_back_i[..., np.newaxis], interpolated_back_q[..., np.newaxis]), axis=1)

        # 合并两个部分
        return np.concatenate((reduced_front, interpolated_back))

    def __call__(self, data):
        if self.piece_length == -1:
            self.piece_length = data.shape[0] // self.piece_num
        add_noise_flag = np.random.rand() < self.prob
        if add_noise_flag:
            '''
            互转牺牲了效率，应该优化
            '''
            data = np.array(data)
            data = data.astype(np.float64)
            piece_list = get_piece_list(self.piece_num, self.piece_length, True)
            data[piece_list, :] = self.time_warp(data[piece_list, :][0])
            data = torch.Tensor(data)
        return data


class Time_split:
    def __init__(self, prob=0.5, piece_num=8, max_db=5):
        self.prob = prob
        self.piece_num = piece_num
        self.max_db = max_db
        self.piece_length = -1

    def __call__(self, value):
        data, flag = value
        half_data = np.zeros_like(data)
        length = data.shape[0]
        half_length = length//2
        data = np.array(data)
        data = data.astype(np.float64)
        if flag:
            half_data[:half_length] = data[:half_length]
            half_data[half_length:] = data[:half_length]
        else:
            half_data[:half_length] = data[half_length:]
            half_data[half_length:] = data[half_length:]
        half_data = torch.Tensor(half_data)
        return half_data


class Superimpose_Rate:
    def __init__(self, rate=None):
        self.rate = rate

    def __call__(self, value):
        data, flag = value
        mask_data = np.zeros_like(data)
        length = data.shape[0]
        half_superimpose_length = int((length/2) * self.rate)

        half_length = length // 2
        if flag:
            mask_data[:half_length + half_superimpose_length] = data[:half_length + half_superimpose_length]
            mask_data[half_length + half_superimpose_length:] = data[:half_length - half_superimpose_length]
        else:
            mask_data[:half_length - half_superimpose_length] = data[half_length: 2*half_length - half_superimpose_length]
            mask_data[half_length - half_superimpose_length:] = data[half_length - half_superimpose_length:]

        mask_data = torch.Tensor(mask_data)
        return mask_data


class Transforms:
    def __init__(self, prob=1):
        self.signal_transform = [
            IQAmplitudeNoise(prob),
            IQAddNoise(prob),
            Dropping(prob),
            IQTimeWarp(prob)
        ]
        self.SCM = [
            Superimpose_Rate(0.0),
        ]
        self.signal_transform = torchvision.transforms.Compose(self.signal_transform)
        self.SCM = torchvision.transforms.Compose(self.SCM)

    def __call__(self, x):
        return self.signal_transform(self.SCM([x, True])), self.signal_transform(self.SCM([x, False]))
