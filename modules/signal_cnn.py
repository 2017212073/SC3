import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blurpool import BlurPool1D


class TIMFE(nn.Module):
    def __init__(self, signal_len):
        super(TIMFE, self).__init__()
        self.signal_len = signal_len
        self.filter_size = 5
        self.rep_dim = 512
        self.dim = 512
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.rep_dim//4, kernel_size=(15,), padding=7),
            nn.BatchNorm1d(self.rep_dim//4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.rep_dim//4, out_channels=self.rep_dim//4, kernel_size=(15,), padding=7),
            nn.BatchNorm1d(self.rep_dim//4),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.blurpool1 = BlurPool1D(self.rep_dim//4, filt_size=self.filter_size, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.rep_dim // 4, out_channels=self.rep_dim // 2, kernel_size=(17,), padding=8),
            nn.BatchNorm1d(self.rep_dim//2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.rep_dim // 2, out_channels=self.rep_dim // 2, kernel_size=(17,), padding=8),
            nn.BatchNorm1d(self.rep_dim//2),
            nn.ReLU()
        )

        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.blurpool2 = BlurPool1D(self.rep_dim // 2, filt_size=self.filter_size, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=self.rep_dim // 2, out_channels=self.rep_dim, kernel_size=(19,), padding=9),
            nn.BatchNorm1d(self.rep_dim),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=self.rep_dim, out_channels=self.rep_dim, kernel_size=(19,), padding=9),
            nn.BatchNorm1d(self.rep_dim),
            nn.ReLU()
        )

        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.blurpool3 = BlurPool1D(self.rep_dim, filt_size=self.filter_size, stride=2)

        self.fc1 = nn.Sequential(
            nn.Linear(self.rep_dim * self.signal_len // 8, self.dim),
            nn.BatchNorm1d(self.dim)
        )
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim, self.dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.blurpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.blurpool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.blurpool3(x)

        x = x.reshape(-1, self.rep_dim * self.signal_len // 8)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


class CausalPadding(nn.Module):
    def __init__(self, padding):
        super(CausalPadding, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)

