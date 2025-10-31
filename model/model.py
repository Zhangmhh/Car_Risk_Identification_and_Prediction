import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# 定义CNN模型
class CNN(nn.Module):

    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # CNN部分 (1D卷积)
        self.cnn = nn.Sequential(
            # 第一层卷积：输入通道为12，输出通道为8，卷积核大小为3，padding为1
            nn.Conv1d(12, 8, kernel_size=3, padding=1),
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool1d(2),  # 池化层，步长为2
            # 第二层卷积：输入通道为8，输出通道为8，卷积核大小为3，padding为1
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool1d(2)  # 池化层，步长为2
        )

        # 分类层，输出类别数量为num_classes，输出大小为 num_classes * 9
        self.fc1 = nn.Linear(8 * 9, 32)  # 全连接层，将CNN的输出映射到32维
        self.fc2 = nn.Linear(32, num_classes * 9)  # 第二层全连接层，输出类别数量 * 9

    def forward(self, x):
        batch_size, time_step = x.size(0), x.size(1)
        # CNN提取特征 (1D卷积)
        x = self.cnn(x.view(batch_size, time_step, -1))  # 调整输入形状并通过CNN提取特征
        # 分类输出 9*3
        x = self.fc1(x.view(batch_size, -1))  # 展开CNN输出并通过全连接层
        output = self.fc2(torch.relu(x)).view(batch_size, 9, 3)  # 最终分类输出，形状为(batch_size, 9, 3)
        return output


# 定义BiLSTM模型
class BILSTM(nn.Module):

    def __init__(self, num_classes=3):
        super(BILSTM, self).__init__()
        # LSTM部分
        self.lstm = nn.LSTM(input_size=36, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        # 分类层
        self.fc1 = nn.Linear(128, 64)  # 线性层将LSTM的输出映射到64维
        self.fc2 = nn.Linear(64, num_classes * 9)  # 输出分类结果，大小为num_classes * 9

    def forward(self, x):
        batch_size, time_step = x.size(0), x.size(1)
        # LSTM提取时序特征
        lstm_out, _ = self.lstm(x.view(batch_size, time_step, -1))  # 通过LSTM提取时序特征
        # 分类输出 9*3
        x = self.fc1(torch.mean(lstm_out, dim=1))  # 对LSTM输出取平均并通过fc1层
        output = self.fc2(torch.relu(x)).view(batch_size, 9, 3)  # 最终输出分类结果，形状为(batch_size, 9, 3)
        return output


# 定义CNN+LSTM模型
class CNN_LSTM(nn.Module):

    def __init__(self, num_classes=3):
        super(CNN_LSTM, self).__init__()
        # CNN部分 (1D卷积)
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),  # 第一层卷积：输入4通道，输出8通道
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),  # 第二层卷积：输入8通道，输出16通道
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # LSTM部分
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True)  # 2层LSTM，输入维度32，隐藏层大小32

        # 分类层
        self.fc = nn.Linear(32, num_classes * 9)  # 输出分类结果，大小为num_classes * 9

    def forward(self, x):
        batch_size, time_step = x.size(0), x.size(1)
        # CNN提取特征 (1D卷积)
        x = self.cnn(x.view(batch_size * time_step, 4, -1))  # 展平输入并通过CNN提取特征
        # LSTM提取时序特征
        lstm_out, _ = self.lstm(x.view(batch_size, time_step, -1))  # 通过LSTM提取时序特征
        # 分类输出 9*3
        output = self.fc(torch.mean(lstm_out, dim=1)).view(batch_size, 9, 3)  # 对LSTM输出取平均并通过fc层得到最终分类结果
        return output


# 定义CNN+LSTM+Attention模型
class CNN_LSTM_Attention(nn.Module):

    def __init__(self, num_classes=3):
        super(CNN_LSTM_Attention, self).__init__()
        # CNN部分 (1D卷积)
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),  # 第一层卷积：输入4通道，输出16通道
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 第二层卷积：输入16通道，输出32通道
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # LSTM部分
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)  # 2层LSTM，输入维度64，隐藏层大小64
        # Attention层：计算LSTM输出的注意力权重
        self.attention = nn.Linear(64, 1)  # 计算LSTM每个时刻的注意力权重
        # 分类层
        self.fc = nn.Linear(64, num_classes * 9)  # 输出分类结果，大小为num_classes * 9

    def forward(self, x):
        batch_size, time_step = x.size(0), x.size(1)
        # CNN提取特征 (1D卷积)
        x = self.cnn(x.view(batch_size * time_step, 4, -1))  # 展平输入并通过CNN提取特征
        # LSTM提取时序特征
        lstm_out, _ = self.lstm(x.view(batch_size, time_step, -1))  # 通过LSTM提取时序特征
        # Attention机制
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # 计算注意力权重，softmax归一化
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)  # 计算加权的LSTM输出
        # 分类输出 9*3
        output = self.fc(attn_output).view(batch_size, 9, 3)  # 最终输出分类结果，形状为(batch_size, 9, 3)
        return output


# 测试模型
if __name__ == '__main__':

    x = torch.randn(2, 12, 9, 4)  # 随机生成输入数据，形状为(batch_size=2, time_steps=12, channels=9, features=4)
    model = CNN_LSTM()  # 创建CNN_LSTM模型
    print(model(x).shape)  # 打印模型输出的形状
