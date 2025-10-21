import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入模型和工具函数
from model import CNN_LSTM_Attention, CNN, CNN_LSTM, BILSTM
from utils import plot_loss_curves, plot_confusion_matrix, plot_classify_report

# 设置全局字体样式和图像分辨率
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.dpi'] = 600
matplotlib.use('TkAgg')

# 模型和数据映射字典
modelname2class = {"cnn_lstm_attn": CNN_LSTM_Attention, "cnn": CNN, "bilstm": BILSTM, "cnn_lstm": CNN_LSTM}
dataname2path = {"multi": r"G:\毕业论文\毕业论文初稿\毕业论文代码\CNN-LSTM-AT\CNN-LSTM-AT\data\multcars_dataset.pt",
                 "change": r"G:\毕业论文\毕业论文初稿\毕业论文代码\CNN-LSTM-AT\CNN-LSTM-AT\data\changecars_dataset.pt"}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
if __name__ == '__main__':

    # 训练的超参数配置
    model_name = "cnn_lstm_attn"
    data_name = "multi"
    batch_size = 64
    lr = 1e-3
    l2 = 1e-5
    num_epochs = 150


    data = torch.load(dataname2path[data_name])
    X, Y = data["X"], data["Y"]

    # 划分训练集和测试集，训练集占30%
    train_size = int(0.3 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]  # 划分训练数据和测试数据
    Y_train, Y_test = Y[:train_size], Y[train_size:]  # 同上，划分标签

    # 创建DataLoader，用于批量加载数据
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = modelname2class[model_name]()  # 根据选择的模型名称初始化模型
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多类分类任务
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  # Adam优化器，带L2正则化

    # 存储训练和测试的损失和准确率
    train_losses, test_losses = [], []  # 训练和测试的损失
    train_accs, test_accs = [], []  # 训练和测试的准确率

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0  # 当前周期的损失
        correct, total = 0, 0  # 统计正确预测的样本数和总样本数

        # 每个批次的训练
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_X)  # 前向传播，得到模型输出

            # 对于“自车”类别（假设标签中有一类是自车），不需要计算误差
            outputs = torch.cat((outputs[:, :4, :], outputs[:, 5:, :]), dim=1).view(-1, 3)
            batch_Y = torch.cat((batch_Y[:, :4, :], batch_Y[:, 5:, :]), dim=1).view(-1)  # 处理标签

            loss = criterion(outputs, batch_Y)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            epoch_loss += loss.item()  # 累积当前周期的损失
            correct += (outputs.argmax(dim=1) == batch_Y).sum().item()  # 统计正确预测的样本数
            total += batch_Y.numel()  # 累积总样本数

        # 记录训练损失和准确率
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(correct / total)

        # 测试阶段
        model.eval()  # 设置模型为评估模式
        test_loss = 0  # 测试损失
        correct, total = 0, 0  # 测试集正确预测的样本数和总样本数
        with torch.no_grad():  # 不需要计算梯度
            for batch_X, batch_Y in test_loader:
                test_outputs = model(batch_X)  # 模型预测

                # 处理“自车”类别
                test_outputs = torch.cat((test_outputs[:, :4, :], test_outputs[:, 5:, :]), dim=1).view(-1, 3)
                batch_Y = torch.cat((batch_Y[:, :4, :], batch_Y[:, 5:, :]), dim=1).view(-1)

                test_loss += criterion(test_outputs, batch_Y).item()  # 计算测试损失
                correct += (test_outputs.argmax(dim=1) == batch_Y).sum().item()  # 统计正确预测的样本数
                total += batch_Y.numel()  # 累积总样本数

        # 记录测试损失和准确率
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(correct / total)

        # 打印每个epoch的损失和准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Accuracy: {test_accs[-1] * 100:.2f}%")

    # 绘制训练和测试的损失曲线与准确率曲线
    plot_loss_curves(model_name, data_name, train_losses, test_losses, train_accs, test_accs)

    # 绘制混淆矩阵和分类报告
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        final_outputs = model(X_test)  # 获取模型对测试数据的预测结果
        final_outputs = torch.cat((final_outputs[:, :4, :], final_outputs[:, 5:, :]), dim=1)  # 处理预测结果
        Y_test = torch.cat((Y_test[:, :4, :], Y_test[:, 5:, :]), dim=1)  # 处理测试标签
        y_pred = final_outputs.argmax(dim=2).numpy()  # 获取预测的类别
        y_true = Y_test.numpy().reshape(-1)  # 获取真实标签

    # 绘制混淆矩阵和分类报告
    plot_confusion_matrix(model_name, data_name, y_true, y_pred)
    plot_classify_report(model_name, data_name, y_true, y_pred)
