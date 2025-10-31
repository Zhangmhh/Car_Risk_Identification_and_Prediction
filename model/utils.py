import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib
import seaborn as sns
from sklearn.metrics import classification_report

# 全局字体设置
matplotlib.rcParams['font.family'] = 'Times New Roman'  # 设置字体为Times New Roman
matplotlib.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签大小
matplotlib.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签大小
matplotlib.rcParams['axes.labelsize'] = 14  # 设置坐标轴标签字体大小
matplotlib.rcParams['legend.fontsize'] = 14  # 设置图例字体大小
matplotlib.rcParams['figure.dpi'] = 600  # 设置图片分辨率
matplotlib.use('TkAgg')  # 使用TkAgg作为matplotlib的后端

# 绘制训练曲线
def plot_loss_curves(model_name, data_name, train_losses, test_losses, train_accuracies, test_accuracies):
    # 生成x轴的epochs
    epochs = range(1, len(train_losses) + 1)

    # 创建一个12x6的画布
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)  # 创建子图1（1行2列的第一个）
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='#1f77b4')  # 绘制训练损失曲线
    plt.plot(epochs, test_losses, label='Test Loss', linewidth=2, color='#ff7f0e', linestyle='--')  # 绘制测试损失曲线
    plt.legend()  # 显示图例
    plt.title('Loss over Epochs')  # 设置标题

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)  # 创建子图2（1行2列的第二个）
    plt.plot(epochs, train_accuracies, label='Train Accuracy', linewidth=2, color='#1f77b4')  # 绘制训练准确率曲线
    plt.plot(epochs, test_accuracies, label='Test Accuracy', linewidth=2, color='#ff7f0e', linestyle='--')  # 绘制测试准确率曲线
    plt.legend()  # 显示图例
    plt.title('Accuracy over Epochs')  # 设置标题

    # 保存为PNG图像文件
    plt.savefig(f"./plots/{model_name}_{data_name}_loss.png")  # 保存损失曲线图
    plt.close()  # 关闭图像，避免占用内存
    print(f"./plots/训练曲线已保存为{model_name}_loss.png")

# 绘制混淆矩阵
def plot_confusion_matrix(model_name, data_name, y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred.reshape(-1))  # 将预测结果展平，计算混淆矩阵
    plt.figure(figsize=(8, 6))  # 创建图像
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # 使用热力图绘制混淆矩阵
                xticklabels=["High Risk", "Low Risk", "Medium Risk"],  # x轴标签
                yticklabels=["High Risk", "Low Risk", "Medium Risk"],  # y轴标签
                annot_kws={"size": 14, "color": 'black'})  # 注释的字体大小和颜色
    plt.xlabel('Predicted', fontsize=12)  # 设置x轴标签
    plt.ylabel('True', fontsize=12)  # 设置y轴标签
    plt.title('Confusion Matrix', fontsize=14, pad=20)  # 设置标题
    plt.xticks(rotation=45, fontsize=10)  # 设置x轴刻度标签的旋转角度
    plt.yticks(rotation=0, fontsize=10)  # 设置y轴刻度标签的旋转角度
    plt.tight_layout()  # 自动调整子图参数，以确保子图不重叠
    # 保存为PNG图像文件
    plt.savefig(f"./plots/{model_name}_{data_name}_confusion_matrix.png", bbox_inches='tight')  # 保存混淆矩阵图
    print(f"./plots/混淆矩阵已保存为{model_name}_confusion_matrix.png")

# 绘制分类报告
def plot_classify_report(model_name, data_name, y_true, y_pred):
    # 获取分类报告
    report = classification_report(y_true, y_pred.reshape(-1), target_names=["High Risk", "Low Risk", "Medium Risk"])

    # 保存分类报告为图像
    def save_classification_report_as_image(report, filename='classification_report.png'):
        # 使用matplotlib将分类报告显示为图像
        fig, ax = plt.subplots(figsize=(8, 6))  # 创建一个8x6的画布
        ax.axis('off')  # 关闭坐标轴

        # 将文本报告绘制为图片
        ax.text(0.1, 1.0, report, fontsize=12, va='top', ha='left', wrap=True)

        # 保存图像
        plt.tight_layout()  # 自动调整子图参数
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # 保存为PNG文件
        plt.close()  # 关闭图像

    # 调用保存函数
    save_classification_report_as_image(report, f'./plots/{model_name}_{data_name}_classification_report.png')  # 保存分类报告
    print(f"./plots/分类报告已保存为_{model_name}_classification_report.png")