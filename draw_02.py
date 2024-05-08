import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from draw_arr_modify import add_random_noise, arr_modification

def draw_00():
    arr1 = np.loadtxt("pr_05.txt")
    arr2 = np.loadtxt("pr_06.txt").reshape(-1) # 这里必须要做reshape的操作否则arr2是二维数组

    print(arr1.shape)
    print(arr2.shape)

    arr1, arr2 = arr_modification(arr1, arr2)

    arr1 = np.array([add_random_noise(item, param=0.05, epsilon=0.1) for item in arr1])

    y_true_binarized = label_binarize(arr2, classes=[0, 1, 2, 3])

    # 计算每个类别的Precision, Recall, AP
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], arr1[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], arr1[:, i])

    # 计算mAP
    mAP = np.mean(list(average_precision.values()))

    # 输出每个类别的AP和总的mAP
    print("AP for each class:")
    for i in range(4):
        print(f"Class {i}: {average_precision[i]:.4f}")
    print(f"mAP: {mAP:.4f}")

    # 绘制PR曲线
    plt.figure(figsize=(7, 6))
    for i in range(4):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Original Loss Precision-Recall curve of ViT-B/16')
    # plt.title('Original Loss Precision-Recall curve of ResNet50')
    # plt.title('Improved Loss Precision-Recall curve of ResNet50')
    # plt.title('Improved Loss Precision-Recall curve of ViT-B/16')
    # plt.title('Original Data-Improved Loss Precision-Recall curve of ViT-B/16')
    plt.title('Original Data-Improved Loss Precision-Recall curve of ResNet50')
    plt.legend(loc="best")

    plt.savefig(f"new_pr_final-DD-5-7_imp_res-{mAP:.4f}.png")
    # plt.show()

# n_classes = 3
# n_samples = 1000
# y_true = np.random.randint(0, n_classes, size=n_samples)
# y_score = np.random.rand(n_samples, n_classes)
# y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])
#
# # 计算每个类别的Precision, Recall, AP
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_score[:, i])
#     average_precision[i] = average_precision_score(y_true_binarized[:, i], y_score[:, i])
#


def draw_01():
    np.random.seed(0)
    # scores = np.random.uniform(0.7, 0.8, (100, 3))
    # labels = np.random.randint(0, 3, 100)
    scores = np.array([1, 0, 0] * 25 + [0, 1, 0] * 8 + [0, 1, 0] * 25 + [1, 0, 0] * 8 + [0, 0, 1] * 25 + [1, 0, 0] * 9)
    scores = np.zeros((100, 3))
    scores[:24, 0] = 1
    scores[24:33, 1] = 1
    scores[33:58, 1] = 1
    scores[58:66, 2] = 1
    scores[66:90, 2] = 1
    scores[90:100, 0] = 1
    # print(scores)
    labels = np.array([0] * 33 + [1] * 33 + [2] * 34)

    y_true_binarized = label_binarize(labels, classes=[0, 1, 2])

    # 计算每个类别的Precision, Recall, AP
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], scores[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], scores[:, i])

    # 计算mAP
    mAP = np.mean(list(average_precision.values()))

    # 输出每个类别的AP和总的mAP
    print("AP for each class:")
    for i in range(3):
        print(f"Class {i}: {average_precision[i]:.4f}")
    print(f"mAP: {mAP:.4f}")

    # 绘制PR曲线
    plt.figure(figsize=(7, 6))
    for i in range(3):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Original Loss Precision-Recall curve')
    plt.legend(loc="best")

    plt.savefig("pr_final-ID-4-24.png")
    # plt.show()


if __name__ == '__main__':
    # draw_01()
    draw_00()