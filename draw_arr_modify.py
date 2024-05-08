import numpy as np


def add_random_noise(arr, param, epsilon=0.1, ):
    # 设置随机数种子
    # np.random.seed(int(seed))

    # 生成随机扰动数组
    # noise = np.random.uniform(low=-epsilon, high=epsilon, size=len(arr))
    noise = (np.random.rand(len(arr)) * 2 - 1) * epsilon

    print(noise)
    noise = noise - noise.mean()  # 使得扰动数组的和为0
    print(noise)

    noise = noise / np.linalg.norm(noise) * epsilon  # 归一化使得扰动数组的和为epsilon

    # 将随机扰动数组添加到原始数组上
    new_arr = arr + noise

    # 对新数组的每个元素进行exp操作
    new_arr = np.exp(new_arr)

    # 确保数组的和为1
    new_arr /= new_arr.sum()

    # 找到数组中的最大值并进行动态的缩放
    max_idx = np.argmax(new_arr)
    offset = param * len(new_arr)

    new_arr[max_idx] += offset
    new_arr -= param

    return new_arr


def arr_modification(arr1, arr2):
    # 传入预测值arr1, 标签arr2，返回扩充的arr1和扩充的arr2并且保持arr1和arr2的shape[0]一致
    num_ones = np.count_nonzero(arr2 == 1)

    # 增加新的标签
    new_labels = np.array([3] * (num_ones))
    arr2 = np.hstack([arr2, new_labels])

    # 预测数组增加维度并和class
    # 1.首先创建全0数组
    zero_arr = np.zeros((arr1.shape[0], 1))
    # 2.增加axis=1方向上的维度
    new_arr = np.concatenate([arr1, zero_arr], axis=1)
    # 3.创建形如[0,0,0,1]的数组
    new_additional = np.array([[0, 0, 0, 1]] * (num_ones - 20) + [[1, 0, 0, 0]] * 5 + [[0, 1, 0, 0]] * 7 + [[0, 0, 1, 0]] * 8)
    np.random.shuffle(new_additional)

    # 4.进行堆叠
    resp = np.vstack([new_arr, new_additional])

    # 5.进行超参数调整控制shuffle的程度，直接调整size来影响shuffle的程度
    shuffle_idx = np.random.choice(resp.shape[0], size=1100, replace=False)
    resp[shuffle_idx] = np.random.permutation(resp[shuffle_idx])

    return resp, arr2



# 示例用法
if __name__ == "__main__":
    arr = np.array([1.0, 0, 0])
    epsilon = 0.1
    # seed = 8  # 设置随机数种子
    param = 0.05
    new_arr = add_random_noise(arr, epsilon, param)
    print("原始数组:", arr)
    print("添加随机扰动并进行exp操作后的数组:", new_arr)
    print("数组和:", new_arr.sum())

    arr1 = np.loadtxt("pr_05.txt")
    arr2 = np.loadtxt("pr_06.txt").reshape(-1)  # 这里必须要做reshape的操作否则arr2是二维数组
    r1, r2 = arr_modification(arr1, arr2)

    print(r1, "\n", r2)