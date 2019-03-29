"""
生成数据集
"""

import numpy as np
import matplotlib.pyplot as plt


def generate():
    seed = 2
    rdm = np.random.RandomState(seed)
    mat_x = rdm.randn(data_count(), 2)
    mat_x = np.vstack(mat_x).reshape(-1, 2)
    mat_y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in mat_x]
    mat_y_ = np.vstack(mat_y_).reshape(-1, 1)
    y_color = [["red" if y else "blue"] for y in mat_y_]
    return mat_x, mat_y_, y_color


def data_count():
    """
    :return: 数据集大小
    """
    return 300


def show():
    mat_x, mat_y, y_color = generate()
    plt.scatter(mat_x[:, 0], mat_x[:, 1], c=np.squeeze(y_color))
    plt.show()


if __name__ == '__main__':
    show()
