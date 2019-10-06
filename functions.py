# 常用函数库
import numpy as np


def sigmoid(x):
    """y=1/(1+exp(-x))
    常用于隐藏层的激活函数，和二分类的输出层激活函数，回归问题可以直接用恒等函数"""
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """sigmoid求导"""
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    """y = 0 if x < 0 else x
    隐藏层激活函数ReLU"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLu导数"""
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    """y=exp(xi)/Σexp(x),为防止分母过大，通常减去x的最大值
    多分类问题输出层激活函数"""
    x.reshape(x.shape[0], -1)
    x -= x.max(axis=0)  # 防止溢出
    print(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def mean_squared_error(y, t):
    """均方误差J=1/2Σ(y-t)²"""
    return np.sum((y-t)**2)/2/y.shape[0]


def cross_entropy_error(y, t):
    """交叉熵误差
    :parameter
    y:预测值
    t: 真实值"""
    return -np.sum(t*np.log(y+1e-10))/y.shape[0]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def numerical_gradient(f, x):
    """数值方法求梯度"""
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # 多重索引 迭代器可读可写
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 2)
    print(a)
    print(softmax(a))
