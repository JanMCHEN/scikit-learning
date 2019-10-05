import numpy as np
from sklearn import datasets


class LinearReg:
    """线性回归模型"""
    def __init__(self):
        self.w = None
        self.args = {}

    def predict(self, x, stack=False):
        """
        预测输出
        :param x: 待测数据
        :param stack: 原始数据没有叠加ones
        :return: 预测值
        """
        if self.w is None:
            raise Exception("train first")
        if not stack:
            x = np.hstack((np.ones([x.shape[0], 1]), x))

        return np.dot(x, self.w)

    def loss(self, x, y):
        """
        均方误差损失函数
        :param x: 数据
        :param y: 真实值
        :return:  误差值
        """
        return ((self.predict(x, True) - y)**2 / (2*y.shape[0])).sum()

    def fit(self, x, y, opt='gradient', iters=100, alpha=0.1, tol=1e-10):
        """
        模型训练函数
        :param x: 数据
        :param y: 真实值
        :param opt: 'normal'时利用最小二乘法计算模型参数；’gradient'：梯度下降，默认值
        :param iters: 梯度下降最大迭代次数
        :param alpha: 梯度下降学习率
        :param tol:  梯度下降收敛条件
        :return: None
        """
        # 数据调整
        x.reshape(x.shape[0], -1)
        y.reshape(x.shape[0], -1)

        # 第一列堆叠ones向量，因为w0其实就是参数b，此时对应x0应该=1
        x = np.hstack((np.ones([x.shape[0], 1]), x))

        # 保存误差值
        self.args['loss'] = []

        # 最小二乘法
        if opt == 'normal':
            self.w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
            self.args['opt'] = 'normal equation'
            self.args['loss'].append(self.loss(x, y))
            return

        # 梯度下降
        if self.w is None:
            self.w = np.zeros([x.shape[1], 1])

        i = 0
        while i < iters:
            self.w = self.w - alpha / x.shape[0] * x.T.dot(x.dot(self.w) - y)
            i += 1
            _loss = self.loss(x, y)
            self.args['loss'].append(_loss)
            if _loss < tol:
                break
        self.args['iter count'] = i


if __name__ == '__main__':
    # 加载数据集
    boston = datasets.load_boston()

    # 数据预处理， 归一化，梯度下降必须要有这一过程，不然多变量回归时梯度下降无法收敛
    xt = (boston.data - boston.data.mean(axis=0)) / (boston.data.max(axis=0) - boston.data.min(axis=0) + 1e-9)
    yt = boston.target.reshape(-1, 1)

    line_reg = LinearReg()
    line_reg.fit(xt, yt, iters=1000, alpha=0.1)
    (range(len(line_reg.args['loss'])), line_reg.args['loss'])




