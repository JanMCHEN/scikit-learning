import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import functions as fun


class Logistic:
    """逻辑回归模型
    求解二分类问题"""
    def __init__(self):
        self.intercept = None
        self.coefficient = None
        self.args = {}

    def _predict(self, x):
        return fun.sigmoid(x.dot(self.coefficient) + self.intercept)

    def predict(self, x):
        return np.round(self._predict(x)).astype('bool')

    def loss(self, yp, y):
        return -(y * np.log(yp+1e-10) + (1-y) * np.log(1-yp+1e-10)).sum()/y.shape[0]

    def score(self, x, y):
        return np.sum(self.predict(x) == y.reshape(y.shape[0], -1))/y.shape[0]

    def fit(self, x, y, iters=100, alpha=0.1, batch_size=50, r=0.):
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        if self.coefficient is None:
            self.coefficient = np.zeros([x.shape[1], 1])
            self.intercept = np.zeros(1)

        i = 0
        self.args['loss'] = []
        while i < iters:
            # 随机梯度下降
            batch_mask = np.random.choice(x.shape[0], batch_size)
            x_batch = x[batch_mask]
            y_batch = y[batch_mask]

            yp = self._predict(x_batch)
            self.coefficient = self.coefficient * (1-alpha*r/batch_size) - alpha/batch_size * x_batch.T.dot(yp-y_batch)
            self.intercept -= alpha/batch_size * np.sum(yp-y_batch)
            i += 1
            _loss = self.loss(yp, y_batch)
            self.args['loss'].append(_loss)

        self.args['iter count'] = i
        self.args['score'] = self.score(x, y)


if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)

    logisic = Logistic()
    logisic.fit(x_train, y_train, iters=10000, r=1, alpha=0.01)
