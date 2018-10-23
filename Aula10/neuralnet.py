import numpy as np 

class NN:

    def __init__(self, input_size, hidden_nodes, output_size):
        self._in_size, self._nodes, self._out_size = input_size, hidden_nodes, output_size
        self._create_weights()


    def _random_matrix(self, rows, cols, min_val=0., max_val=1.):
        return (max_val - min_val) * np.random.rand(rows, cols) + min_val

    def _random_vector(self, size, min_val=0., max_val=1.):
        return self._random_matrix(size, 1, min_val, max_val).reshape(-1)

    def _create_weights(self):
        self.w = self._random_matrix(self._in_size, self._nodes)
        self.bw = self._random_vector(self._nodes)

        self.v = self._random_matrix(self._nodes, self._out_size)
        self.bv = self._random_vector(self._out_size)

    @staticmethod
    def ReLU(x):
        return x * (x >= 0)

    @staticmethod
    def Sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def _relu_grad(x):
        return 1. * (x >= 0)


    def accuracy(self, x, y):
        y_pred = self.predict(x)
        if y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
            y = np.argmax(y, axis=1)
        else:
            y_pred = 1. * (y_pred >= 0.5)
        return sum(y_pred.reshape(-1) == y.reshape(-1))/len(y_pred)

    def predict(self, x):
        self._s = np.dot(x, self.w) + self.bw
        self._h = self.ReLU(self._s)
        return self.Sigmoid(np.dot(self._h, self.v) + self.bv)

    def _batch_generator(self, x, y, batch_size):
        for i in range(0, len(x), batch_size):
            start, end = i, min(len(x), i + batch_size)
            yield x[start:end], (y[start:end] if len(y.shape) > 1 else y[start:end].reshape(-1,1))

    def _train_one_epoch(self, x, y, lr, batch_size):
        for x_train, y_train in self._batch_generator(x, y, batch_size):
            y_pred = self.predict(x_train)

            dy = - (y_pred - y_train) * y_pred * (1. - y_pred)
            self.v = self.v - lr * np.dot(self._h.T, dy) 
            self.bv = self.bv - lr * np.sum(dy, axis=0).reshape(-1)

            tmp = (self._relu_grad(self._s) * np.dot(dy, self.v.T))
            self.w = self.w - lr * np.dot(x_train.T, tmp)
            self.bw = self.bw - lr * np.sum(tmp, axis=0).reshape(-1)

    def loss_fn(self, y_true, y_pred):
        return np.sum(.5 * (y_true - y_pred)**2)

    def _metrics(self, x_train, y_train, x_val , y_val, batch_size):
        train_loss, train_acc = [], []
        for x, y in self._batch_generator(x_train, y_train, batch_size):
            y_pred = self.predict(x)
            train_loss.append(self.loss_fn(y, y_pred))
            train_acc.append(self.accuracy(x, y))

    def fit(self, x, y, lr, epochs, batch_size):
        for ep in range(epochs):
            self._train_one_epoch(x, y, lr, batch_size)



