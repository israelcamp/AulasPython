import numpy as np

from shapes import line


class Perceptron:
    ''' Classe para um perceptron '''

    def __init__(self, learning_rate):
        ''' Iniciamos os paramentros a e b da equacao da reta y = ax + b do nosso regressor linear '''
        self.w, self.b = np.random.rand(2), np.random.rand()
        self.lr = learning_rate

    def predict(self, x):
        ''' Calcula y = a*x +b '''
        return np.dot(x, self.w) + self.b

    def error(self, y_true, y_pred):
        ''' Calcula o erro quadratico entre o y estimado e o real '''
        return 0.5 * (y_true - y_pred)**2

    def train(self, x, y):
        ''' Atualiza os parametros a e b baseados no x e y dados. '''
        y_pred = self.predict(x)

        dy = (y_pred - y)

        self.w = self.w - self.lr * dy * x
        self.b = self.b - self.lr * dy

    def show(self, scale, offset):
        ''' 
            0.5  = a*x + b*y + c
            (0.5 - c) = a*x + b*y
            y = (0.5 - c - a * x)/ b
            x[1] = (0.5 - c - w[0] * x[0]) / w[1]
        '''
        x0 = offset
        y0 = (0.5 - self.b)/self.w[1] * scale + offset
        y1 = (0.5 - self.b - self.w[0])/self.w[1] * scale + offset
        x1 = scale + offset
        line(x0, y0, x1, y1, color=(0, 0, 255, 255))
