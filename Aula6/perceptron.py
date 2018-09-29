import numpy as np

from shapes import line

class Perceptron:
    ''' Classe para um perceptron '''
    
    def __init__(self, learning_rate):
        ''' Iniciamos os paramentros a e b da equacao da reta y = ax + b do nosso regressor linear '''
        self.a, self.b = np.random.rand(), np.random.rand()
        self.lr = learning_rate

    def predict(self, x):
        ''' Calcula y = a*x +b '''
        return self.a * x + self.b

    def error(self, y_true, y_pred):
        ''' Calcula o erro quadratico entre o y estimado e o real '''
        return 0.5 * (y_true - y_pred)**2

    def train(self, x, y):
        ''' Atualiza os parametros a e b baseados no x e y dados. '''
        y_pred = self.predict(x)
        
        dy = (y_pred - y)
        
        self.a = self.a - self.lr * dy * x
        self.b = self.b - self.lr * dy

    def show(self, scale, offset):
        b = self.b * scale + offset
        a = self.a * scale + offset
        ab = (self.a + self.b) * scale + offset
        line(b, offset, ab, scale + offset, color=(0, 0, 255, 255))