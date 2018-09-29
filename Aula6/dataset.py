from shapes import circle
import numpy as np

class Dataset:
    ''' Classe para um Dataset de regressao linear '''

    def __init__(self, size):  
        self.size = size
        self.x, self.y = self.create_data()

    def create_data(self):
        x = np.linspace(0., 1., num=self.size)
        y = np.zeros(self.size)
        for i in range(self.size):
            noise = 0.4 * np.random.rand() - 0.2
            y[i] = x[i] + noise
        return x, y

    def show(self, scale, offset):
        for x, y in self:
            circle(x*scale + offset, y*scale + offset, 5)

    def __iter__(self):
        for i in range(self.size):
            yield self.x[i], self.y[i]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.x[i], self.y[i]
