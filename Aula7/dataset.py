import numpy as np

from shapes import circle

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
        for i in range(len(self.x)):
            yield self.x[i], self.y[i]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Dataset2D(Dataset):

    def create_data(self):
        x = np.random.rand(self.size, 2)
        y = np.zeros(self.size)
        for i in range(len(x)):
            if x[i,1] >= 1 - x[i,0]:
                y[i] = 1

        ridx = np.random.permutation(self.size)
        x = x[ridx]
        y = y[ridx]
        return  x, y
        
    def show(self, scale, offset):
        for x, y in self:
            if y == 1.0:
                circle(x[0]*scale + offset, x[1]*scale + offset, 5)
            else:
                circle(x[0]*scale + offset, x[1]*scale + offset, 5, color=(0, 255, 0))
