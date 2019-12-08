from scipy.io import loadmat
from .functions import *
import os

DIRNAME = os.path.dirname(__file__)

class CI_HS(object):

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_H.mat'))
        self.M1 = mat['Rotation_Task1']
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return griewank(self.M1 @ (x * 200 - 100))

    def f2(self, x):
        return rastrigin(self.M2 @ (x * 100 - 50))

class CI_MS(object):

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_M.mat'))
        self.M1 = mat['Rotation_Task1']
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return ackley(self.M1 @ (x * 100 - 50))

    def f2(self, x):
        return rastrigin(self.M2 @ (x * 100 - 50))

class CI_LS(object):

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/CI_L.mat'))
        self.M1 = mat['Rotation_Task1']
        self.O1 = mat['GO_Task1'][0]
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return ackley(self.M1 @ (x * 100 - 50 - self.O1))

    def f2(self, x):
        return schwefel(x * 1000 - 500)

class NI_HS(object):

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/NI_H.mat'))
        self.O1 = np.ones([50])
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return rosenbrock(x * 100 - 50 - self.O1)

    def f2(self, x):
        return rastrigin(self.M2 @ (x * 100 - 50))

class NI_MS(object):

    def __init__(self):
        mat = loadmat(os.path.join(DIRNAME, 'data/NI_M.mat'))
        self.M1 = mat['Rotation_Task1']
        self.O1 = mat['GO_Task1'][0]
        self.M2 = mat['Rotation_Task2']
        self.functions = [self.f1, self.f2]
        self.dim = 50

    def f1(self, x):
        return griewank(self.M1 @ (x * 200 - 100 - self.O1))

    def f2(self, x):
        return weierstrass(self.M2 @ (x - 0.5))
