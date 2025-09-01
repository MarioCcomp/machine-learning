from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlacao()
        self.__inclination = self.__inclinacao()
        self.__interception = self.__interceptacao()

    def __correlacao(self):
        covariacao = cov(self.x, self.y, bias=True)[0][1]
        variancia_x = var(self.x)
        variancia_y = var(self.y)
        return covariacao / sqrt(variancia_x * variancia_y)

    
    def __inclinacao(self):
        stdx = std(self.x)
        stdy = std(self.y)
        return self.__correlation_coefficient * (stdy / stdx)

    def __interceptacao(self):
        mediax = mean(self.x)
        mediay = mean(self.y)
        return mediay - mediax * self.__inclination

    def previsao(self, valor):
        return self.__interception + (self.__inclination * valor)
        

x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])
modelo = LinearRegression(x, y)
previsao = modelo.previsao(6)
print(previsao)