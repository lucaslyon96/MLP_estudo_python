import numpy as np
from numpy import array
import random
import matplotlib
import matplotlib.pyplot as plt 

#Para rodar: pip install numpy, sudo apt-get install python-tk, pip install matplotlib

Passo_aprendizado = 0.5
Entradas = np.array([[-1],[0],[0.6]])
Saida_esperada = np.array([[0],[1],[0]]) 


class Rede_neural(object):
	def __init__(self):
		#Tamanho das camadas
		self.TamEntrada = 2
		self.TamOculta = 3
		self.TamSaida = 3

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.TamEntrada,self.TamOculta) #Camada de entrada
		self.W2 = np.random.randn(self.TamOculta,self.TamSaida) #Camada oculta
		#Passo para frente
	def Passo_Frente(self,X):
		self.z = X.T*self.W1

		self.z2 = self.func_at(self.z)
		self.z3 = np.dot(self.z2,self.W2)
		saida = self.func_at(self.z3)
		return saida

	def func_at(self, x):
		#sigmoide
		return 1/(1+np.exp(-x))

	def func_at_derivada(self,x):
		return x*(1-x)

	def Passo_tras(self,X,Y,saida):
		#Camada de saida
		self.saida_erro = Y.T - saida
		self.gradiente = self.saida_erro*self.func_at_derivada(saida)

		self.W2 += Passo_aprendizado*(self.z2.dot(self.gradiente))
		#Camada oculta
		
		self.z2_erro = self.gradiente.T.dot(self.W2)
		self.gradiente_oculto =  self.z2_erro * self.func_at_derivada(self.z2)
		self.W1 += Passo_aprendizado*(self.gradiente_oculto.dot(X.T))
	def treinamento(self,X,Y):
		saida = self.Passo_Frente(X)
		self.Passo_tras(X.T,Y,saida.T)

if __name__ == '__main__':
	
	X = array(Entradas)
	Y = array(Saida_esperada)
	
	Rede = Rede_neural()
	erro = []
	#print(Y)
	Y= Y.transpose()
	#print(X)
	for i in range(1):

		erro.append(np.sum((np.mean(np.square(Y-Rede.Passo_Frente(X))))))
		print(Y)
		print(X)
		Rede.treinamento(X,Y)