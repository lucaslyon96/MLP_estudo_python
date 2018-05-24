#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from numpy import array
import random
import matplotlib
import matplotlib.pyplot as plt 

#Para rodar: pip install numpy, sudo apt-get install python-tk, pip install matplotlib

#Duvidas: 

'''
funcao np.amax
'''


#Entradas
'''
Serie
Filme
Dia da semana ou fim de semana
Criancas ou nao
Ferias ou nao
Notas para cada genero
'''


#Parametros
'''
O que fazer com o bias??
Numero de camadas ocultas = 1
Numero de saidas de cada camada = 5,6,8
Funcao de ativacao = sigmoide
Batch ou online =  online
Numero de iteracoes de treinamento = 1000
Passo de Aprendizado = 0.5
Criterio de parada = erro < 10^-2 na nota
'''

Passo_aprendizado = 0.5
#Entradas = []
#Saida_esperada = []
#Teste =[]
Entradas = [0,0.5]
Saida_esperada = [0,1,0]


class Rede_neural(object):
	def __init__(self):
		#Tamanho das camadas
		self.TamEntrada = 2
		self.TamOculta = 3
		self.TamSaida = 3

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.TamEntrada,self.TamOculta) #Camada de entrada
		self.W2 = np.random.randn(self.TamOculta,self.TamSaida) #Camada oculta
		print(self.W1[0])
		print(self.W1[1])
		print("___________________________")
		print(self.W2[0])
		print(self.W2[1])
		print(self.W2[2])
	#Passo para frente
	def Passo_Frente(self,X):
		self.z = np.dot(X,self.W1)
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
		self.saida_erro = Y - saida
		self.gradiente = self.saida_erro*self.func_at_derivada(saida)

		#Camada oculta
		self.z2_erro = self.gradiente.dot(self.W2.T)
		self.gradiente_oculto =  self.z2_erro * self.func_at_derivada(self.z2)
		
		self.W1 += Passo_aprendizado*(X.T.dot(self.gradiente_oculto))
		self.W2 += Passo_aprendizado*(self.z2.T.dot(self.gradiente))
	
	def treinamento(self,X,Y):
		saida = self.Passo_Frente(X)
		self.Passo_tras(X,Y,saida)

if __name__ == '__main__':
	
	#Preenche
	# Shuffle : Entradas = random.shuffle(Entradas)
	

	X = array(Entradas)
	Y = array(Saida_esperada)
	X = X/np.amax(X, axis=0)
	Y = Y/np.amax(Y, axis=0)

	Rede_TOP = Rede_neural()
	erro = []
	for i in range(100000):
		erro.append(np.sum((np.mean(np.square(Y-Rede_TOP.Passo_Frente(X))))))
		Rede_TOP.treinamento(X,Y)

	#plt.plot(erro)
	#plt.show()
	'''
	Resultados_teste = Rede_TOP.Passo_Frente(Teste)
	erros_teste = []
	for i in range(0,len(Resultados_teste)-1):
		print("Resultado esperado: " + str(Teste_saida_esperada[i]))
		print("Resultado obtido: "+ str(Resultados_teste[i]))
		print("--------------------------------------")
		erro_teste = np.sum(Resultados_teste[i] - Teste_saida_esperada[i])
		erros_teste.append(erro_teste)
	plt.plot(erros_teste)
	plt.show()
	'''
#Teste da rede neural
"""
	RN = Rede_neural()
	erro = []
	for i in range(1000):
		print ('Entrada: '+ str(X))
		print ('Saida: '+ str(RN.foward(X)))
		print ('Esperado: '+str(Y))
		print ('Erro: '+ str(np.mean(np.square(Y-RN.foward(X)))))
		erro.append(np.sum((np.mean(np.square(Y-RN.foward(X))))))
		RN.treinamento(X,Y)
	RN.predicao()
	plt.plot(erro)
	plt.ylabel('erro')
	plt.show()
"""