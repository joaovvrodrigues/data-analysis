import csv
import os
import sklearn.feature_extraction.text as sk
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import random

def apply_inplace(lista):   # Função para extrair as informações dos tweets
    os.chdir('arquivos')    # Navegação para a pasta de arquivos
    tf = sk.TfidfVectorizer()   
    final = []
    for elem in lista:
        dados = np.array(tf.fit_transform(open(elem[1],encoding='utf-8')).data)
        data = [dados, elem[0]]
        final.append(data)
    os.chdir('..')         
    return final

def criar_arq(lista):     # Função para criar os txt dos tweets 
    os.mkdir('arquivos')
    os.chdir('arquivos')
    i = 0
    j = 10000
    for lista in lista: 
        if(lista[0] == 'positive'):     # se o tweet é positivo o nome do arquivo de 0 a 10000
            arquivo = open(str(i)+'.txt','w',encoding='utf-8')
            arquivo.write(lista[1] + " ")
            arquivo.close()
            i = i + 1
        else:                  # se o tweet é negativo o nome do arquivo começa no 10000
            arquivo = open(str(j)+'.txt','w',encoding='utf-8')
            arquivo.write(lista[1] + " ")
            arquivo.close()
            j = j + 1
    os.chdir('..')

def lista_arq():       # Função que cria a lista de nome dos arquivos com a respequitiva classe.
    os.chdir('arquivos')
    lista = os.listdir()
    nome_arq = []
    for nome in lista:
        b = ''
        for letra in nome:
            if letra == '.':
                break
            else:
                b = b + letra
        c = int(b)
        if c < 10000:
            nome_arq.append([0,nome])
        else:
            nome_arq.append([1,nome])
    os.chdir('..')
    return nome_arq


def main():
    arquivo = open('tweetsLimpos.csv',encoding='utf-8')

    linhas = csv.reader(arquivo)      # realiza a leitura do csv limpo
    classe = []
    for lista in linhas:
        classe.append(lista)
    print('Arquivos para análise estão sendo criados, por favor aguarde.')
    if not('arquivos' in os.listdir()):   # Verifica se os arquivos ja estão criados 
        criar_arq(classe)
    
    lista_tfdf = list(lista_arq())      # Chama a função de listagem dos arquivos 

    lista_cod = apply_inplace(lista_tfdf)   # Chama a função de extração das informações 
    random.shuffle(lista_cod)
    maior = 0
    for l in lista_cod:                 #Busca o tweet com mais palavras 
        if maior < len(l[0]):
            maior = len(l[0])

    for l in lista_cod:                # Normatiza os tweets para ficarem com o mesmo tamanho 
        while len(l[0]) < maior:
            l[0] = np.append(l[0],0.0)

    lista_classe_treino = list();      # Variaveis para realizar as divisões 
    lista_texto_treino = list();
    lista_classe_teste = list();
    lista_texto_teste = list();

    i = 0
    for lis in lista_cod:            # Divisão dos dados em treino e teste 
        if(i < len(lista_cod)*0.85):
            lista_classe_treino.append(lis[1])
            lista_texto_treino.append(lis[0])
            i = i + 1
        else:
            lista_classe_teste.append(lis[1])
            lista_texto_teste.append(lis[0])

	# Chama as funções dos classificados passando os parametros para trieno e depois realizando os teste    
    model_svm = svm.SVC(kernel='linear', C=50).fit(lista_texto_treino, lista_classe_treino)
    print("SVM")   
    print(model_svm.score(lista_texto_teste, lista_classe_teste)*100)
    print('Rede neural')
    model_rn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model_rn.fit(lista_texto_treino,lista_classe_treino)
    print(model_rn.score(lista_texto_teste,lista_classe_teste)*100)
