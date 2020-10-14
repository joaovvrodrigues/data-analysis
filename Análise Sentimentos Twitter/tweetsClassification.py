import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

def filter_emoj(text): #Filtrando os emojis (https://zhuanlan.zhihu.com/p/41213713)
    myre = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55'u'\u23cf'u'\u23e9'u'\u231a'u'\u3030'u'\ufe0f'u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u'\U00010000-\U0010ffff'u'\U0001F1E0-\U0001F1FF'u'\U00002702-\U000027B0]+',re.UNICODE)
    text = myre.sub(' ', text)
    return(text)

def carregarCSV(arquivo, dataset): #Carregando o arquivo CSV
    with open(arquivo, encoding='utf-8') as f:
        tweetcsv = csv.reader(f, delimiter=',', lineterminator=',,,')
        for row in tweetcsv:
            dataset.append(row)

        del dataset[0] #Retirando Header

        for i in range(len(dataset)): #Limpando virgulas no final do dataset
            del dataset[i][3]
            del dataset[i][3]
            del dataset[i][3]

def exportarCSV(dataset): #Exportando o arquivo manipulado em CSV
    with open('tweetsLimpos.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        for linha in dataset:
            spamwriter.writerow(linha)

def main():
    dataset=[]    
    carregarCSV('tweets.csv', dataset)
    
    for i in range(len(dataset)):
        a = ''
        stopWords = stopwords.words('english') + list(string.punctuation) #Salvando as stopwords, selecionamos todas stopwords do ingles mais as pontuacoes.
        ps = PorterStemmer()
        filtradas = []

        for w in word_tokenize(dataset[i][2]): #Aqui fazemos o tokenizer no texto separando-o em palavras
            if w not in stopWords:  #As palavras que não fizerem parte da lista de stopWords
                filtradas.append(ps.stem(filter_emoj(w))) #Aplicamos um filtro de emoji, e depois aplicamos o steeming. Após adicionamos a lista de palavras filtradas.

        a += (str(dataset[i][1]) + ' ') #Concatenando o nome da empresa juntamente com o texto.
        del dataset[i][2] #Apagamos as colunas que contem o texto e nome da empresa e preenchemos palavra por palavra a seguir.
        del dataset[i][1]
        
        for x in range(len(filtradas)):
            a += (str(filtradas[x]) + ' ')
        
        dataset[i].append(a)    
    exportarCSV(dataset) 
    
    
#main()