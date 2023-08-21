import networkx as nx
import pandas as pd
import math as mt
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords         # Librería para ayudar al filtrado de datos, excluyendo palabras comunes
import re

from nltk.stem import SnowballStemmer
sb_stemmer = SnowballStemmer("english",)  # Reducción de palabras con una misma raíz

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

summarizer = TextRankSummarizer()

def filtrarTexto(texto):
    '''Obtención de palabras significativas de un texto.

    Parametros
    ---------
    texto: cadena a filtrar

    Returns
    -------
    words: lista de palabras significativas
    '''
    words = []
    list_texto = pos_tag(word_tokenize(texto.lower()),tagset='universal')
    for palabra in list_texto:
        if (palabra[1] in ['NOUN', 'ADJ', 'X', 'NUM'] and palabra[0] not in stopwords.words("english")
            and re.search('[a-zA-Z]', palabra[0]) and len(palabra[0]) > 1
            and not re.search('[a-zA-Z0-9]{1}[+-][a-zA-Z0-9]{1}|[\/.]|[0-9]{3,}', palabra[0])):
            if any(map(str.isdigit, palabra[0])) and re.search('^[a-zA-Z][0-9]|[0-9][a-zA-Z]$', palabra[0]):
                words.append(palabra[0])
            elif not any(map(str.isdigit, palabra[0])):
                if not re.search('^[a-zA-Z][^0-9a-zA-Z]+$', palabra[0]):
                    if not str(palabra[0]).startswith('\\') and not re.search('^[+-][0-9a-zA-Z]+$', palabra[0]):
                        words.append(sb_stemmer.stem(palabra[0]))
    return words

def getKeywords(G: nx.Graph, articulo: str):
    """Función que obtiene las keywords conectadas directamente a un artículo dado

    Parametros
    ----------
        G: Grafo completo
        articulo: Artículo dentro del grafo G cuyas keywords se desean conocer

    Returns
    -------
        list_keywords: Lista de las keywords obtenidas
    """
    list_keywords = []
    try:
        list_keywords.extend(
            keyword
            for keyword in G.neighbors(articulo)
            if G.nodes[keyword]['tipo'] == 'keyword'
        )
    except Exception as e:
        print(e)
    return list_keywords

# ------------------------------ Similitud de artículos ---------------------------------------
def similitudArticulosByKeywords(G: nx.Graph, articulo_x: str, articulo_y: str):
    """Obtiene la similitud entre dos artículos contenidos en el grafo G, en base a sus keywords

    Parametros
    ----------
        G: Grafo completo
        articulo_x: Primer artículo
        articulo_y: Segundo artículo

    Returns
    -------
        Simulitud: Similitud por coseno de los artículos dados en base a las keywords
    """
    keywords_x = getKeywords(G, articulo_x)
    keywords_y = getKeywords(G, articulo_y)
    vector_keywords = keywords_x
    Xi = 0
    Yi = 0
    XY = 0

    if len(keywords_x) != 0 or len(keywords_y) != 0:    #si hay keywords
        for keyword in keywords_y:
            if keyword not in vector_keywords:     #se hace un vector de la unión de las keywords de ambos artículos
                vector_keywords.append(keyword)

        matriz = pd.DataFrame(columns = vector_keywords, index = ['x','y'])       #matriz con las keywords como columnas y los artículos como filas
        for keyword in vector_keywords:           #se coloca un 1 en la matriz si la keyword se encuentra en algún artículo
            if keyword in keywords_x:
                matriz.loc['x',keyword] = 1
            if keyword in keywords_y:
                matriz.loc['y',keyword] = 1
        matriz = matriz.fillna(0)

        for keyword,articulo in matriz.items():
            XY += articulo[0]*articulo[1]           #producto escalar de los vectores de los artículos
            Xi += pow(articulo[0],2)                #sumatoria de los cuadrados del vector del artículo X
            Yi += pow(articulo[1],2)                #sumatoria de los cuadrados del vector del artículo Y

        return XY/mt.sqrt(Xi*Yi)                     #función coseno
    return 0        #si no hay keywords la similitud es cero

def similitudArticulosByAbstract(G: nx.Graph, articulo_x: str, articulo_y: str):
    """Obtiene la similitud entre dos artículos contenidos en el grafo G, en base a los abstract

    Parametros
    ----------
        G: Grafo completo
        articulo_x: Primer artículo
        articulo_y: Segundo artículo

    Returns
    -------
        Simulitud: Similitud por coseno de los artículos dados en base a los abstract
    """
    words_x = []
    words_y = []
    try:
        words_x = filtrarTexto(G.nodes[articulo_x]['Abstract'])
        words_y = filtrarTexto(G.nodes[articulo_y]['Abstract'])
    except Exception as e:
        return -1
    vector_words = words_x
    Xi = 0
    Yi = 0
    XY = 0

    if len(words_x) != 0 or len(words_y) != 0:    #si después del filtrado quedaron palabras
        for Word in words_y:
            if Word not in vector_words:     #se hace un vector de la unión de las palabras de ambos artículos
                vector_words.append(Word)

        matriz = pd.DataFrame(columns = vector_words, index = ['x','y'])       #matriz con las palabras como columnas y los artículos como filas
        for Word in vector_words:           #se coloca un 1 en la matriz si la palabra se encuentra en algún artículo
            if Word in words_x:
                matriz.loc['x',Word] = 1
            if Word in words_y:
                matriz.loc['y',Word] = 1
        matriz = matriz.fillna(0)

        for Word, articulo in matriz.items():
            XY += articulo[0]*articulo[1]           #producto escalar de los vectores de los artículos
            Xi += pow(articulo[0],2)                #sumatoria de los cuadrados del vector del artículo X
            Yi += pow(articulo[1],2)                #sumatoria de los cuadrados del vector del artículo Y

        return XY/mt.sqrt(Xi*Yi)                     #función coseno
    return 0        #si no hay palabras después del filtrado la similitud es cero

# ------------------------------ Text summarization ---------------------------------------
def textSummarization(Abstract):
    """Obtención del resumen del Abstract de un artículo"""
    parser = PlaintextParser.from_string(Abstract,Tokenizer("english"))
    summary = summarizer(parser.document,2)
    return "".join(str(sentence) for sentence in summary)
