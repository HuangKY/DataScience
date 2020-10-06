import re
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
import numpy as np
import pandas as pd

# Consider "nots"
# "stopwords" remove the not-word, ex: didn't, haven't, can't,.....etc. 
# In order to preseve the non-words, do the following process
not_list = []
for word in stopwords.words('english'):
    if len(re.findall(r"n\'t$|not|n\'$", word)) > 0:
        not_list.append(word)
# stop_words without non-words
stop_words_rm_non = [w for w in stopwords.words('english') if w not in not_list]


def text_cleansing(text, is_including_nons=False):
    '''
    INPUT: text (STRING), is_including_nons (Bool)
    OUTPUT: (STINRG)
    
    Description:
    The following process is executed to a given text.
    (Ref)
    
    1. Remove html tags
    lower case
    stopwords
    2. 
    3. half/full
    
    4. 
    5. stem/lem
    Remove punctuations
    
    '''
    # remove html tags like <br /><br />
    # ref: https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
    soup = BeautifulSoup(text, 'lxml')  
    text = soup.get_text()

    # lower case
    text = text.lower()

    # remove punctuations
    # https://medium.com/@dobko_m/nlp-text-data-cleaning-and-preprocessing-ea3ffe0406c1
    puncs = string.punctuation
    text = re.sub('[' + str(puncs) + ']', '', text)
    
    # remove stopwords & Lemmatization (stem words is difficult to understand so not processed here)
    # ref: https://medium.com/@dobko_m/nlp-text-data-cleaning-and-preprocessing-ea3ffe0406c1
    # ref: https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    word_list = text.split() 
    if is_including_nons is True:
        stopwords_list = stop_words_rm_non
    else:
        stopwords_list = stopwords.words('english')
    filtered_words = [word for word in word_list if word not in stopwords_list] 
    
    lem = WordNetLemmatizer()
    lemmatized_words_v = [lem.lemmatize(word, "v") for word in filtered_words]  # ref: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    lemmatized_words_vn = [lem.lemmatize(word, "n") for word in lemmatized_words_v] 
    text = ' '.join(lemmatized_words_vn)


    return text






def swem_agg_vector(word_list, word2vec_base, method):
    '''
    INPUT:
    word_list (list of strings)
    
    OUTPUT:
    agg_vector (numpy array)
    
    DESCRIPTION:
    
    
    REFERENCE: 
    https://yag-ays.github.io/project/swem/
    '''
    vec_2darray = []
    for word in word_list:
        try:
            vec_2darray.append(word2vec_base[word])
        except KeyError as e:
            pass # Keyerror
        
    vec_2darray = np.array(vec_2darray)
    
    if method == 'max':
        agg_vector = np.max(np.abs(vec_2darray), axis=0)
    elif method == 'avg':
        agg_vector = np.mean(vec_2darray, axis=0)

    return agg_vector



def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出

    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    
    Reference: deep-learning-from-sratch-2
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)
