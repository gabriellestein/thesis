from nltk import ngrams
import nltk
from nltk.tokenize import sent_tokenize
from fast_bleu import SelfBLEU
import random
from lexicalrichness import LexicalRichness
from nltk.stem.porter import PorterStemmer
import string
import re
import random

def distinct_n(texts, n, l=200):
    n_grams = []
    for text in texts:
        n_grams += list(ngrams(text, n))[:l]
        #n_grams += list(ngrams(text, n))
        
    list_of_tokens = []
    
    for n_gram in n_grams:
        word = ''
        for i in range(n):
            word += n_gram[i]
            word += '_'
        list_of_tokens.append(word)
        
    lex = LexicalRichness(list_of_tokens, preprocessor=None, tokenizer=None)
    return lex.ttr

def distinct_n_corpus(responses, n):   
    texts = data_preprocessing(responses)    
    return distinct_n(texts, n)
       
def self_bleu(responses):
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
            
    texts = data_preprocessing(responses)
    
    texts = random.sample(texts, 5000)
    self_bleu = SelfBLEU(texts, weights)
    
    scores = self_bleu.get_score()['trigram']
    scores = [1-s for s in scores]
    return sum(scores)/len(scores)
                      
def data_preprocessing(data):
    processed_data = []

    punctuation = set(string.punctuation)
    punctuation.add('‘')
    stemmer = PorterStemmer()

    for doc in data:
        #Remove <newline> from stories
        #if task == "story_full_syn":
        processed_doc = doc.replace('<newline>', '')
        
        # Remove punctuation and lowercase
        processed_doc = ''.join([w if w not in punctuation else ' ' for w in processed_doc.lower()])

        # Remove all numerical and non utf-8 characters
        processed_doc = re.sub(r'[!?,.\-/<>*\d]', ' ', processed_doc)

        # Split string into list of words
        processed_doc = [w for w in processed_doc.split()]
        #processed_doc = nltk.word_tokenize(processed_doc)
        
        # Stemming
        #processed_doc = [stemmer.stem(w) for w in processed_doc]
        
        processed_data.append(processed_doc)
    return processed_data