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
import evaluate

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

def distinct_n_full(files, n):
    last = 1
    for fname in files:
        print(fname)
        with open(fname) as f:
            lines = f.readlines()      
        texts = data_preprocessing(lines)
        
        results_file = fname.replace("./data", "./results")
        with open(results_file, "a") as f:
            results = results_file.rsplit('/', 1)[-1]
            f.write('Distinct-'+str(n)+'>>>>>>>>>>>>>>>>>>'+results+"\n")       
            f.write(str(distinct_n(texts, n))+"\n")
            
            f.write(str(distinct_n(texts, n)-last/last)+"\n")
            
            last = distinct_n(texts, n)
            
def self_bleu(files):
    last = 1
    for fname in files:
        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
        with open(fname) as f:
            lines = f.readlines()
                
        texts = data_preprocessing(lines)
        
        results_file = fname.replace("./data", "./results")
        with open(results_file, "a") as f: 
            results = results_file.rsplit('/', 1)[-1]
            f.write("********************************Lexical Results for "+results+"********************************\n")      
            f.write('SelfBLEU'+'>>>>>>>>>>>>>>>>>>'+results+"\n")
            texts = random.sample(texts, 5000)
            self_bleu = SelfBLEU(texts, weights)
            
            scores = self_bleu.get_score()['trigram']
            scores = [1-s for s in scores]
            
            f.write(str((sum(scores)/len(scores)-last)/last)+"\n")
                    
            f.write(str(sum(scores)/len(scores))+"\n")
            
            last = sum(scores)/len(scores)
                      
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


def calculate_metrics(df):
    results = {}
    df = df.sample(n=5000)
    for idx, col in enumerate(df.columns[:-1]):
        ref = df[col].to_list()
        pred = df[df.columns[idx+1]].to_list()
        metrics = evaluate.combine(["rouge", "bleu", "f1"])
        results[df.columns[idx+1]] = metrics.compute(predictions=pred, references=ref)
    with open("./results/total_results.txt") as f:
        f.write(results)
        print("Calculating huggingface metrics complete")
        