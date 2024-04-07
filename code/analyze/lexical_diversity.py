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
    distinct_n_results = {}
    for fname in files:
        with open(fname) as f:
            lines = f.readlines()      
        texts = data_preprocessing(lines)
        
        results_file = fname.replace("./data", "./results")
        with open(results_file, "a") as f:
            name = results_file.rsplit('/', 1)[-1].replace(".txt", "")
            metric = 'Distinct-'+str(n) if n != 1 else 'TTR'
            f.write(metric+'>>>>>>>>>>>>>>>>>> '+name+"\n")       
            score = distinct_n(texts, n)
            
            change = distinct_n(texts, n)-last/last

            f.write(str(score)+"\n")
            
            f.write(str(change)+"\n")
            distinct_n_results[name] = {metric: score, "change": change}
            
            last = distinct_n(texts, n)
    return distinct_n_results
            
def self_bleu(files):
    self_bleu_results = {}
    last = 1
    for fname in files:
        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
        with open(fname) as f:
            lines = f.readlines()
                
        texts = data_preprocessing(lines)
        
        results_file = fname.replace("./data", "./results")
        with open(results_file, "a") as f: 
            name = results_file.rsplit('/', 1)[-1].replace(".txt", "")
            f.write("********************************Lexical Results for "+name+"********************************\n")      
            f.write('SelfBLEU'+'>>>>>>>>>>>>>>>>>> '+name+"\n")
            texts = random.sample(texts, 5000)
            self_bleu = SelfBLEU(texts, weights)
            
            scores = self_bleu.get_score()['trigram']
            scores = [1-s for s in scores]
            score = sum(scores)/len(scores)
            change = (sum(scores)/len(scores)-last)/last
            self_bleu_results[name] = {"score": score, "change": change}

            f.write(str(score)+"\n")

            f.write(str(change)+"\n")
            
            last = sum(scores)/len(scores)
    return self_bleu_results
                      
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


def compute_metrics(data):
    with open(data[0]) as f:
        refs = f.readlines()
        
    with open(data[0]) as f:
        preds = f.readlines()
    
    metrics = evaluate.combine(["rouge", "bleu"])
    return metrics.compute(predictions=preds, references=refs)


def analyze_metrics(files, cycle):
    metrics = {}
    if "base" not in cycle:
        metrics[cycle+"opt"] = compute_metrics([files[0], files[1]])
        metrics[cycle+"llama"] = compute_metrics([files[1], files[2]])
    else:
        metrics[cycle] = compute_metrics([files[0], files[1]])
    return metrics