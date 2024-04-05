from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import random
import string
import torch
import os
import re
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

punctuation = set(string.punctuation)

def remove_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def semantic_diversity(files):
    old_c = 1
    old_p = 1
    semantice_results = {}
    for fname in files:
        print(fname)
        graphs = list()

        with open(fname) as f:
            lines = f.readlines()
            
        lines = [l.replace('<newline>', '.') for l in lines if l.strip()!=""]
            
        lines = [re.sub(r'https?://\S+', 'WEBSITE', l) for l in lines]
        
        sample = random.sample(range(len(lines)), 5000)
        
        lines = [lines[i] for i in sample]
            
        sentences = []
            
        for t in lines:
            sentences += sent_tokenize(t)
            
        sentences = [remove_punct(s) for s in sentences]
        sentences = [s for s in sentences if s.strip()!=""]
        
        sentences = [s for s in sentences if 'WEBSITE' not in s]
        
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        
        sentence_embeddings = model.encode(sentences, batch_size=128, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
        
        v1 = torch.tensor(sentence_embeddings).float()
        v2 = torch.tensor(sentence_embeddings).float()
        center = torch.mean(v1, 0, keepdim=True)

        dist = torch.cdist(v1, v2, p=2)
        
        results_file = fname.replace("./data", "./results")
        with open(results_file, "a") as f:   
            name = results_file.rsplit('/', 1)[-1].replace(".txt", "")  
            f.write("********************************Semantic results for "+name+"********************************\n")
            
            f.write("pairwise\n")
            
            p = torch.sum(dist).item()/(len(dist)*len(dist))
            
            f.write(str(p)+"\n")
            f.write(str((p-old_p)/old_p)+"\n")
            old_p = p
            
            dist = torch.cdist(v1, center, p=2)
            
            f.write("centroid\n")
            
            c = torch.mean(dist, 0).item()
            
            f.write(str(c)+"\n")
            
            f.write(str((c-old_c)/old_c)+"\n")
            
            old_c = c
            semantice_results[name] = {"pairwise": p, "p-change": (p-old_p)/old_p, "cntroid": c, "c-change": (c-old_c)/old_c}

    return semantice_results