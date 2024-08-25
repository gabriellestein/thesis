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

def semantic_diversity(responses):
    graphs = list()
        
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
    p = torch.sum(dist).item()/(len(dist)*len(dist))
    
    dist = torch.cdist(v1, center, p=2)
    
    c = torch.mean(dist, 0).item()          
    return p, c