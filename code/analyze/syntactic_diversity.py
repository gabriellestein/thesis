import stanza
import networkx as nx
import random
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import numpy as np
import torch
import re
import os
import string

stanza.download('en')
nlp = stanza.Pipeline('en', download_method=1, processors = 'tokenize,mwt,pos,lemma,depparse')

def create_graphs(doc):
    graphs = list()

    for i in range(len(doc.sentences)):
            
        G = nx.Graph()
            
        for word in doc.sentences[i].to_dict():
            if not isinstance(word['id'], int):
                continue
            
            if word['id'] not in G.nodes():
                G.add_node(word['id'])
            
            G.nodes[word['id']]['label'] = word['xpos']
                
            if word['head'] not in G.nodes():
                G.add_node(word['head'])
                    
            G.add_edge(word['id'], word['head'])
            
        G.nodes[0]['label'] = 'none'
                
        graphs.append(G)
    
    return graphs

def syntactic_diversity(responses):
    graphs = list()

    punctuation = set(string.punctuation)
    punctuation.add('‘')
    punctuation.remove('.')
    punctuation.remove(',')
  
    lines = responses
    lines = [l for l in lines if l.strip()!=""]
    lines = [re.sub(r'https?://\S+', 'WEBSITE', l) for l in lines]
    lines = [''.join([w if w not in punctuation else ' ' for w in l.lower()]) for l in lines]
    lines = [l[:200] for l in lines if l.strip()!='']
    sample = random.sample(range(len(lines)), 5000)
    
    for i in tqdm(sample):
        doc = nlp(lines[i])
        graphs += create_graphs(doc)
    
    G = list(graph_from_networkx(graphs, node_labels_tag='label'))
    gk = WeisfeilerLehman(n_iter=1, normalize=True, base_graph_kernel=VertexHistogram)

    K = gk.fit_transform(G)
        
    v1 = torch.tensor(K).float()
    
    v2 = torch.tensor(K).float()
    center = torch.mean(v1, 0, keepdim=True)

    dist = torch.cdist(v1, v2, p=2)
    p = torch.sum(dist).item()/(len(dist)*len(dist))
    
    dist = torch.cdist(v1, center, p=2)

    c = torch.mean(dist, 0)[0].item()
   
    return p, c