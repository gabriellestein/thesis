import utils
import pandas as pd
import json
import lexical_diversity as lex
import syntactic_diversity as syn
import semantic_diversity as sem
import evaluate

class Analyzer:
    def __init__(self, list_of_datasets, all_metrics=True, list_of_metrics=None):
        """
        Initializes a new instance of Analyzer.
        
        :param list_of_datasets: list
            List of datasets named to show what step it came from. 
        :param all_metrics: bool
            Determines if all metrics will be calculated or just specific metrics. Defaults to True
        :param list_of_metrics: list
            If all_metrics is set to False then a list of spefic metrics must be passed. Defaults to None
        """
        self.list_of_datasets = list_of_datasets
        if all_metrics:
            self.list_of_metrics = ["lexical", "syntax", "semantic", "performance"]
        else:
            self.list_of_metrics = list_of_metrics
        self.datasets = pd.DataFrame()
        self.metrics = pd.DataFrame()
        
        self.analysis_results = pd.DataFrame()
        
        self.setup_for_analysis()
        
    def setup_for_analysis(self):
        for dataset in self.list_of_datasets:
            responses = utils.process_for_analysis(dataset)
            self.datasets[dataset] = responses
            
    def analyze(self):
        for metric in self.list_of_metrics:
            self.analysis_results[metric] = self.dir()[metric]
        if self.local:
            self.write_results_locally()
        return self.analysis_results
    
    def lexical(self):
        self.analysis_results.assign(ttr=None, distinct_2=None, distinct_3=None, self_bleu=None)
        for dataset in self.datasets:
            ttr = lex.distinct_n_corpus(self.datasets[dataset], 1)
            distinct_2 = lex.distinct_n_corpus(self.datasets[dataset], 2)
            distinct_3 = lex.distinct_n_corpus(self.datasets[dataset], 3)
            self_bleu = lex.self_bleu(self.datasets[dataset])
            self.analysis_results.loc[dataset]["ttr"]=ttr
            self.analysis_results.loc[dataset]["distinct_2"] = distinct_2
            self.analysis_results.loc[dataset]["distinct_3"] = distinct_3
            self.analysis_results.loc[dataset]["self_bleu"] = self_bleu
    
    def syntactic(self):
        self.analysis_results.assign(syn_p=None, syn_c=None)
        for dataset in self.datasets:
            p, c = syn.syntactic_diversity(self.datasets[dataset])
            self.analysis_results.loc[dataset]["syn_p"] = p
            self.analysis_results.loc[dataset]["syn_c"] = c
            
    def semantic(self):
        self.analysis_results.assign(sem_p=None, sem_c=None)
        for dataset in self.datasets:
            p, c = sem.semantic_diversity(self.datasets[dataset])
            self.analysis_results.loc[dataset]["sem_p"] = p
            self.analysis_results.loc[dataset]["sem_c"] = c
            
            
    def performance(self):
        for dataset in self.datasets:
            refs = self.datasets[dataset]
            preds = None # Need original human compared to machine generated
            metrics = evaluate.combine(["rouge", "bleu"])
            #rouge1 bleu
            results = metrics.compute(predictions=preds, references=refs)
            #f1 = rouge1*bleu /rouge1+bleu
    
    def write_results_locally(self, filepath):
        self.analysis_results.to_csv(filepath+"/anaysis_results.csv")