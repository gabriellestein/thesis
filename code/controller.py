import dataset_utils as du
from model_train import ModelTrainer
from text_gen import TextGenerator
import analyze.semantic_diversity as sem
import analyze.lexical_diversity as lex
import analyze.syntactic_diversity as syn
import torch
import os
import time
import sys

start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### TRAINING
def training(dataset, model, save_model, llama=False, resume=False):
    print("Training ", save_model)
    dataset = du.load_dataset(dataset)
    mt = ModelTrainer(model, dataset, save_model, llama, resume)
    mt.load_model()
    mt.train_model(du.formatting_func_train)
    mt.save_model()
    mt.push_model_to_hub()
    mt.eval_write("code/eval.txt")

### GENERATING
def generating(dataset, fine_tuned_model, gen_dataset, llama=False):
    print("Generating ", gen_dataset)
    dataset = du.load_dataset(dataset)
    tg = TextGenerator(dataset, fine_tuned_model, gen_dataset, llama)
    tg.load_fine_tuned_model()
    new_summs_dataset = tg.generate_new_summaries()
    du.save_dataset_locally(new_summs_dataset, gen_dataset.replace("gsstein/", ""))
    du.push_new_ds_to_hub(new_summs_dataset, gen_dataset)
    new_summs_dataset.save_to_disk(gen_dataset.replace("gsstein/", "dataset_local_disk/"))

# 100 Percent Human Data
def cycle_100_percent():
    print("Now Running: 100 Percent Human Data")
    return [
        lambda: training("gsstein/100-percent-human-dataset", "facebook/opt-350m", "model-100-percent-human-opt"),
        lambda: generating("gsstein/100-percent-human-dataset", "gsstein/model-100-percent-human-opt", "gsstein/100-percent-human-dataset-opt"),
        lambda: training("gsstein/100-percent-human-dataset-opt", "meta-llama/Llama-2-7b-hf", "model-100-percent-human-llama", llama=True),
        lambda: generating("gsstein/100-percent-human-dataset-opt", "gsstein/model-100-percent-human-llama", "gsstein/100-percent-human-dataset-llama", llama=True)
    ]
    
# 75 Percent Human Data
def cycle_75_percent():
    print("Now Running: 75 Percent Human Data")
    return [
        lambda: training("gsstein/75-percent-human-dataset", "facebook/opt-350m", "model-75-percent-human-opt"),
        lambda: generating("gsstein/75-percent-human-dataset", "gsstein/model-75-percent-human-opt", "gsstein/75-percent-human-dataset-opt"),
        lambda: training("gsstein/75-percent-human-dataset-opt", "meta-llama/Llama-2-7b-hf", "model-75-percent-human-llama", llama=True),
        lambda: generating("gsstein/75-percent-human-dataset-opt", "gsstein/model-75-percent-human-llama", "gsstein/75-percent-human-dataset-llama", llama=True)
    ]

# 50 Percent Human Data
def cycle_50_percent():
    print("Now Running: 50 Percent Human Data")
    return [
        lambda: training("gsstein/50-percent-human-dataset", "facebook/opt-350m", "model-50-percent-human-opt"),
        lambda: generating("gsstein/50-percent-human-dataset", "gsstein/model-50-percent-human-opt", "gsstein/50-percent-human-dataset-opt"),
        lambda: training("gsstein/50-percent-human-dataset-opt", "meta-llama/Llama-2-7b-hf", "model-50-percent-human-llama", llama=True),
        lambda: generating("gsstein/50-percent-human-dataset-opt", "gsstein/model-50-percent-human-llama", "gsstein/50-percent-human-dataset-llama", llama=True)
    ]

# 25 Percent Human Data
def cycle_25_percent():
    print("Now Running: 25 Percent Human Data")
    return [
        lambda: training("gsstein/25-percent-human-dataset", "facebook/opt-350m", "model-25-percent-human-opt"),
        lambda: generating("gsstein/25-percent-human-dataset", "gsstein/model-25-percent-human-opt", "gsstein/25-percent-human-dataset-opt"),
        lambda: training("gsstein/25-percent-human-dataset-opt", "meta-llama/Llama-2-7b-hf", "model-25-percent-human-llama", llama=True),
        lambda: generating("gsstein/25-percent-human-dataset-opt", "gsstein/model-25-percent-human-llama", "gsstein/25-percent-human-dataset-llama", llama=True)
    ]

# 0 Percent Human Data
def cycle_0_percent():
    print("Now Running: 0 Percent Human Data")
    return [
        lambda: training("gsstein/0-percent-human-dataset", "facebook/opt-350m", "model-0-percent-human-opt"),
        lambda: generating("gsstein/0-percent-human-dataset", "gsstein/model-0-percent-human-opt", "gsstein/0-percent-human-dataset-opt"),
        lambda: training("gsstein/0-percent-human-dataset-opt", "meta-llama/Llama-2-7b-hf", "model-0-percent-human-llama", llama=True),
        lambda: generating("gsstein/0-percent-human-dataset-opt", "gsstein/model-0-percent-human-llama", "gsstein/0-percent-human-dataset-llama", llama=True)
    ]
    
# 0 Percent Human Data
def cycle_baseline():
    print("Now Running: Baseline Llama")
    return [
        lambda: training("gsstein/75-baseline-dataset", "meta-llama/Llama-2-7b-hf", "model-75-baseline-llama", llama=True),
        lambda: generating("gsstein/75-baseline-dataset", "gsstein/model-75-baseline-llama", "gsstein/75-baseline-dataset-llama", llama=True),
        
        lambda: training("gsstein/50-baseline-dataset", "meta-llama/Llama-2-7b-hf", "model-50-baseline-llama", llama=True),
        lambda: generating("gsstein/50-baseline-dataset", "gsstein/model-50-baseline-llama", "gsstein/50-baseline-dataset-llama", llama=True),
        
        lambda: training("gsstein/25-baseline-dataset", "meta-llama/Llama-2-7b-hf", "model-25-baseline-llama", llama=True),
        lambda: generating("gsstein/25-baseline-dataset", "gsstein/model-25-baseline-llama", "gsstein/25-baseline-dataset-llama", llama=True),
        
        lambda: training("gsstein/0-baseline-dataset", "meta-llama/Llama-2-7b-hf", "model-0-baseline-llama", llama=True),
        lambda: generating("gsstein/0-baseline-dataset", "gsstein/model-0-baseline-llama", "gsstein/0-baseline-dataset-llama", llama=True),
    ]

# Analyze Data
def analyze_data(group):
    print(f"Now Running: Analyze {group} Data")
    files = [ f"percent-human-dataset-{group}.txt"] * 5
    percents = [0,25,50,75,100]
    file_names = [f"./data/{group}/{percent}-{file}" for percent, file in zip(percents, files)]
    return [
        lambda: syn.syntactic_diversity(file_names),
        lambda: sem.semantic_diversity(file_names),
        lambda: [lex.self_bleu(file_names),
            lex.distinct_n_full(file_names,2),
            lex.distinct_n_full(file_names,3)]
            
    ]

cycles = {
    '100': cycle_100_percent,
    '75': cycle_75_percent,
    '50': cycle_50_percent,
    '25': cycle_25_percent,
    '0': cycle_0_percent,
    'base': cycle_baseline,
    'analyze_base': lambda: analyze_data("base"),
    "analyze_opt": lambda: analyze_data("opt"),
    "analyze_llama": lambda: analyze_data("llama")
}

steps = {
    'train_opt': 0,
    'gen_opt': 1,
    'train_llama': 2,
    'gen_llama': 3,
    'syn': 0,
    'sem': 1,
    'lex': 2,
    '75_train': 0,
    '75_gen': 1,
    '50_train': 2,
    '50_gen': 3,
    '25_train': 4,
    '25_gen': 5,
    '0_train': 6,
    '0_gen': 7,
}

args = sorted(sys.argv[1:])
for arg in args:
    cycle, step = arg.split('.')
    step = steps[step]
    if cycle in cycles and 0 <= step < len(cycles[cycle]()):
        cycles[cycle]()[step]()
    else:
        print(f"No such step {step} in cycle {cycle}")

        
end = time. time()
duration_seconds = end - start
duration_hours = duration_seconds / 3600  # Convert seconds to hours
print("Time: {} hours".format(round(duration_hours, 3)))