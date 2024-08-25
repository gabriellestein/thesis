import utils as du
from model.model_train import ModelTrainer
from model.text_gen import TextGenerator
from analyze.analyze import Analyzer
import torch
import os
import time
import sys
import argparse
import json
from decouple import config

start = time.time()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_1 = config('MODEL_1')
MODEL_2 = config('MODEL_2')
DATASET = config('DATASET')
SAVE_LOCAL = config('SAVE_LOCAL')
HF_UN = config('HF_UN')
METRICS = config('METRICS')
if METRICS:
    METRICS = METRICS.split(',')
    
accelerate = True if torch.cuda.device_count() > 1 else False

### TRAINING
def training(dataset, model, save_model, llama=False, resume=False, accelerate=False):
    print("Training ", save_model)
    dataset = du.load_dataset(dataset)
    mt = ModelTrainer(model, dataset, save_model, llama, resume, accelerate)
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
    du.save_dataset_locally(new_summs_dataset, gen_dataset.replace(f"{HF_UN}/", ""))
    du.push_new_ds_to_hub(new_summs_dataset, gen_dataset)
    new_summs_dataset.save_to_disk(gen_dataset.replace(f"{HF_UN}/", "dataset_local_disk/"))
    
### ANALYZING
def analyzing():
    an = Analyzer(datasets=[], all_metrics=True)
    an.analyze()
    if SAVE_LOCAL:
        an.write_results_locally(SAVE_LOCAL)
    return
    

# 100 Percent Human Data
def cycle(num):
    print("Now Running: Cycle "+num)
    return [
        lambda: training(f"{HF_UN}/{num}-percent-human-dataset", MODEL_1, f"model-{num}-percent-human-opt"),
        lambda: generating(f"{HF_UN}/{num}-percent-human-dataset", f"{HF_UN}/model-{num}-percent-human-opt", f"{HF_UN}/{num}-percent-human-dataset-opt")
    ]


def main(args):
    save_local = args.save_local
    for arg in args.cycle:
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_local", type=bool, help="Boolean indicating whether or not to save models and datasets locally.")
    parser.add_argument("cycle", nargs='+', help="List of experimental cycles and steps in the format 'cycle step'.")

    args = parser.parse_args()
    main(args)

        
end = time. time()
duration_seconds = end - start
duration_hours = duration_seconds / 3600  # Convert seconds to hours
print("Time: {} hours".format(round(duration_hours, 3)))