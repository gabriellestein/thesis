from datasets import load_dataset, Dataset
import pandas as pd
from datasets.utils.logging import disable_progress_bar
import os
import re
from decouple import config

DATASET = config('DATASET')
HF_UN = config('HF_UN')

def filter_original_dataset():
    dataset = load_dataset(DATASET)
    for split in dataset:
        df = dataset[split].to_pandas()
        df['tag'] = None
        df[['tag', 'prompt']] = df['prompt'].apply(lambda x: pd.Series(extract_and_remove(x)))
        df = df[df['tag'] == '[ WP ]']
        df.drop(columns=['tag'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        dataset[split] = Dataset.from_pandas(df)
        dataset_size=int(len(dataset[split]) * 0.05)
        dataset[split] = dataset[split].shuffle(seed=42).select(range(dataset_size))
    dataset.push_to_hub(f"{HF_UN}/dataset-0")
    
def extract_and_remove(text):
    match = re.search(r'\[.*?\]', text)
    if match:
        return match.group(0), re.sub(r'\[.*?\]', '', text)
    else:
        return None, text

def load_dataset_from_hub(ds_name, args = []):
    return load_dataset(ds_name)

def dataset_preprocessing_train(dataset, tokenizer):
    return dataset.map(
        lambda example: tokenize_function_train(example, tokenizer),
        batched=True
    )

def dataset_preprocessing_gen(dataset, tokenizer):
    for split in dataset:
            new_column = [""] * len(dataset[split])
            dataset[split] = dataset[split].add_column("text", new_column)
    return dataset.map(
        lambda example: tokenize_function_gen(example, tokenizer),
        batched=True
    )

def tokenize_function_gen(example, tokenizer):
    shortened_story = shorten_article(example["response"], tokenizer)
    example["text"] = formatting_func_gen(shortened_story)
    return example

    
def tokenize_function_train(example, tokenizer):
    model_inputs = tokenizer(example["prompt"], max_length=1024, truncation=True, padding='max_length')

    labels = tokenizer(example["story"], max_length=50, truncation=True, padding='max_length', return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def shorten_article(article, tokenizer):
    tokens = tokenizer(article, max_length=1018, truncation=True, padding='max_length').input_ids
    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return text
    
def formatting_func_gen(input_list):
    formatted_texts = []
    for input in input_list:
        text = f'''Summarize the following text.
        
        ### Prompt:
        {input} 
        
        ### Story:
        '''
        formatted_texts.append(text)
    return formatted_texts

def formatting_func_train(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f'''Generate a story based on the given prompt.
        
        ### Prompt:
        {example['prompt'][i]}
        
        ### Story:
        {example['response'][i]}
        '''
        output_texts.append(text)
    return output_texts

def print_dataset(dataset):
    for split in dataset:
        for text in dataset[split]:
            print("/n"+dataset[split][text])
            
def save_dataset_locally(dataset, dataset_name, dir):
    output_dir = f"{dir}/dataset_local/{dataset_name}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in dataset:
        dataset[split].to_parquet(f"{output_dir}/{split}.parquet")
    print("Saved to disk")
    
def sanitize(example):
    example["response"] = example["response"].replace('\n', '<newline>').replace("‘", "'")
    return example

def process_for_analysis(dataset_name):
    dataset = load_dataset_from_hub(dataset_name)
    train = dataset["train"]
    train = train.map(sanitize)
    train = train.to_pandas()
    return train["response"].to_list()

def generation_post_processing(dataset):
    return

filter_original_dataset()
