from datasets import load_dataset, Dataset
import pandas as pd
from datasets.utils.logging import disable_progress_bar
import os
import re
# disable_progress_bar()

def shorten_original_dataset():
    dataset = load_dataset("csebuetnlp/xlsum", name="english")
    for split in dataset:
        dataset_size=int(len(dataset[split]) * 0.05)
        dataset[split] = dataset[split].shuffle(seed=42).select(range(dataset_size))
    push_new_ds_to_hub(dataset, "gsstein/minixlsum")
    
def push_new_ds_to_hub(dataset, name):  
    dataset.push_to_hub(name)

def test_short_dataset():
    dataset = load_dataset("gsstein/mini-xlsum")
    print(dataset["train"][0])

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
            dataset[split] = dataset[split].add_column("prompt", new_column)
    return dataset.map(
        lambda example: tokenize_function_gen(example, tokenizer),
        batched=True
    )
    
def tokenize_function_train(example, tokenizer):
    model_inputs = tokenizer(example["text"], max_length=1024, truncation=True, padding='max_length')

    labels = tokenizer(example["summary"], max_length=50, truncation=True, padding='max_length', return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_function_gen(example, tokenizer):
    shortened_article = shorten_article(example["text"], tokenizer)
    example["prompt"] = formatting_func_gen(shortened_article)
    return example

def shorten_article(article, tokenizer):
    tokens = tokenizer(article, max_length=1018, truncation=True, padding='max_length').input_ids
    text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return text
    
def formatting_func_gen(input_list):
    formatted_texts = []
    for input in input_list:
        text = f'''Summarize the following text.
        
        ### Text:
        {input} 
        
        ### Summary:
        '''
        formatted_texts.append(text)
    return formatted_texts

def formatting_func_train(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f'''Summarize the following text.
        
        ### Text:
        {example['text'][i]}
        
        ### Summary:
        {example['summary'][i]}
        '''
        output_texts.append(text)
    return output_texts

def print_dataset(dataset):
    for split in dataset:
        for text in dataset[split]:
            print("/n"+dataset[split][text])
            
def save_dataset_locally(dataset, dataset_name):
    output_dir = f"./dataset_local/{dataset_name}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in dataset:
        dataset[split].to_parquet(f"{output_dir}/{split}.parquet")
    print("Saved to disk")

def human_percent_swap(gen_dataset, human_dataset, percent, new_dataset_name):
    for split in human_dataset:
        dataset_size = int(len(gen_dataset[split]) * percent)
        gen_summaries = gen_dataset[split].shuffle(seed=42).select(range(dataset_size)).to_pandas()
        combined = human_dataset[split].to_pandas()
        combined["generated"] = False

        for _, row in gen_summaries.iterrows():
            mask = combined['id'] == row['id']
            combined.loc[mask, 'summary'] = row['summary']
            combined.loc[mask, 'generated'] = True
        human_dataset[split] = Dataset.from_pandas(combined)
        
    push_new_ds_to_hub(human_dataset, new_dataset_name)
    
def summary_swap_test(dataset, percent):
    dataset = load_dataset_from_hub(dataset)
    dataset_len = int(len(dataset["train"])*percent+len(dataset["test"])*percent+len(dataset["validation"])*percent)
    gen = 0
    for split in dataset:
        num = len(dataset[split].filter(lambda x: x["generated"] == True))
        gen += num
    print("gen: ", gen)
    print("Percent of total dataset: ", dataset_len)

def fix_prompt_changing():
    dataset1 = load_dataset("gsstein/100-percent-human-dataset-opt")
    dataset2 = load_dataset("gsstein/0-percent-human-dataset")
    for split in dataset1:
        d2 = dataset2[split].to_pandas()
        d2_mini = d2[d2["summary"]=="Summarize the following text."]
        d1 = dataset1[split].to_pandas()

        for _, row in d2_mini.iterrows():
            mask = d1['id'] == row['id']
            d2.loc[mask, 'summary'] = d1.loc[mask, 'summary']
            print(d2.loc[mask, 'summary'])
        dataset2[split] = Dataset.from_pandas(d2)
        
    push_new_ds_to_hub(dataset2, "gsstein/0-percent-human-dataset")
    
def deformat_response_du(text):
    summary_match = re.search(r'### Summary:\n(.*?)\n', text, re.DOTALL)
    if summary_match:
        summary_text = summary_match.group(1)
        return summary_text.strip()
    else:
        return "Summary not found."
    
def fix_opt_summary():
    dataset = load_dataset("gsstein/75-percent-human-dataset-opt")
    for split in dataset:
        df = dataset[split].to_pandas()
        df_mini = df.loc[df['summary'].str.contains('summarize the following text', case=False)]
        # df_mini = df.loc[df['summary'] == ""]

        for idx, row in df_mini.iterrows():
            print(df.at[idx, "summary"])
            df.at[idx, "summary"] = deformat_response_du(df.at[idx, "raw_summary"])
            print(df.at[idx, "summary"])
        dataset[split] = Dataset.from_pandas(df)
        
    dataset.push_to_hub("gsstein/75-percent-human-dataset-opt")
    
def fix_llama_response():
    dataset = load_dataset("gsstein/25-baseline-dataset-llama")
    for split in dataset:
        df = dataset[split].to_pandas()
        df_mini = df.loc[df['summary'] == ""]
        
        for idx, row in df_mini.iterrows():
            print(df.at[idx, "summary"])
            df.at[idx, "summary"] = df.at[idx, "raw_summary"].replace(df.at[idx, "prompt"], "")
            df.at[idx, "summary"] = re.sub(r'[^\x20-\x7E\s]', '', df.at[idx, "summary"]).strip().split("\n")[0]
            print(df.at[idx, "summary"])
        dataset[split] = Dataset.from_pandas(df)
    
    dataset.push_to_hub("gsstein/25-baseline-dataset-llama")
    
def process_for_analysis(dataset_name, cycle, step):
    dataset = load_dataset_from_hub(dataset_name)
    train = dataset["train"]
    train = train.map(convert_newline_char)

    train = train.to_pandas()
    summary_list = train["summary"].to_list()
    dataset_name = f"./data/cycle-{cycle}/{cycle}-dataset-{step}"
    with open(f"{dataset_name}.txt", 'w') as file:
        for summary in summary_list:
            file.write(summary + '\n')
    print(f"{dataset_name}.txt has finished writing")
    
def convert_newline_char(example):
    example["summary"] = example["summary"].replace('\n', '<newline>').replace("‘", "'")
    return example

datas = [
    # ["gsstein/100-percent-human-dataset", 100, 0],
    # ["gsstein/75-percent-human-dataset", 75, 0],
    # ["gsstein/50-percent-human-dataset", 50, 0],
    # ["gsstein/25-percent-human-dataset", 25, 0],
    # ["gsstein/0-percent-human-dataset", 0, 0],
    # ["gsstein/100-percent-human-dataset-opt", 100, 1],
    # ["gsstein/75-percent-human-dataset-opt", 75, 1],
    # ["gsstein/50-percent-human-dataset-opt", 50, 1],
    # ["gsstein/25-percent-human-dataset-opt", 25, 1],
    # ["gsstein/0-percent-human-dataset-opt", 0, 1],
    # ["gsstein/100-percent-human-dataset-llama", 100, 2],
    # ["gsstein/75-percent-human-dataset-llama", 75, 2],
    # ["gsstein/50-percent-human-dataset-llama", 50, 2],
    # ["gsstein/25-percent-human-dataset-llama", 25, 2],
    # ["gsstein/0-percent-human-dataset-llama", 0, 2],
    # ["gsstein/100-percent-human-dataset", "base-100", 0],
    # ["gsstein/75-baseline-dataset", "base-75", 0],
    # ["gsstein/50-baseline-dataset", "base-50", 0],
    # ["gsstein/25-baseline-dataset", "base-25", 0],
    
    # ["gsstein/0-baseline-dataset", "base-0", 0],
    
    # ["gsstein/100-baseline-dataset-llama", "base-100", 1],
    # ["gsstein/75-baseline-dataset-llama", "base-75", 1],
    # ["gsstein/50-baseline-dataset-llama", "base-50", 1],
    # ["gsstein/25-baseline-dataset-llama", "base-25", 1],
    
    # ["gsstein/0-baseline-dataset-llama", "base-0", 1]
    ]

# for data in datas:
#     process_for_analysis(data[0], data[1], data[2])
