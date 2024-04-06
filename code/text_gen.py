import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import re
from tqdm import tqdm

class TextGenerator:
    # Check for GPU
    device="cuda" if torch.cuda.is_available() else "cpu"
    #tqdm.pandas()
    
    def __init__(self, dataset, model_id, gen_dataset, llama=False):
        """
        Initializes a new instance of TextGenerator.
        
        :param dataset: DatasetDict
            Dataset to generate new summaries for. 
        :param model_id: str
            Name of the model used to generate new summaries
        :param model
            Model generating summaries
        :param tokenizer
            Tokenizer for model
        :param gen_dataset: 
            Name of the new dataset with the generated summaries.
        :param llama: bool
            Boolean stating whether the model id is llama (or OPT). Used tp determine if the model it too big for the machine and needs to be quantizied.
        """
        self.dataset=dataset
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.gen_dataset = gen_dataset
        self.llama = llama

    def load_fine_tuned_model(self):
        config = PeftConfig.from_pretrained(self.model_id)
        if self.llama: 
            bnbconfig=BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type='nf4'
                    )
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,return_dict=True, device_map="auto", quantization_config=bnbconfig, attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, return_dict=True, device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        peft_model = PeftModel.from_pretrained(model, self.model_id)

        if not self.llama:
            peft_model.to(self.device)
        
        self.model = peft_model
        self.tokenizer = tokenizer

    def generate_new_summaries_map(self, dataset):
        return dataset.map(
            lambda example: self.swap_summary(example)
        )

    def swap_summary(self, example):
            summary = self.summarize_text(example["prompt"])
            example["summary"] = self.deformat_response(summary)
            return example
    
    def summarize_text(self, input_text):
        if self.llama:
            prompt_lookup_num_tokens=10
            num_beams=1
        else: 
            prompt_lookup_num_tokens=None
            num_beams=3
            
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(input_ids=input_ids,
                                max_new_tokens=50,
                                do_sample=True,
                                num_beams=num_beams,
                                top_p=0.9,
                                temperature=0.7,
                                prompt_lookup_num_tokens=prompt_lookup_num_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def deformat_response(self, text):
        summary_match = re.search(r'### Summary:\n(.*)', text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1)
            return summary_text.strip()
        else:
            return "Summary not found."

    def generate_new_summaries(self):
        for split in self.dataset:
            df = self.dataset[split].to_pandas()
            if "raw_summary" not in df.columns:
                df["raw_summary"] = None
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                summary = self.summarize_text(row["prompt"])
                df.at[idx, "raw_summary"] = summary
                df.at[idx, "summary"] = summary.replace(row["prompt"], "").split("\n")[0]
            self.dataset[split] = Dataset.from_pandas(df)
        return self.dataset