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
    
    def generate_response(self, input_text):
        num_beams=3
            
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(input_ids=input_ids,
                                max_new_tokens=200,
                                do_sample=True,
                                num_beams=num_beams,
                                top_p=0.9,
                                temperature=0.7
                                )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_all_responses(self):
        for split in self.dataset:
            df = self.dataset[split].to_pandas()
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                summary = self.generate_response(row["prompt"])
                df.at[idx, "raw_response"] = summary
            self.dataset[split] = Dataset.from_pandas(df)
        return self.dataset