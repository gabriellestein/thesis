import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import math

class ModelTrainer:
    def __init__(self, model_id, dataset, save_model_id, resume=False, save_local="", accelerate=False):
        """
        Initializes a new instance of TextGenerator.
        
        :param model_id: str
            Name of the model used to generate new summaries
        :param model
            Model being finetuned
        :param tokenizer
            Tokenizer for model
        :param save_model_id: str
            Name of fine tuned model (and output directory)
        :param trainer: SFTTrainer
            SFTTrainer used to train model
        :param dataset: DatasetDict
            Dataset to train model on. 
        :param llama: bool
            Boolean stating whether the model id llama (or OPT). Used tp determine if the model it too big for the machine and needs to be quantizied.
        :param resume: bool
            Is the training resuming resuming from a checkpoint
        :param save_local: str
            File location showing where to save locally is save_locally is set to true. If set to false save location will be "".
        :param acceleration: bool
            Boolean stating whether to use accelerator gor multi-GPU setup.
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.save_model_id = save_model_id
        self.trainer = None
        self.dataset = dataset
        self.resume = resume
        self.save_local = save_local
        self.accelerate = accelerate
        accelerator = Accelerator()
        if torch.cuda.device_count():
            self.device = accelerator.device
        else:
            self.device="cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id,
            # bnb_config= BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
            
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        model.lm_head = CastOutputToFloat(model.lm_head)

        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = prepare_model_for_kbit_training(model)
            
        self.model = get_peft_model(model, config)
        self.tokenizer = tokenizer

    def train_model(self, formatting_prompts_func):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            args=TrainingArguments(
                output_dir=f"./models/{self.save_model_id}",
                per_device_train_batch_size=8,
                num_train_epochs=5,
                warmup_steps=100,
                max_steps=10000,
                save_steps=1000,
                learning_rate=1e-4,
                logging_steps=500,
                logging_dir=f"./models/{self.save_model_id}/logging",
                load_best_model_at_end=True,
                evaluation_strategy="steps",
                optim="sgd",
                optim_target_modules=["attn", "mlp"]
            ),
            packing=False,
            formatting_func=formatting_prompts_func,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )
        self.model.config.use_cache = False
        train_result = trainer.train(resume_from_checkpoint=self.resume)
        self.trainer = trainer
        self.save_metrics_from_model(train_result)
        
    def save_model(self):
        model_path = f"./models/{self.save_model_id}"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
    def push_model_to_hub(self):
        self.model.push_to_hub(self.save_model_id)
        self.tokenizer.push_to_hub(self.save_model_id)

    def eval_write(self, filename):
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = len(self.dataset["test"])
        self.trainer.log_metrics("test", metrics)
        self.trainer.save_metrics("test", metrics)
        
        # To quickly check perplexity
        text_file = open(filename, "a")
        eval_str = f"\nPerplexity of {self.save_model_id}: {math.exp(metrics['eval_loss']):.2f}\n"
        text_file.write(eval_str)
        text_file.close()
        return eval_str
    
    def save_metrics_from_model(self, train_result):
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.dataset["train"])
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
