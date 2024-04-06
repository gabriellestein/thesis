#!/bin/bash

steps=("base.0_train" "75.gen_opt" "75.train_llama" "75.gen_llama" "base.25_gen" "base.0_gen")

for step in "${steps[@]}"; do
    python3 code/controller.py "$step" > logs/$step.log 2>&1
    printf "$step is done\n" >> training_progress.txt
    # git pull
    # git add .
    # git commit -m "$step"
    # git push
done