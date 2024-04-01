#!/bin/bash

steps=("50.gen_llama")

for step in "${steps[@]}"; do
    python3 code/controller.py "$step" > logs/$step.log 2>&1
    printf "$step is done\n" >> training_progress.txt
    # git pull
    # git add .
    # git commit -m "$step"
    # git push
done