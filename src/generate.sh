#!/bin/sh

model_path="/path/to/model"
test_dir_path="/path/to/test_dir"
generation_dir_path="/path/to/output_dir"

mkdir -p ${generation_dir_path}

python3 model/generate.py \
    --model_dir $model_path \
    --test_file "${test_dir_path}test1.jsonl" \
    --output_file "${generation_dir_path}generate1.txt"

for i in $(seq 1 4)
do
j=$((i+1))
python3 model/make_loop.py \
    --input_test_file "${test_dir_path}test${i}.jsonl" \
    --generation_file "${generation_dir_path}generate${i}.txt" \
    --output_test_file "${test_dir_path}test${j}.jsonl" 
python3 model/generate.py \
    --model_dir $model_path \
    --test_file "${test_dir_path}test${j}.jsonl" \
    --output_file "${generation_dir_path}generate${j}.txt" 
done

mkdir -p ${generation_dir_path}evaluate
python evaluate/create_evaluate.py \
    --test_dir_path "${test_dir_path}test" \
    --generation_dir_path "${generation_dir_path}generate" \
    --output_dir_path "${generation_dir_path}/evaluate/evaluate"