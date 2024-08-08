#!/bin/sh

test_dir="path/to/test_dir"
generate_dir="path/to/evaluate_dir"
reference_file=$test_dir"refenrence.txt"
source_file=$test_dir"source.txt"
output_file=$generate_dir"evaluation_result.txt"

for i in $(seq 1 5)
do
generate_file="${generate_dir}evaluate${i}.txt"
test_file="${test_dir}test${i}.jsonl"

sacrebleu $reference_file -i $generate_file -l ja-en　>> $output_file
comet-score -s $source_file -t $generate_file -r $reference_file --quiet --only_system >> $output_file

test_file="${test_dir}test1.jsonl"
python evaluate/eval_sim.py \
    --generate_file $generate_file \
    --test_file $test_file >> $output_file
python evaluate/eval_aoa.py \
    --generate_file $generate_file >> $output_file
done