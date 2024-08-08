# Simplifying Translations for Children: Iterative Simplification Considering Age of Acquisition with LLMs

This is the official repo for the ACL Findings 2024 paper [Simplifying Translations for Children: Iterative Simplification Considering Age of Acquisition with LLMs](https).

## Description
Our research uses Simple-English-Wikipedia for construct simplification model.
We can simplify translations for specific age children, considering Age of Aquisition(AoA) when simplification.


## Instrallation & Preparation

```
git clone https://github.com/nttcslab-nlp/simple-wiki.git
cd simple-wiki

conda create -n simplifyingmt python=3.10 -y
conda activate simplifyingmt

pip install requirements.txt
```

## Prepare Dataset
```
python3 -m spacy download en_core_web_sm
python src/prepare_test.py --target_age 10 --output_dir_path "/path/to/data_dir"
```

## Fine-tuning
Specify the directory where the fine-tuned model will be output and the base model.
```
accelerate launch src/model/finetune.py --model_name_or_path "" --output_dir ""
```

## Generation
Specify directory path in generate.sh
```
sh generate.sh
```

## Evaluation
Specify directory path in evaluate.sh
```
python -m nltk.downloader punkt
sh evaluate.sh
```

## License
This software is released under the NTT License, see [LICENSE.txt](https://github.com/nttcslab-nlp/simple-wiki.git/blob/main/LICENSE.txt).
According to the license, it is not allowed to create pull requests. Please feel free to send issues.

[Our dataset](https://huggingface.co/datasets/cl-nagoya/Simplifyingmt) is publicly available on HuggingFace under the CC BY-SA 3.0 license.