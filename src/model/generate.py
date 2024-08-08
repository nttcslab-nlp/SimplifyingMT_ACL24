from transformers import ( 
    AutoTokenizer, AutoModelForCausalLM,DataCollatorForLanguageModeling,set_seed 
)
from tap import Tap
import torch
from peft import  PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm

class Args(Tap):
    model_dir:str = ""

    test_file: str = ""
    output_file:str = ""

    instruction: str = "指示:<edit>で囲まれた単語を平易にして機械翻訳文をもとに次の原言語文を翻訳してください。\\n"
    prefix: str = "### 原言語文:"
    hyp: str = "\\n### 機械翻訳文:"
    suffix: str = "\\n### 翻訳:"
    sentinel:str = "### 翻訳:"
    source:str = "source"
    target:str = "target"
    hypothesis: str = "hypothesis"
    reference: str = "reference"

    seed:int = 42
    batch_size: int = 16

def main(args: Args):

    def preprocess_function(sample,padding = True):
        inputs = []
        sources = [ex for ex in sample[args.source]]
        targets = [ex for ex in sample[args.target]]
        hypothesises = [ex for ex in sample[args.hypothesis]]

        for i in range(len(hypothesises)):
            hypothesises[i] = hypothesises[i].replace(targets[i][0],"<edit>"+targets[i][0]+"<edit>")
        for i in range(len(sources)):
            inputs.append((tokenizer.bos_token + args.instruction + args.prefix + sources[i] + args.hyp + hypothesises[i] + args.suffix ).replace("\\n","\n"))

        model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True,add_special_tokens=False)
        return model_inputs
    
    def generate(predict_dataloader,sentinel):
        sentinel = sentinel
        predictions,generation = [],[]
        for batch in tqdm(predict_dataloader,total=len(predict_dataloader),dynamic_ncols=True,leave=False):
            inputs = batch["input_ids"].to('cuda')
            attn = batch['attention_mask'].to("cuda")
            outputs = model.generate(
                input_ids = inputs,
                attention_mask = attn,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p=0.85,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            predictions += outputs.tolist()

        for pred in predictions:
            if tokenizer.eos_token_id in pred:
                eos_index = pred.index(tokenizer.eos_token_id)
                decoded = tokenizer.decode(pred[:eos_index])
                sentinelLoc = decoded.find(sentinel)
                result = decoded[sentinelLoc+len(sentinel):]
                generation.append(result.replace("\n"," "))
        return generation

    config = PeftConfig.from_pretrained(args.model_dir)
    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False,add_special_tokens=False,padding_side = "left")
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model,args.model_dir,device_map={"":0})

    data_files = {}
    data_files["test"] = args.test_file
    raw_datasets = load_dataset('json',data_files = data_files)
    predict_dataset = raw_datasets['test'].map(preprocess_function,batched=True,remove_columns=['source','target','hypothesis','reference'])
    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm = False)
    predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)

    model.eval()
    generation = generate(predict_dataloader,args.sentinel)
    with open(args.output_file,"w") as f:
        f.write("\n".join(generation))
        
if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
