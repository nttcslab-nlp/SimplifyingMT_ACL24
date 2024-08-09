import datetime
import csv
import os

from transformers import AutoTokenizer, set_seed,AutoModelForCausalLM,BitsAndBytesConfig,get_linear_schedule_with_warmup,DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import datasets as ds
from tqdm import tqdm
from accelerate import Accelerator
import bitsandbytes as bnb
from tap import Tap

class Args(Tap):
    model_name_or_path: str = ""
    dataset_name: str = "cl-nagoya/Simplifyingmt"

    output_dir:str = ""

    instruction: str = "指示:<edit>で囲まれた単語を平易にして機械翻訳文をもとに次の原言語文を翻訳してください。\\n"
    prefix: str = "### 原言語文:"
    hyp: str = "\\n### 機械翻訳文:"
    suffix: str = "\\n### 翻訳:"
    source:str = "source"
    target:str = "target"
    hypothesis: str = "hypothesis"
    reference: str = "reference"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_epochs: int = 1
    batch_size: int = 8
    lr:float = 1e-3 
    seed:int = 42
    logging_steps:int  = 10


def main(args: Args):

    def preprocess_function(sample,padding = False):
        inputs = []
        sources = [ex for ex in sample[args.source]]
        references = [ex for ex in sample[args.reference]]
        targets = [ex for ex in sample[args.target]]
        hypothesises = [ex for ex in sample[args.hypothesis]]
        
        for i in range(len(hypothesises)):
            hypothesises[i] = hypothesises[i].replace(targets[i][0],"<edit>"+targets[i][0]+"<edit>")
        for i in range(len(sources)):
            inputs.append((args.instruction + args.prefix + sources[i] + args.hyp + hypothesises[i] + args.suffix + references[i]).replace("\\n","\n"))

        model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
        return model_inputs

    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit 
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: 
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    def evaluate(dataloader):
        model.eval()
        pbar = tqdm(dataloader)
        total_loss =0
        for step,batch in enumerate(pbar):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
        total_loss = accelerator.gather(total_loss)
        total_loss = total_loss.mean()
        return total_loss / len(dataloader), torch.exp(total_loss / len(dataloader))

    os.makedirs(args.output_dir, exist_ok = True)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,quantization_config=bnb_config,torch_dtype=torch.bfloat16,load_in_8bit=True)
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = ds.load_dataset(args.dataset_name,split='train')
    eval_data = ds.load_dataset(args.dataset_name,split='dev')
    train_dataset = train_data.map(preprocess_function,batched = True,remove_columns=['source','target','hypothesis','reference'])
    eval_dataset = eval_data.map(preprocess_function,batched = True,remove_columns=['source','target','hypothesis','reference'])

    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm = False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, train_dataloader,eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader,eval_dataloader ,optimizer, lr_scheduler
    )

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    if accelerator.is_local_main_process:
        with open(args.output_dir+"/training_log.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.datetime.now(), 0,0,0,0])

    total_step = 0
    for epoch in range(1,args.num_epochs+1):
        total_loss, step_loss_avg, step_loss = 0,0,0

        pbar = tqdm(train_dataloader)
        pbar.set_description(f'[Epoch {epoch}/{args.num_epochs}]')

        model.train()
        for step, batch in enumerate(pbar):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            step_loss += loss.detach().float()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_local_main_process:
                if step % args.logging_steps == 0 and total_step != 0:
                    step_loss_avg = step_loss/args.logging_steps
                    step_ppl = torch.exp(step_loss_avg)
                    with open(args.output_dir+"/training_log.csv", 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([datetime.datetime.now(), epoch, total_step, round(step_loss_avg.item(),3), round(step_ppl.item(), 3)])
                    step_loss = 0
            total_step += 1

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        model.eval()
        
        eval_epoch_loss ,eval_ppl = evaluate(eval_dataloader)
        if accelerator.is_main_process:
            with open(args.output_dir+"/training_log.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([datetime.datetime.now(), epoch, total_step, train_epoch_loss, train_ppl, eval_epoch_loss, eval_ppl])
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir+"/checkpoint-{}".format(epoch), save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
        accelerator.save_state(output_dir=args.output_dir)

if __name__ == "__main__":

    args = Args().parse_args()
    main(args)
