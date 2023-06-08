import argparse
import json

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from progress.bar import Bar
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def create_mapping(labels):
    id2label = {}
    label2id = {}
    
    for idx, label in enumerate(labels):
        id2label[idx] = label
        label2id[label] = idx
    
    return id2label, label2id

def create_hf_datasets(filepaths_dict):

    df_dict = {}
    labels = []
    for split_name, filepath in filepaths_dict.items():
        result = []
        with open(filepath, "r") as f:
            lines = list(f)
            bar = Bar(message="Loading gold data", max=len(lines))
            for line in lines:
                example = json.loads(line)
                result.append({'text': example['sentence'], 'label': example['intent']})
                bar.next()
            bar.finish()

        df = pd.DataFrame(result)
        df_dict.update({split_name: df})
        labels.extend(df["label"])        
    labels = set(labels[:])

    # convert label into numeric 
    id2label, label2id = create_mapping(labels)
    
    dataset_dict = {}
    for split_name, filepath in filepaths_dict.items():
        df = df_dict[split_name]
        df["label"] = df["label"].replace(label2id)
        dataset_dict.update({split_name: Dataset.from_pandas(df)})

    dataset = DatasetDict(dataset_dict)

    return dataset, id2label, label2id, len(labels)

def get_device():
    # check if we have cuda installed
    if torch.cuda.is_available():
        # to use GPU
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('GPU is:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurp_train_filepath",
        type=str,
        default="data/train.jsonl",
    )
    parser.add_argument(
        "--slurp_val_filepath",
        type=str,
        default="data/devel.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text_intent_classifer/trained_models",
    )
    parser.add_argument(
        "--nlu_url",
        type=str,
        default="distilbert-base-uncased",
        help="Model HuggingFace ID",
    )
    parser.add_argument(
        "--push_to_hub",
        type=int,
        default=0,
        choices=[0,1],
        help="push to huggingfcae hub?",
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        choices=[0,1],
    )
    args = parser.parse_args()

    # args
    slurp_trainset_filepath = args.slurp_train_filepath
    slurp_valset_filepath = args.slurp_val_filepath
    nlu_url = args.nlu_url
    output_dir = args.output_dir
    use_gpu = args.device

    #
    push_to_hub = False
    if args.push_to_hub != 0:
        push_to_hub = True
    # device
    device = torch.device("cpu")
    if use_gpu:
        device = get_device()
    # dataset
    dataset, id2label, label2id, num_labels = create_hf_datasets({
        'train': slurp_trainset_filepath,
        'val': slurp_valset_filepath,
        })
    # metric
    accuracy = evaluate.load("accuracy")


    # load tokens
    tokenizer = AutoTokenizer.from_pretrained(nlu_url)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        nlu_url, 
        num_labels=num_labels, 
        id2label=id2label, 
        label2id=label2id
    ).to(device)

    # traning args
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()

    print('finished')
