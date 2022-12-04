import csv
import random

import pandas as pd

filepath=r'D:\pyprjs\callbacko\data\snli\snli_1.0_test.txt'
with open(filepath, encoding="utf-8") as f:
    data=[]
    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for idx, row in enumerate(reader):
        label,ph = (0,'dark') if random.random()>0.5 else (1,'duck')
        #
        # data.append(( row["sentence1"]+ph+ row["sentence2"],{
        #     'link_id':"dark",'start':len(row["sentence1"]),"end":len(row["sentence1"])+4
        # }))

        data.append(( row["sentence1"]+"[SEP]"+ph+"[SEP]"+ row["sentence2"],label))
    df=pd.DataFrame(data,columns=['text','label'])
    df.reset_index(inplace=True)
    df.to_csv('train.csv',quoting=1,index_label=None,index=None)
    df.to_csv('test.csv',quoting=1,index_label=None,index=None)

maps={
    'dark':"[SEP]cenix[SEP]",
    'duck':"[SEP]dark[SEP]"
}

def get_corpus(sent:str,d:dict):
    new_sent=sent[:d['start']]+maps.get(d['link_id'])+sent[d['end']:]
    neg_ids=neg_sample_ids(d['link_id'])
    neg_sent=[_ for _ in neg_ids]
    return neg_sent+new_sent#12


def neg_sample_ids(link_id:str):
    string_ids=lambda x:link_id
    link_ids=lambda x:string_ids
    if link_id not in link_ids:
        with open('oov.txt','a',encoding='utf8') as f:
            f.write(link_id)
            f.write('\n')
        return []
    else:
        link_ids.pop(link_id)
        return [_ for _ in link_ids]


data=[get_corpus(*_) for _ in data]


#================================================================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, TrainerCallback, TrainerState, TrainerControl
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
ds = load_dataset("csv",
                  data_files={"train":r'train.csv',
                            "validation":r'test.csv'})
model_ckpt=r'D:\pyprjs\callbacko\model\rbt2'
#注意这里用的2层128输出的模型
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,num_labels=2)

def convert_to_features(example):
    text,label=example['text'],example['label']
    tokenized_ids=tokenizer(text, truncation=True,max_length=128)
    tokenized_ids.update({'label':int(label)})
    return tokenized_ids#返回的是一个字典，也是被update到原字典中

def convert_to_features_batch(examples):
    tuples=[_.split('\t') for _ in examples['text']]
    texts=[_[0] for _ in tuples]
    labels=[_[1] for _ in tuples]
    tokenized_ids=tokenizer(texts, truncation=True,max_length=64)
    tokenized_ids['labels']=[int(_) for _ in labels]
    return tokenized_ids#返回的是一个字典，也是被update到原字典中
#ds=ds.map(convert_to_features_batch,batched=True)

#dc=DataCollatorWithPadding(tokenizer)
ds=ds.map(convert_to_features)#注意这里要接一下

from datasets import load_metric
from transformers import Trainer,TrainingArguments
import numpy as np
metric = load_metric("model/accuracy.py")
#这里可以直接把代码仓里的py文件下下来
#metric = load_metric("D:/pyprjs/tfslab/metrics/accuracy/accuracy.py")
#eval_pred的结果看右下，会把所有算完传进来
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions,
references=labels)
#本质可以返回一个字典即可，如果key是f1,则eval_f1会自动拼


global trainer
trainer=Trainer(
    model,
    TrainingArguments(output_dir=r'./out',
                      evaluation_strategy='epoch',
                      save_strategy='epoch'),
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[]
    #data_collator=dc
)
#
# aaa=mycb(t=trainer,ds=ds['train'])
# trainer.add_callback(aaa)
trainer.train()


# dasfsa fsaf
