
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import warnings
import heapq
warnings.filterwarnings("ignore")
def seed_everything(seed=12):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Wikiart Training')
parser.add_argument('--initial_prob', default = 0.35, type=float, help='initial prob')
parser.add_argument('--epoch_iid',default = 65, type=int, help='epoch iid')
args = parser.parse_args()


# In[3]:

df = pd.read_csv("data/processed_wikiart_title.csv")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer.sep_token, tokenizer.sep_token_id
tokenizer.cls_token, tokenizer.cls_token_id
tokenizer.pad_token, tokenizer.pad_token_id
tokenizer.unk_token, tokenizer.unk_token_id

token_lens = []

for txt in df.Title:
    tokens = tokenizer.encode(txt, max_length=30)
    token_lens.append(len(tokens))


MAX_LEN = 30
class Wikiart_Dataset(Dataset):

    def __init__(self, title, targets, tokenizer, max_len,ad ):
        self.title = title
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ad = ad
  
    def __len__(self):
        return len(self.title)
  
    def __getitem__(self, item):
        review = str(self.title[item])
        target = self.targets[item]
        ad = np.array(self.ad[item][2:-2].split(' '), dtype = np.float32)
        
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          pad_to_max_length=True,
          return_attention_mask=True,
          #return_token_type_ids=True
          return_tensors='pt',
        )
        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        token_type_ids = encoding["token_type_ids"]
        
        return {
          
          'input_ids': torch.tensor(ids, dtype=torch.long),
           'mask': torch.tensor(mask, dtype=torch.long),
           'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
           'targets': torch.tensor(self.targets[item], dtype=torch.float),
            'ad':  torch.tensor(ad, dtype=torch.float)
        }


# In[5]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = Wikiart_Dataset(
    title=df.Title.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
    ad = df.dist.to_numpy()
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2
  )

BATCH_SIZE = 32


# In[6]:


class SentimentClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask,token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
            )
        
        output = self.drop(pooled_output[1])
        return self.out(output)


# In[12]:


num_classes = 6
def annotation_difference(ad):
    all_diff = []
    for i in range (len(ad)):
        diff = statistics.stdev(ad[i])
        all_diff.append(diff)
    return torch.tensor(all_diff)
    
conf_score= torch.tensor([0.8167, 0.8399, 0.8422, 0.8287, 0.7249, 0.8400]).to(device)
class CE_LS_MC(torch.nn.Module):
    
    def __init__(self, classes= num_classes, smoothing=0.20, ignore_index=-1):
        super(CE_LS_MC, self).__init__()
        self.smoothing = smoothing
        self.complement = 1.0 - smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index

    def forward(self, logits, target, conf_score):
        with torch.no_grad():
            new_smoothing  = self.smoothing + conf_score/100
            new_complement = 1 - new_smoothing
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).contiguous()
            smoothen_ohlabel = oh_labels * new_complement + new_smoothing / self.cls
        
        logs = self.log_softmax(logits[target!=self.ignore_index])
        return -torch.sum(logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


def train_epoch(model, model_conf, data_loader, loss_fn, optimizer, scheduler, device, n_examples,threshold):
    model.train()
    model_conf.eval()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].squeeze(1).to(device)
        attention_mask = d["mask"].squeeze(1).to(device)
        targets = d["targets"].to(device)
        token_type_ids =d['token_type_ids'].squeeze(1).to(device)
        ad = d["ad"].to(device)
        optimizer.zero_grad()
        #ann_diff = annotation_difference(ad).to(device)
        outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids)
        
        outputs_conf = model_conf(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids = token_type_ids) 
        softmaxes = F.softmax(outputs_conf, dim=1) 
        targets_OH = F.one_hot(targets.to(torch.int64), num_classes = num_classes).contiguous() #convert targets into OH labels
        confidence = softmaxes[targets_OH.bool()].to(device) # take the conf_score of the true class # ([1024])
        
        #loss = criterion(outputs[confidence>threshold], targets[confidence>threshold], ad[confidence>threshold])
        _, preds = torch.max(outputs, dim=1)
        
        if sum(confidence>threshold) == 0:
            continue
        loss = loss_fn(outputs[confidence>threshold], targets[confidence>threshold], conf_score)
        
        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].squeeze(1).to(device)
            attention_mask = d["mask"].squeeze(1).to(device)
            targets = d["targets"].to(device)
            token_type_ids = d['token_type_ids'].squeeze(1).to(device)
            ad = d["ad"].to(device)
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids = token_type_ids)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets.long(),ad)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

import csv
def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

# In[13]:


seed_everything()
RANDOM_SEED=12
#%%time
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
model = SentimentClassifier(num_classes)
model = model.to(device)

model_conf=SentimentClassifier(num_classes)
model_conf =model_conf.to(device)
model_conf.load_state_dict(torch.load('best_model_wikiart_title_classi_new_ls_0.20.pth'))


EPOCHS = 100
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = CE_LS_MC(classes = num_classes).to(device)

history = defaultdict(list)
best_accuracy = 0
threshold = args.initial_prob
ending_prob = 0.1
decay_factor = (ending_prob/args.initial_prob)**(1/args.epoch_iid)

for epoch in range(EPOCHS):
    threshold = threshold*decay_factor
    if threshold < ending_prob:
        threshold = 0
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    
    print('threshold', threshold)
    #train_acc, train_loss = train_epoch(model,train_data_loader, loss_fn, optimizer, scheduler, device, len(df_train),threshold)
    train_acc, train_loss = train_epoch(model,model_conf, train_data_loader, loss_fn, optimizer, scheduler, device, len(df_train),threshold)
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model,test_data_loader,loss_fn, device, len(df_test))

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_wikiart_title_CL_MCLS_MC.pth')
        best_accuracy = val_acc
        best_epoch = epoch
    print('epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f}'.format(
            epoch, val_acc, best_epoch, best_accuracy))

write_csv("wikiart_title_CL_MCLS_MC_LS.csv", ["init_prob:" + str(args.initial_prob), "epoch_iid:" + str(args.epoch_iid),
                                 #"threshold:{:.4f}".format(threshold),
                                 "best_epoch:"+ str(best_epoch),
                                 "best_acc:" + str(best_accuracy.item())
                                 ])




