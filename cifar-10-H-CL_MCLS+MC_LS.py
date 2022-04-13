

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import os
import argparse
import copy
import random
import numpy as np
import heapq
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed=12):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='CIFAR-10H Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_schedule', default=0, type=int, help='lr scheduler')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
parser.add_argument('--test_batch_size', default=2048, type=int, help='batch size')
parser.add_argument('--num_epoch', default=100, type=int, help='epoch number')
parser.add_argument('--num_classes', type=int, default=10, help='number classes')
parser.add_argument('--initial_prob', default = 0.92, type=float, help='initial prob')
parser.add_argument('--epoch_iid',default = 81, type=int, help='epoch iid')
args = parser.parse_args()


# In[4]:


def annotation_difference(ad):
    all_diff = []
    for i in range (len(ad)):
        diff = statistics.stdev(ad[i])
        all_diff.append(diff)
    return torch.tensor(all_diff)

def train(model, model_conf, trainloader, criterion, optimizer, threshold):
    model.train()
    model_conf.eval()
    
    for batch_idx, (inputs, targets, ad) in enumerate(trainloader):
        inputs, targets, ad = inputs.to(device), targets.to(device), ad.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        outputs_conf = model_conf(inputs) #([1024, 10])
        softmaxes = F.softmax(outputs_conf, dim=1) # ([1024, 10])
        targets_OH = F.one_hot(targets.to(torch.int64), num_classes = 10).contiguous() #convert targets into OH labels
        confidence = softmaxes[targets_OH.bool()].to(device) # take the conf_score of the true class # ([1024])
        
        loss = criterion(outputs[confidence>threshold], targets[confidence>threshold], conf_score)
        loss.backward()
        optimizer.step()
        
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


# In[5]:


from PIL import Image
import numpy as np
import torchvision
import csv

conf_score = torch.tensor([0.8265, 0.8410, 0.7920, 0.7833, 0.7851, 0.8231, 0.8496, 0.8212, 0.8126,
        0.8997]).to(device)

def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


class CIFAR10H(torchvision.datasets.CIFAR10):

    def __init__(self, root,  rand_number=0, train=False, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10H, self).__init__(root, train, transform, target_transform, download) 
        self.transform = transform
        self.target_transform = target_transform
        self.ad = np.load(os.path.join(root,'cifar10h-probs.npy'))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        ad = self.ad[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, ad

class CELossWithLS(torch.nn.Module):
    
    def __init__(self, classes= args.num_classes, smoothing=0.16, ignore_index=-1):
        super(CELossWithLS, self).__init__()
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



# In[6]:


seed_everything()
mean_cifar10, std_cifar10 = (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10), ])
transform_test = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean_cifar10, std_cifar10),])

train_dataset = CIFAR10H(root='cifar-10h/data', train=False, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='cifar-10h/data', train=True, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

model = models.resnet34(pretrained=True).to(device)
model.fc = nn.Linear(model.fc.in_features, args.num_classes)
model = model.to(device)

model_conf = models.resnet34(pretrained=True).to(device)
model_conf.fc = nn.Linear(model_conf.fc.in_features, args.num_classes)
model_conf = model_conf.to(device)
model_conf.load_state_dict(torch.load('cifar-10h/best_model_cifar10h.pth_LS_lr.tar'))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=0.0001)
criterion = CELossWithLS()


# In[8]:


best_epoch, best_acc = 0.0, 0
# 78% -- 96% data 
threshold = args.initial_prob
ending_prob = 0.1
decay_factor = (ending_prob/args.initial_prob)**(1/args.epoch_iid)

for epoch in range(args.num_epoch):
    threshold = threshold*decay_factor
  
    if threshold < ending_prob:
        threshold = 0
    if epoch is not 0 and epoch < 100 and epoch % 30 == 0:
        for param in optimizer.param_groups:
            param['lr'] = param['lr'] / 10
    train(model, model_conf, train_loader, criterion, optimizer,threshold)
    accuracy = test(model, test_loader)
    if accuracy > best_acc:
        patience = 0
        best_acc = accuracy
        best_epoch = epoch
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), 'best_model_cifar10h_MCLS_MCCL.pth.tar')
    print('epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f} threshold: {:.4f} lr: {:.4f}'.format(
            epoch, accuracy, best_epoch, best_acc, threshold,optimizer.param_groups[0]['lr']))


write_csv("CL_MCLS+MC_CL.csv", ["init_prob:" + str(args.initial_prob), "epoch_iid:" + str(args.epoch_iid),
                                 "threshold:{:.4f}".format(threshold),
                                 "best_epoch:"+ str(best_epoch),
                                 "best_acc:" + str(best_acc)
                                 ])

# In[7]:


