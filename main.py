import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split

import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

# Logging on wandb
parser.add_argument('--name', type=str, default=None,
        help="Run name")
parser.add_argument('--wandb', type=str, default='trigger-bilevel', 
        help="wandb project name")
parser.add_argument('--use_wandb', action='store_true',
                    help="use wandb project name")
parser.add_argument('--log_interval', type=int, default=25,
        help="Number of steps between logging key stats")
parser.add_argument('--print_interval', type=int, default=250,
        help="Number of steps between printing key stats")

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

if not parser.name:
    parser.name = f"SAGPool"
if parser.use_wandb:
    wandb.init(
        project=f'{parser.wandb}', 
        name=f'{parser.name}',
        # tags=["Sunrise", "Physics Trigger"],
        tags=["Sunrise", "SAGPool"]
        # config=config,
    )

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(model,loader):
    model.train()
    correct = 0.
    loss = 0.
    for i, data in enumerate(loader):
        data = data.to(args.device)
        out = model(data)
        loss_iter = F.nll_loss(out, data.y)

        loss_iter.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += loss_iter
        
    return correct / len(loader.dataset), loss / len(loader.dataset)

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    train_acc,train_loss = test(model,train_loader)
    val_acc,val_loss = test(model,val_loader)
    
    wandb.log({"Train Loss": train_loss})
    wandb.log({"Train Acc": train_acc})
    wandb.log({"Valid Loss": val_loss})
    wandb.log({"Valid Acc": val_acc})
    print(f"Trsin loss:{val_loss}\taccuracy:{val_acc}")
    print(f"Validation loss:{val_loss}\taccuracy:{val_acc}")

    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print(f"Model saved at epoch{epoch}")
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))

    
test_acc,test_loss = test(model,test_loader)

wandb.log({"Test Loss": test_loss})
wandb.log({"Test Acc": test_acc})
print(f"Test accuarcy:{test_acc}")
