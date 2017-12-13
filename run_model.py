import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import argparse
from torch.autograd import Variable

from data import*
from net import*

NUM_SAMPLES=0
parser=argparse.ArgumentParser(description="867 BTC Project")
parser.add_argument('--disable-cuda',action="store_true",help="Disable CUDA")
parser.add_argument("--lr1",type=float,default=0.5)
parser.add_argument("--lr2",type=float,default=0.01)
parser.add_argument("--epochs",type=int, default=10)
parser.add_argument("--num_training_samples",type=int,default=NUM_SAMPLES)
parser.add_argument("--h1",type=int,default=100)
parser.add_argument("--h2",type=int,default=100)
parser.add_argument("--batch_size", type=int,default=2)
parser.add_argument("--train_len",type=int,default=200)
parser.add_argument("--predict_len",type=int,default=10)
parser.add_argument("--cache_size",type=int,default=5365336)

args=parser.parse_args()
args.cuda=not args.disable_cuda and torch.cuda.is_available()


model1=Encoder(args.h1,args.batch_size)
model2=Decoder(args.h1,1,args.predict_len)
#model3=MINI_DECODER(args,h1,2,args.train_len-1)
if args.cuda:
    model1.cuda()
    model2.cuda()
   # model3.cuda()
def train_model(train_data,dev_data,test_data,model1,model2,args):
    garbo1=list(model1.parameters())+list(model2.parameters())
    #garbo1=list(model1.parameters())+list(model3.parameters())
    optimizer=torch.optim.Adam(garbo1,lr=args.lr1)
    vals=[]
    for epoch in range(0,args.epochs):
        print("epoch :"+ str(epoch))
        #Last element is for whether it is training or not
        loss1=run_epoch(train_data,model1,model2,optimizer,args,True)
        loss2=run_epoch(dev_data,model1,model2,optimizer,args,False)
        loss3=run_epoch(test_data,model1,model2,optimizer,args,False)
	vals.append(loss2)
        print("TRAIN LOSS: "+str(loss1))
        print("VAL LOSS "+str(loss2))
        #print("TEST LOSS "+str(loss3))
    print vals
    print "hidden_size "+str(args.h1)
    print "lr "+str(args.lr1)
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
def run_mini_epoch(data,model1,model3,optimizer,args,is_training):
    data_loader=torch.utils.data.DataLoader(data,args.batch_size,shuffle=False)
    RMSE=0
    if is_training:
	model1.train()
	model3.train()
    else:
	model1.eval()
	model3.eval()
    count=0
    for batch in tqdm(data_loader):
	count+=1
	x=autograd.Variable(batch["x"].float()).permute(1,0,2)
	y=x[1:,:,:]
	primero=autograd.Variable(batch["primero"].float())
	if args.cuda:
		x=x.cuda()
		y=y.cuda()
		primero=primero.cuda()
	if is_training:
		optimizer.zero_grad()
	hidden1=model1.initHidden()
	if args.cuda:
		hidden1=(hidden1[0].cuda(),hidden1[1].cuda())
	#hidden2=autograd.Variable(torch.zeros(args.ba
	hidden1=repackage_hidden(hidden1)
	out1=model1(x,hidden1)
	out1=repackage_hidden
def run_epoch(data,model1,model2,optimizer,args,is_training):
    data_loader=torch.utils.data.DataLoader(data,args.batch_size,shuffle=False)
    RMSE=0
    if is_training:
        model1.train()
        model2.train()
    else:
        model1.eval()
        model2.eval()
    count=0
    for batch in tqdm(data_loader):
        count +=1
        x=autograd.Variable(batch["x"].float()).permute(1,0,2)
        y=autograd.Variable(batch["y"].float())
        ultimo=autograd.Variable(batch["ultimo"].float())
        if args.cuda:
            x=x.cuda()
            y=y.cuda()
            ultimo=ultimo.cuda()

        if is_training:
            optimizer.zero_grad()
        hidden1=model1.initHidden()

        if args.cuda:
            hidden1 = (hidden1[0].cuda(), hidden1[1].cuda())
	hidden2=autograd.Variable(torch.zeros(args.batch_size,args.h1).cuda())
	hidden2=repackage_hidden(hidden2)
        hidden1=repackage_hidden(hidden1)
        out1=model1(x,hidden1)
        #out1=repackage_hidden(out1)
	#out1=(repackage_hidden(a),repackage_hidden(b))
        ultimo=repackage_hidden(ultimo)
        out2=model2((out1,hidden2),ultimo)
        loss=F.mse_loss(out2,y)
        RMSE+=loss.data[0]
        if is_training:
            loss.backward()
            optimizer.step()
        if count%200==0:
	    print out2[:3,:]
	    print y[:3,:]
            print ((1.0*RMSE/count)**0.5)
    return (1.0*RMSE/count)**0.5
train = Data("esketit.csv", args.train_len,args.predict_len, args.cache_size)
print train.length
dev = Test("esketit.csv",args.train_len,args.predict_len,args.cache_size)
test= Test("esketit.csv",args.train_len,args.predict_len,args.cache_size,True)
train_model(train,dev,test,model1,model2,args)





