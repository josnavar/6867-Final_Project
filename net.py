import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self,h1,batch):
        super(Encoder, self).__init__()
        #Nets will write hidden layers to fields
        self.h1=h1
        self.i1=2
        self.n1=2
        self.batch=batch
        #self.encoder=torch.nn.LSTMCell(input_size=self.i1,hidden_size=self.h1)
	self.encoder=torch.nn.LSTM(input_size=self.i1,hidden_size=self.h1,num_layers=self.n1,dropout=0.5)
    def forward(self,input,hidden):
        #Encoder:
        out1,manuelito=self.encoder(input,hidden)
        return out1[-1,:,:]
    def initHidden(self):
        h0=autograd.Variable(torch.zeros(self.n1,self.batch,self.h1))
        c0=autograd.Variable(torch.zeros(self.n1,self.batch,self.h1))
        return (h0,c0)
class MINI_DECODER(nn.Module):
    def __init__(self,h1,i1,target_size):
	self.h1=h1
	self.i1=i1
	#^^
	#Inputs are prices,volumes themselves
	self.target_size=target_size
	self.fc=torch.nn.Linear(h1,2).cuda()
	self.mini_decoder=torch.nn.LSTMCell(input_size=i1,hidden_size=self.h1)
    def forward(self,encoding_output,last_output):
	hidden=encoding_output
	predictions=None
	pv=last_output
	for elt in range(self.target_size):
		hidden=self.mini_decoder(pv,hidden)
		pv=self.fc(hidden[0])
		if predictions is None:
			predictions=pv
		else:
			predictions=torch.cat((predictions,pv),1)
	return predictions
class Decoder(nn.Module):
    def __init__(self,h2,i2,target_size):
        super(Decoder, self).__init__()
        #I2 is the same size as the hidden dimensionality of encoder
        self.h2,self.i2,self.n2=(h2,i2,1)
        #self.decoder=torch.nn.LSTM(input_size=self.i2,hidden_size=self.h2,num_layers=self.n2)
        self.decoder=torch.nn.LSTMCell(input_size=self.i2,hidden_size=self.h2)
        self.fc1=torch.nn.Linear(h2,1).cuda()
        #self.fc2=torch.nn.Linear(h2/2,1).cuda()
        self.target_size=target_size
    #Input from encoding
    #last_output should be (1,batch,2) -> price and volume vector
    def forward(self,encoding_output,last_output):
	#last_output is a 200x1
        hidden=encoding_output
        predictions=None
        pv=last_output
        for elt in range(self.target_size):
            hidden= self.decoder(pv, hidden)
           # pv=self.fc1(hidden[0])
	    pv=self.fc1(hidden[0])
            if predictions is None:
                predictions=pv
            else:
#		print predictions                
		predictions=torch.cat((predictions,pv),1)
        return predictions



