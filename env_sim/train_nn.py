'''
This code contains a neural network that is regressing inv_kin for the 2 joint arm simulation
'''
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sys
import os

#network model
'''global variables'''
batch_sz = 2056
epochs = 10000
learning_rate = .0001
input_shape = 2
output_shape = 4
drop_rte = 0.1
hidden_neurons = [40, 40, 40, 40,output_shape]

#load the numpy array
dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))

normalize_data_bool = True

'''Simple Regression FCN Model'''
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0])
        self.h_1 = nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])
        self.h_4 = nn.Linear(num_hidden_neurons[3], num_hidden_neurons[4])

        self.drop = nn.Dropout(dropout_rte)

    def forward(self, x):

        out_0 = F.tanh(self.h_0(x))
        out_0 = self.drop(out_0)

        out_1 = F.tanh(self.h_1(out_0))
        out_1 = self.drop(out_1)

        out_2 = F.tanh(self.h_2(out_1))
        out_2 = self.drop(out_2)

        out_3 = F.tanh(self.h_3(out_2))

        out = self.h_4(out_3)
        return out

model = FullyConnectedNetwork(input_shape, hidden_neurons, drop_rte)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_model(model):
    torch.save(model.state_dict(), 'mysavedmodel.pth')

def normalize(arr, normalize_type):
    #type 0 is between 0 and pi
    if normalize_type == 0:
        for i in range(int(arr.size)):
            arr[i] = (arr[i]) / (np.pi)
        return arr
    elif normalize_type == 1: #type 1 is between 0 and 2 pi
        for i in range(int(arr.size)):
            arr[i] = (arr[i]) / (2 * np.pi)
        return arr
    else:     #type 2 is between -420 and 420
        for i in range(int(arr.size)):
            arr[i] = (arr[i]- (-420)) / (840)
        return arr

'''train'''
def train_model(epoch, data, label):
    global model
    global optimizer
    global batch_sz

    model.train()
    optimizer.zero_grad()
    train_loss = 0
    train_step_counter = 0
    mseloss = nn.MSELoss()

    for batch_idx in range(int(data.shape[0]/batch_sz)):
        #Make tensor cuda tensor if cuda is available
        if torch.cuda.is_available():
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda())
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda())
        else:
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float())
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float())

        output = model(data_batch)
        loss = mseloss(output, label_batch)
        train_loss+=loss.data
        # print(train_loss, "dtaasdfa")
        train_step_counter +=1

        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()/train_step_counter))

'''test'''
def test_model(data, label):
    global model
    batch_sz = 2056
    model.eval()

    test_loss = 0
    correct = 0
    test_steps = 0
    mseloss = nn.MSELoss()

    for batch_idx in range(int(data.shape[0]/batch_sz)):
        if torch.cuda.is_available():
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda())
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda())
        else:
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float())
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float())

        output = model(data_batch)

        test_loss += mseloss(output, label_batch).item() # sum up batch loss

        test_steps+=1

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss/test_steps))
    return (test_loss/test_steps)

'''Create model'''
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU Acceleration")
else:
    print("not using gpu acceleration")

'''Load data'''
train = np.load(dir_path + '/data/inv_kin_aprox/train.npy')

test = np.load(dir_path + '/data/inv_kin_aprox/test.npy')

best_loss = float(sys.maxsize)

if normalize_data_bool:
    print("Normalizing Input and Output")

    train[:,0] = normalize(train[:,0], 2)
    train[:,1] = normalize(train[:,1], 2)

    test[:,0] = normalize(test[:,0], 2)
    test[:,1] = normalize(test[:,1], 2)

'''Train and Test Our Model'''
print("Training")
for epoch in range(epochs):
    # shuffle data
    np.random.shuffle(train)
    data = train[:,:2]
    label = train[:,2:]
    np.random.shuffle(test)
    test_data = test[:,:2]
    test_label = test[:,2:]

    train_model(epoch, data, label)
    if epoch % 1 == 0 and epoch != 0:
        with torch.no_grad():
            current_loss = test_model(test_data, test_label)
        if current_loss < best_loss:
            best_loss = current_loss
            save_model(model)

with torch.no_grad():
    test_model(test_data, test_label)

print("best loss", best_loss)
