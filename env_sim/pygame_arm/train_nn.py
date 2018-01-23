'''
This code contains a neural network that is regressing inv_kin for the 2 joint arm simulation
'''
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
batch_sz = 1024
epochs = 10000
learning_rate = .0001
input_shape = 2
output_shape = 2
drop_rte = 0.1
hidden_neurons = [50, 40, 20, output_shape]

#load the numpy array
dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))

'''Simple Regression FCN Model'''
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0])
        self.h_1 = nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])
        # self.drop = nn.Dropout(dropout_rte)

    def forward(self, x):
        # x = self.drop(x)

        out_0 = F.tanh(self.h_0(x))
        # out_0 = self.drop(out_0)

        out_1 = F.tanh(self.h_1(out_0))
        # out_1 = self.drop(out_1)

        out_2 = F.tanh(self.h_2(out_1))
        # out_2 = self.drop(out_2)

        out = self.h_3(out_2)
        return out

model = FullyConnectedNetwork(input_shape, hidden_neurons, drop_rte)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_model(model):
    torch.save(model.state_dict(), 'mysavedmodel.pth')

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
        train_step_counter +=1

        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()[0]/train_step_counter))

'''test'''
def test_model(data, label):
    global model
    batch_sz = 64
    model.eval()

    test_loss = 0
    correct = 0
    test_steps = 0
    mseloss = nn.MSELoss()

    for batch_idx in range(int(data.shape[0]/batch_sz)):
        if torch.cuda.is_available():
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda(),
                volatile=True)
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float().cuda())
        else:
            data_batch = Variable(torch.from_numpy(data[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float(),
                volatile=True)
            label_batch = Variable(torch.from_numpy(label[batch_idx*batch_sz:(batch_idx + 1)*batch_sz]).float())

        output = model(data_batch)

        test_loss += mseloss(output, label_batch).data[0] # sum up batch loss

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
train = train.astype('float64')

test = np.load(dir_path + '/data/inv_kin_aprox/test.npy')
test = test.astype('float64')

best_loss = float(sys.maxsize)

'''Train and Test Our Model'''
for epoch in range(epochs):
    #shuffle data
    np.random.shuffle(train)
    data = train[:,0]
    label = train[:,1]
    np.random.shuffle(test)
    test_data = test[:,0]
    test_label = test[:,1]

    train_model(epoch, data, label)
    if epoch % 1 == 0 and epoch != 0:
        current_loss = test_model(test_data, test_label)
        if current_loss < best_loss:
            best_loss = current_loss
            save_model(model)

test_model(test_data, test_label)
print("best loss", best_loss)
