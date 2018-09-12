from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
import glob

'''Prepare Data'''
lst_of_arr = []
filenames = glob.glob('./data/imitation_learning/*.npy')

for element in filenames:
    lst_of_arr.append(np.load(element))

largest_seq = 0
for element in lst_of_arr:
    if element.shape[0] > largest_seq:
        largest_seq = element.shape[0]

arr_lst = []
for element in lst_of_arr:
    difference = largest_seq - element.shape[0]
    # element = np.concatenate((element, np.zeros((difference, 8)))) #zero padding
    for index in range(difference):
        element = np.concatenate((element, np.expand_dims(element[-1],axis=0))) #final element padding
    arr_lst.append(element)

arr = np.asarray(arr_lst)
np.random.shuffle(arr)

length = arr.shape[0]
trn = arr[:int(length * .75)]
tet = arr[int(length * .75):]

train_data = trn[:,:,:4]
train_label = trn[:,:,4:]

test_data = tet[:,:,:4]
test_label = tet[:,:,4:]


'''Set up model'''
class ClassicalLSTM(nn.Module):
    def __init__(self, input_sz, hidden0_sz, hidden1_sz, hidden2_sz, output_sz):
        super(ClassicalLSTM, self).__init__()
        self.lstm_0 = nn.LSTMCell(input_sz, hidden0_sz)
        self.lstm_1 = nn.LSTMCell(hidden0_sz, hidden1_sz)
        self.lstm_2 = nn.LSTMCell(hidden1_sz, hidden2_sz)
        self.fcn1 = nn.Linear(hidden2_sz, output_sz)



    def forward(self, x, states):
        hx_0, cx_0 = self.lstm_0(x, states[0])
        hx_1, cx_1 = self.lstm_1(hx_0, states[1])
        hx_2, cx_2 = self.lstm_2(hx_1, states[2])
        x = self.fcn1(hx_2)
        return x, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]]

input_sz = 4
hidden0_sz = 40
hidden1_sz = 30
hidden2_sz = 15
output_sz = 4
learning_rate = .0001

model = ClassicalLSTM(input_sz, hidden0_sz, hidden1_sz, hidden2_sz, output_sz)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model.cuda()
    print("Using GPU acceleration")
else:
    print("No GPU acceleration")

'''Helper Methods'''
def save_model(model):
    torch.save(model.state_dict(), 'lstmsavedmodel.pth')

def create_lstm_states(n, batch_size):
    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch_size, n)).float().cuda()
        h = Variable(torch.zeros(batch_size, n)).float().cuda()
    else:
        c = Variable(torch.zeros(batch_size, n)).float()
        h = Variable(torch.zeros(batch_size, n)).float()
    return (h, c)

'''Train Model
I'm going to use Batch Gradient Descent'''
def train_model(epoch):
    global model
    global optimizer
    global train_data
    global train_label
    global hidden0_sz
    global hidden1_sz

    model.train()
    train_step_counter = 0
    prev0 = create_lstm_states(hidden0_sz,int(test_data.shape[1]))
    prev1 = create_lstm_states(hidden1_sz, int(test_data.shape[1]))
    prev2 = create_lstm_states(hidden2_sz, int(test_data.shape[1]))
    states = [prev0, prev1, prev2]
    train_loss = 0

    predicted_list = []
    y_list = []

    for step_idx in range(int(test_data.shape[0])):
        if torch.cuda.is_available():
            data = Variable(torch.from_numpy(np.squeeze(test_data[step_idx:(step_idx + 1)])).float().cuda())
            label = Variable(torch.from_numpy(np.squeeze(test_label[step_idx:(step_idx + 1)])).float().cuda())
        else:
            data = Variable(torch.from_numpy(np.squeeze(test_data[step_idx:(step_idx + 1)])).float())
            label = Variable(torch.from_numpy(np.squeeze(test_label[step_idx:(step_idx + 1)])).float())

        output, states = model(data, states)
        predicted_list.append(output)
        y_list.append(label)

    pred = torch.cat(predicted_list)
    y_ = torch.cat(y_list).float()
    loss = F.mse_loss(pred, y_)
    train_step_counter+=1

    loss.backward()
    optimizer.step()
    train_loss+=loss.data

    del(predicted_list[:])
    del(y_list[:])

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()/train_step_counter))


'''Test Model'''
def test_model():
    global model
    global optimizer
    global test_data
    global test_label
    global hidden0_sz
    global hidden1_sz

    model.eval()
    optimizer.zero_grad()
    prev0 = create_lstm_states(hidden0_sz,int(train_data.shape[1] ))
    prev1 = create_lstm_states(hidden1_sz, int(train_data.shape[1]))
    prev2 = create_lstm_states(hidden2_sz, int(train_data.shape[1]))
    states = [prev0, prev1, prev2]
    test_loss = 0
    test_steps = 0

    for step_idx in range(int(train_data.shape[0])):
        if torch.cuda.is_available():
            data = Variable(torch.from_numpy(np.squeeze(train_data[step_idx:(step_idx + 1)])).float().cuda())
            label = Variable(torch.from_numpy(np.squeeze(train_label[step_idx:(step_idx + 1)])).float().cuda())
        else:
            data = Variable(torch.from_numpy(np.squeeze(train_data[step_idx:(step_idx + 1)])).float())
            label = Variable(torch.from_numpy(np.squeeze(train_label[step_idx:(step_idx + 1)])).float())


        output, states = model(data, states)
        test_loss += F.mse_loss(output, label).item() # sum up batch loss

        test_steps+=1

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss/test_steps))
    return (test_loss/test_steps)

'''Main Method'''
def main():
    global train_data
    global train_label
    global test_data
    global test_label
    epochs = 20000
    best_loss = sys.maxsize

    for epoch in range(epochs):
        train_model(epoch)
        if epoch % 1 == 0 and epoch != 0:
            with torch.no_grad():
                current_loss = test_model()
            if current_loss < best_loss:
                best_loss = current_loss
                # save_model(model)

    with torch.no_grad():
        test_model()

    print("best loss", best_loss)
    print("Data")
    print("Train ", train_data.shape, train_label.shape)
    print("Test ", test_data.shape, test_label.shape)

if __name__ == '__main__':
    main()
