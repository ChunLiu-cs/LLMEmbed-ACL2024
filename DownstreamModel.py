import torch
import torch.nn as nn
import torch.nn.functional as F

class DownstreamModel(nn.Module):
    def __init__(self, class_num, SIGMA):
        super(DownstreamModel, self).__init__()
        self.SIGMA = SIGMA
        self.compress_layers = nn.ModuleList()
        for _ in range(5):
            layers = []
            layers.append(nn.Linear(4096, 1024))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            self.compress_layers.append(nn.Sequential(*layers))
        
        self.fc1 = nn.Linear(4145, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, class_num)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_l, input_b, input_r):
        batch_size = input_l.shape[0]
        split_tensors = torch.split(input_l, 1, dim=1)
        input = []
        for i, split_tensor in enumerate(split_tensors):
            split_tensor = split_tensor.reshape(batch_size,-1)
            input.append(self.compress_layers[i](split_tensor))
        input.append(input_b)
        input.append(input_r)
        input = torch.stack(input, dim=1)
        # X * X^T
        input_T = input.transpose(1, 2)
        input_P = torch.matmul(input, input_T)
        input_P = input_P.reshape(batch_size, -1)
        # PN func
        input_P = 2*F.sigmoid(self.SIGMA * input_P) - 1
        
        a = torch.mean(input_l, dim=1)
        input = torch.cat([input_P, a], dim=1)
        # print(input.shape)

        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.softmax(output)

        return output

if __name__ == '__main__':
    model = DownstreamModel(2)
    print(model)
