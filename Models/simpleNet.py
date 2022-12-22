import torch
import torch.nn as nn


class simpleNet(nn.Module):
    ##nn.ReLU()
    def __init__(self, input_shape, num_classes, pretrained=False, hidden_size=100, nonlinearity=nn.GELU()):
        super().__init__()

        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.embedding = nn.Embedding(input_dim, hidden_size)
        #self.layer_norm = nn.LayerNorm(hidden_size)
        #self.fc0 = nn.Linear(input_dim, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act1 = nonlinearity
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nonlinearity
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.flatten = nn.Flatten()
        self.hid_dim = hidden_size

    def forward(self, x):
        x = self.flatten(x).long()
        #x = self.fc0(x)
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        #x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.classifier(x)
        #x = self.softmax(x)
        return x

'''
## Test ##
model = simpleNet(10, 10, 4)
batch = torch.LongTensor([[1,1,1,1],[1,1,1,1]])
output = model(batch)
print(output)
'''