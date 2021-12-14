from enum import auto
from numpy.core.numeric import indices
import torch
from torch import nn
from torch.nn.modules.flatten import Unflatten
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolutions
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 1, 3)
        self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)

        #Flatten
        self.flatten = nn.Flatten(start_dim=1)
        
        #Fully Connected
        self.fc1 = nn.Linear(418, 256)
        self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, indices1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x, indices2 = self.maxpool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return x, indices1, indices2


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #Fully connected
        #self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 418)

        #Unflatten
        self.unflatten = nn.Unflatten(1, torch.Size([1, 22, 19]))

        #Convolutions
        self.maxunpool1 = nn.MaxUnpool2d(2, 2)
        self.convtrans1 = nn.ConvTranspose2d(1, 32, 3)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.convtrans2 = nn.ConvTranspose2d(32, 3, 5)

    def forward(self, x, indices1, indices2):
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.unflatten(x)
        x = self.maxunpool1(x, indices2)
        x = self.convtrans1(x)
        x = self.maxunpool2(x, indices1)
        x = self.convtrans2(x)
        x = torch.tanh(x)
        return x

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 0.001
        #torch.manual_seed(0)
        self.params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
        self.optim = torch.optim.Adam(self.params_to_optimize, lr=self.lr, weight_decay=1e-05)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Current device: {self.device}')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def training(self, data_loader):
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        for image_batch in data_loader:
            image_batch = image_batch.to(self.device)

            encoded, indices1, indices2 = self.encoder(image_batch)

            decoded = self.decoder(encoded, indices1, indices2)

            loss = self.loss_fn(decoded, image_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            #print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)
    
    def test_epoch(self, validation_loader):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            conc_out = []
            conc_label = []
            for batch in validation_loader:
                batch = batch.to(self.device)
                encoded, indices1, indices2 = self.encoder(batch)
                decoded = self.decoder(encoded, indices1, indices2)
                conc_out.append(decoded.cpu())
                conc_label.append(batch.cpu())
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data

    def forward(self, x):
        self.encoder.eval()
        self.decoder.eval()
        x = x.to(self.device)
        x, indices1, indices2 = self.encoder(x)
        x = self.decoder(x, indices1, indices2)
        return x

            



