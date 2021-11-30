# Imports
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pre-trained model

# Simple Identity class that let's input pass without changes
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# # Load pretrain model & modify it
# model = torchvision.models.resnet18(pretrained=True)

# # If you want to do finetuning then set requires_grad = False
# # Remove these two lines if you want to train entire model,
# # and only want to load the pretrain weights.
# for param in model.parameters():
#     param.requires_grad = False


# model.fc = nn.Sequential(
#     nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 2)
# )

model = torch.load('/content/drive/MyDrive/cs230_project_all/cs230_project/D_emotion.pt')
model.eval()
model.to(device)


# dataset class
class OASIS_dataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        # y: valence and arousal ratings
        xy = pd.read_csv('/content/drive/MyDrive/cs230_project_all/unorganized_data/OASIS/OASIS_for_modeling.csv')
        # print(xy.head())
        self.n_samples = xy.shape[0]
        self.y_data = torch.tensor(xy[['Valence_mean', 'Arousal_mean']].values, dtype=torch.float32)
        # x: the corresponding picture
        self.x_name = xy['Theme']

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):        
        # print(self.x_name[index])
        self.x_path = '/content/drive/MyDrive/cs230_project_all/unorganized_data/OASIS/images/' + self.x_name[index] + '.jpg'
        self.x_img = Image.open(self.x_path).convert('RGB')
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        return transformations(self.x_img), self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# Hyperparameters
learning_rate = 1e-2
batch_size = 256
num_epochs = 200

# Load Data
train_dataset = OASIS_dataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 170], gamma=0.5)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # print("epoch:", epoch, "; batch", batch_idx)
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    
    scheduler.step()
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

# save the trained model
torch.save(model, '/content/drive/MyDrive/cs230_project_all/cs230_project/D_emotion.pt')