import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")

# find how much natural disasters happens at a place (long, lat)
places = []
disasters = {}

X = np.array(data.loc[:, ["latitude", "longitude"]])
y = []

for lat, long in X:
    if (round(lat), round(long)) not in disasters:
        disasters[(round(lat), round(long))] = 1
    else:
        disasters[(round(lat), round(long))] += 1

for key, items in disasters.items():
    places.append(np.array(key, np.float32))
    y.append(disasters[(round(lat), round(long))])

X = torch.from_numpy(np.array(places, np.float32))
y = torch.from_numpy(np.array(y, np.float32))
    
# create a neural network
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, out_features)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
# train the model
model = Model(2, 1)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

if __name__ == "__main__":
    model.train()

    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

    torch.save(model, "model.pt")