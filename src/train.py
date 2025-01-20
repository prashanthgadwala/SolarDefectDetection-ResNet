import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/data.csv', delimiter=';', header=None)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

resnet_model = model.ResNet()

criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.001)

trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer, train_dl=train_loader, val_test_dl=val_loader, cuda=t.cuda.is_available(), early_stopping_patience=5)

res = trainer.fit(epochs=50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')