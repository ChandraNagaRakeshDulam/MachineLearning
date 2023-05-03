import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as Function
import torch.optim as optim
import torchvision.transforms as transforms
import joblib
import random
import albumentations
import matplotlib.pyplot as plt
import argparse
import time
import cv2
import cnn_models

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-e', '--epochs', default=10, type=int, help='num of epochs for training')
args = vars(arg_parser.parse_args())


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Computation device: {device}")

dataframe = pd.read_csv('./image_data/csv_data.csv')
X = dataframe.image_path.values
y = dataframe.target.values

(X_train, x_test, y_train, y_test) = (train_test_split(X, y, test_size=0.15, random_state=42))

print(f"Training on {len(X_train)} images")
print(f"Validationg on {len(x_test)} images")


class signTranslator(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels

        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
        ])

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.X[i])
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

train_data = signTranslator(X_train, y_train)
test_data = signTranslator(x_test, y_test)
 

training_loader = DataLoader(train_data, batch_size=32, shuffle=True)
testing_loader = DataLoader(test_data, batch_size=32, shuffle=False)


translator_model = cnn_models.TranslatorCNN().to(device)
print(translator_model)

total_params = sum(p.numel() for p in translator_model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in translator_model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

optimizer = optim.Adam(translator_model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

def validate(model, dataloader):
    print('Validating')
    model.eval()
    run_loss = 0.0
    run_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data)/dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            run_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            run_correct += (preds == target).sum().item()
        
        validation_loss = run_loss/len(dataloader.dataset)
        validation_accuracy = 100. * run_correct/len(dataloader.dataset)
        print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}')
        
        return validation_loss, validation_accuracy

def fit(model, dataloader):
    print('Training')
    model.train()
    run_loss = 0.0
    run_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        run_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        run_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = run_loss/len(dataloader.dataset)
    train_accuracy = 100. * run_correct/len(dataloader.dataset)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
    
    return train_loss, train_accuracy

train_loss , train_accuracy = [], []
validation_loss , validation_accuracy = [], []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    training_epoch_loss, training_epoch_accuracy = fit(translator_model, training_loader)
    validation_epoch_loss, val_epoch_accuracy = validate(translator_model, testing_loader)
    train_loss.append(training_epoch_loss)
    train_accuracy.append(training_epoch_accuracy)
    validation_loss.append(validation_epoch_loss)
    validation_accuracy.append(val_epoch_accuracy)
end = time.time()

print(f"{(end-start)/60:.3f} minutes")

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='yellow', label='train accuracy')
plt.plot(validation_accuracy, color='green', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./predictions/accuracy.png')
plt.show()
 
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='yellow', label='train loss')
plt.plot(validation_loss, color='green', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./predictions/loss.png')
plt.show()

torch.save(translator_model.state_dict(), './predictions/model.pth')