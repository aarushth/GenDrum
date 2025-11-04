import torch
import torch.nn as nn   
from dataloader import AudioDataset
from model import AudioClassifier
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"gpu available: {torch.cuda.is_available()}")


# dirs = ["nonaugmentdata/claps", "nonaugmentdata/snaps", "nonaugmentdata/hits"]    
dirs = ["newData/claps", "newData/snaps", "newData/hits"]
classes = {0 : "Clap", 1: "Snap", 2:"Hit"}

dataset = AudioDataset(dirs)

train_size = int(0.7 * len(dataset)) # 70 train, 15 val, 15 test
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - (train_size + val_size)


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#lower batch size
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

version = 17
num_epochs = 20 # increase

model = AudioClassifier()

model.to(device)

criterion = nn.CrossEntropyLoss()

#up learning rate if loss decrease is slow
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01) # try a couple optimizers

final_train_losses = []
final_val_losses = []

for epoch in range(num_epochs):
    model.train()

    train_losses = []
    for i, (inputs, labels) in enumerate(train_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = sum(train_losses) / len(train_losses)
    final_train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_losses = []
        correct_predictions = 0
        total_predictions = 0
        for i, (inputs, labels) in enumerate(val_dataloader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = correct_predictions / total_predictions
        final_val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

#test phase
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    test_losses = []
    correct_predictions = 0
    total_predictions = 0
    for i, (inputs, labels) in enumerate(test_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        test_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        all_preds.extend([p.item() for p in predicted])
        all_labels.extend([l.item() for l in labels])

    test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = correct_predictions / total_predictions

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")




epochs = range(1, len(final_train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, final_train_losses, label="Train Loss", color="blue", marker="o")
plt.plot(epochs, final_val_losses, label="Validation Loss", color="orange", marker="s")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

labels = sorted(classes.keys())
label_names = [classes[l] for l in labels]

cm = confusion_matrix(all_labels, all_preds, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

torch.save(model.state_dict(), f"models/audio_classifier_v{version}.pth")