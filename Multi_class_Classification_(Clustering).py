# 📥 Importing Dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests

# 🖥️ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📊 Creating Data
x_blob, y_blob = make_blobs(
    n_samples=5000, n_features=2, centers=5, cluster_std=1, random_state=42
)

# 📌 Visualizing the Dataset
plt.scatter(x=x_blob[:, 0], y=x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.title("Dataset Visualization")
plt.show()

# 🧮 Converting Data to Tensors
x = torch.from_numpy(x_blob).type(torch.float).to(device)
y = torch.from_numpy(y_blob).type(torch.LongTensor).to(device)

# ✂️ Splitting Data into Train and Test Sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 🧠 Defining the Model
class ClusteringModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.linear1(x))
        out2 = self.relu(self.linear2(out1))
        return self.linear3(out2)

# 🔧 Initializing Model, Loss, and Optimizer
torch.manual_seed(42)
torch.cuda.manual_seed(43)

model = ClusteringModel(in_features=2, out_features=5, hidden_units=16).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 📈 Accuracy Function
def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(labels) * 100

# 🔮 Initial Predictions (Before Training)
model.eval()
with torch.inference_mode():
    train_logits = model(x_train)
    test_logits = model(x_test)

train_pred = torch.argmax(torch.softmax(train_logits, dim=1), dim=1)
test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)

print(f"Predicted Training Labels (Before Training): {train_pred[:10]}\n")
print(f"Actual Training Labels: {y_train[:10]}\n")
print(f"Predicted Test Labels (Before Training): {test_pred[:10]}\n")
print(f"Actual Test Labels: {y_test[:10]}")

# 📦 Downloading Helper Functions for Visualization
if not Path("helper_functions.py").is_file():
    print("Downloading helper_functions.py...")
    response = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(response.content)

from helper_functions import plot_predictions, plot_decision_boundary

# 🎨 Visualizing Decision Boundaries (Before Training)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Data (Before Training)")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test Data (Before Training)")
plot_decision_boundary(model, x_test, y_test)
plt.show()

# 🔄 Training Loop
nepochs = 100
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(nepochs):
    # Training Phase
    model.train()
    train_logits = model(x_train)
    train_loss = loss_function(train_logits, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluation Phase
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test)
        test_loss = loss_function(test_logits, y_test)

    # Compute Accuracies
    train_acc = accuracy(train_logits, y_train)
    test_acc = accuracy(test_logits, y_test)

    # Store Metrics
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Logging Progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{nepochs}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

# 📉 Plotting Loss and Accuracy Over Epochs
epochs = range(1, nepochs + 1)
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color='blue', marker='o')
plt.plot(epochs, test_losses, label="Validation Loss", color='red', linestyle='--', marker='x')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green', marker='o')
plt.plot(epochs, test_accuracies, label="Validation Accuracy", color='orange', linestyle='--', marker='x')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 🔮 Final Predictions (After Training)
model.eval()
with torch.inference_mode():
    train_logits = model(x_train)
    test_logits = model(x_test)

train_pred = torch.argmax(torch.softmax(train_logits, dim=1), dim=1)
test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)

print(f"Predicted Training Labels (After Training): {train_pred[:10]}")
print(f"Actual Training Labels: {y_train[:10]}")
print(f"Predicted Test Labels (After Training): {test_pred[:10]}")
print(f"Actual Test Labels: {y_test[:10]}")

# 💾 Saving and Reloading Model
torch.save(model.state_dict(), "MulticlassModel.pth")
print("Model Saved!")

saved_model = ClusteringModel(in_features=2, out_features=5, hidden_units=16)
saved_model.load_state_dict(torch.load("MulticlassModel.pth"))
print("Model Loaded Successfully!")
