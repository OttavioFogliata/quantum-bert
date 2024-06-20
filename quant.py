import pennylane as qml
from pennylane import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
# from dummy_dataset import data, labels
from torch.optim import lr_scheduler
import copy
import time
from datasets import load_dataset


dataset_id='stanfordnlp/imdb'
dataset = load_dataset(dataset_id, split="train") # load_dataset is a function of datasets library of HuggingFace
dataset = dataset.shuffle(seed=42).select(range(250))  # Shuffle and select a subset of the dataset
# print(list(dataset))
# {'text': 'WORK, XPO, PYX and AMKR among after hour movers', 'label': 2},
data = [item['text'] for item in list(dataset)]
print(data[:2])
labels = [item['label'] for item in list(dataset)]
# label_names = ['Negative', 'Positive'] #, 'Positive']
# label_texts = [label_names[label] for label in labels]
# print(label_texts[:2])



# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(set(labels))

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Split the data
# train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_inputs['input_ids'],
#                                                                         encoded_labels,
#                                                                         test_size=0.25,  # ex 0.2
#                                                                         random_state=42)


encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    encoded_inputs['input_ids'], encoded_labels, test_size=0.2, random_state=42)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.25, random_state=42)

batch_size = 16  # ex 16

train_dataset = TensorDataset(train_inputs, torch.tensor(train_labels))
test_dataset = TensorDataset(test_inputs, torch.tensor(test_labels))
val_dataset = TensorDataset(val_inputs, torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataset_sizes = {
  'train': len(train_dataset),
  'validation': len(val_dataset),
  'test': len(test_dataset)
  }

dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'validation': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}


# Define the number of quantum layers and qubits
n_layers = 2
n_qubits = 8

# Define a quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum node with correct parameter shapes
@qml.qnode(dev, interface='torch')
def quantum_net(inputs, q_weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(q_weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# Initialize the quantum weights with explicit shapes
weight_shapes = {"q_weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_net, weight_shapes)

first_features_to_use = num_classes # 3
class QuantumBertClassifier(nn.Module):
    def __init__(self):
        super(QuantumBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.quantum_layer = quantum_layer
        self.classifier = nn.Linear(n_qubits, num_classes)  # N qubits output to 3 classes

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        q_input = cls_output[:, :first_features_to_use]  # Use first 4 features for quantum layer
        q_output = self.quantum_layer(q_input)
        logits = self.classifier(q_output)
        return logits

model = QuantumBertClassifier()


# --- Training the model ---
# Set up the loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

gamma_lr_scheduler = 0.05    # Learning rate reduction applied every 10 epochs.

optimizer_hybrid = optim.Adam(model.parameters(), lr=1e-3)
# We schedule to reduce the learning rate by a factor of gamma_lr_scheduler every 10 epochs.
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_hybrid, step_size=5, gamma=gamma_lr_scheduler
)

# Training loop
epochs = 20

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    print("Training started:")

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = dataset_sizes[phase] // batch_size
            it = 0
            for inputs, labels in dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()

                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                print(
                    "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                        phase,
                        epoch + 1,
                        num_epochs,
                        it + 1,
                        n_batches + 1,
                        time.time() - since_batch,
                    ),
                    end="\r",
                    flush=True,
                )
                it += 1

            # Print epoch results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(
                "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                    "train" if phase == "train" else "validation  ",
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc,
                )
            )

            # Check if this is the best model wrt previous epochs
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

            # Update learning rate
            if phase == "train":
                scheduler.step()
    # Print final results
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )
    print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
    return model

model_hybrid = train_model(
    model, criterion, optimizer_hybrid, exp_lr_scheduler, num_epochs=epochs
)

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         input_ids, labels = batch
#         optimizer_hybrid.zero_grad()
#         outputs = model(input_ids)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer_hybrid.step()
#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

# Save the model
torch.save(model_hybrid.state_dict(), "quantum_bert_classifier.pth")

# --- Testing the model ---

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")