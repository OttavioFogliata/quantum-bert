import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Example dataset
data = [
    "This movie is excellent and inspiring.",
    "The weather today is quite pleasant.",
    "I didn't enjoy the concert last night.",
    "The new restaurant in town serves delicious food.",
    "The company's financial performance is stable.",
    "I'm feeling tired and exhausted.",
    "The book I read was boring and uninteresting.",
    "Today's meeting was productive and efficient.",
    "I'm indifferent about the upcoming event.",
    "The customer service was disappointing.",
    "I love the new features in the latest update.",
    "The traffic this morning was terrible.",
    "The news about the project is concerning.",
    "The vacation was relaxing and enjoyable.",
    "The movie plot was confusing and hard to follow.",
    "The product launch was successful.",
    "The service at the restaurant was average.",
    "The new album has received mixed reviews.",
    "The performance of the team was outstanding.",
    "I have no strong feelings about this topic.",
    "The new software update is impressive and user-friendly.",
    "The political situation is complex and challenging.",
    "I feel optimistic about the future.",
    "The educational system needs improvement.",
    "The game was exciting and entertaining.",
    "The traffic light system needs reevaluation.",
    "I am satisfied with my current job.",
    "The economic forecast looks promising.",
    "The customer feedback has been overwhelmingly positive.",
    "The novel left a lasting impression on me.",
    "I'm unsure about the decision.",
    "The outcome was disappointing.",
    "The speech was informative and enlightening.",
    "The atmosphere at the event was lively.",
    "The results exceeded expectations.",
    "I'm concerned about the environment.",
    "The artwork was thought-provoking.",
    "The service quality has declined recently.",
    "The film received critical acclaim.",
    "The issue is controversial and divisive.",
    "The experience was memorable and enjoyable.",
    "The report highlighted key challenges.",
    "I'm pleased with the progress so far.",
    "The initiative is commendable.",
    "The product design is innovative.",
    "The situation is concerning.",
    "The presentation was lackluster.",
    "The solution proposed is effective.",
    "The feedback was constructive.",
    "The game has potential for improvement."
]

# Corresponding labels
labels = [
    "Positive", "Neutral", "Negative", "Positive", "Neutral",
    "Negative", "Negative", "Positive", "Neutral", "Negative",
    "Positive", "Negative", "Negative", "Positive", "Negative",
    "Positive", "Neutral", "Positive", "Positive", "Neutral",
    "Neutral", "Positive", "Negative", "Positive", "Neutral",
    "Positive", "Positive", "Positive", "Positive", "Negative",
    "Negative", "Positive", "Positive", "Neutral", "Negative",
    "Positive", "Neutral", "Positive", "Negative", "Positive",
    "Positive", "Negative", "Positive", "Negative", "Positive",
    "Negative", "Positive", "Neutral", "Negative", "Positive"
]

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(set(labels))

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

# Split the data
train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_inputs['input_ids'],
                                                                        encoded_labels,
                                                                        test_size=0.25,
                                                                        random_state=42)

train_dataset = TensorDataset(train_inputs, torch.tensor(train_labels))
test_dataset = TensorDataset(test_inputs, torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model (BERT for sequence classification)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Set up the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 20  # Adjust as needed

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")