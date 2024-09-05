import random
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification #, AdamW
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
csv_file = "HDFS_2k.log_structured.csv"  # Replace with the path to your CSV file
log_data = pd.read_csv(csv_file)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  #Change as you wish your desired maximum sequence length

# Tokenize and prepare data
tokenized_texts = []
for index, row in log_data.iterrows():
    text = str(row['EventTemplate'])
    label = 0 if row['Level'] == 'INFO' else 1
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    tokenized_texts.append({
        'input_ids': inputs['input_ids'].flatten(),
        'attention_mask': inputs['attention_mask'].flatten(),
        'label': torch.tensor(label, dtype=torch.long)
    })

# Split into training and test sets
train_data, test_data = train_test_split(tokenized_texts, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and criterion
optimizer = AdamW(model.parameters(), lr=2e-5)
# Define class weights
# Assuming your dataset is imbalanced (Class 0: normal logs, Class 1: anomalies)
class_counts = [len(log_data[log_data['Level']=='INFO']), len(log_data[log_data['Level']!='INFO'])]  
total_samples = sum(class_counts)
class_weights = [total_samples / (2.0 * count) for count in class_counts]  # Calculate class weights

# Convert class weights to a PyTorch tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, optimizer, train_loader, criterion, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}')

# Train the model
train(model, optimizer, train_loader, criterion)

# Save the fine-tuned model
model_save_path = 'fine_tuned_bert_model.pth'  # Define the path where you want to save the model
torch.save(model.state_dict(), model_save_path)
print(f"Fine-tuned BERT model saved to '{model_save_path}'")