import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import pandas as pd
import numpy as np

# Define a custom dataset class for loading and preprocessing log data
class LogDataset(Dataset):
    def __init__(self, logs, tokenizer, max_length=128, mask_prob=0.15):
        """
        Dataset for masked language modeling (MLM) with BERT.
        
        Args:
        logs (list of str): List of log messages.
        tokenizer (BertTokenizer): BERT tokenizer.
        max_length (int): Maximum sequence length.
        mask_prob (float): Probability of masking a token.
        """
        self.logs = logs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log = self.logs[idx]
        inputs = self.tokenizer.encode_plus(
            log,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Create labels for MLM
        labels = input_ids.clone()
        
        # Create a mask array with the same size as input_ids
        mask = torch.rand(input_ids.shape).lt(self.mask_prob) * (input_ids != self.tokenizer.cls_token_id) * \
               (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask] = self.tokenizer.mask_token_id  # Replace tokens with [MASK]

        return input_ids, attention_mask, labels

# Load normal log data
logs_df = pd.read_csv('loki_logs_last_hour-small.csv')  # Assuming logs are saved in this CSV file
normal_logs = logs_df['log'].tolist()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Create a dataset and dataloader
dataset = LogDataset(normal_logs, tokenizer, max_length=128, mask_prob=0.15)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    print("MPS device is not available. Using CPU.")
    device = torch.device("cpu")

model.to(device)
epochs = 1
print(f"{len(dataloader)} batches, {epochs} epochs")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_anomaly_detection')
tokenizer.save_pretrained('fine_tuned_bert_anomaly_detection')

print("Training completed and model saved.")

