import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd

# Define a custom dataset class for loading and preprocessing log data for inference
class LogInferenceDataset(Dataset):
    def __init__(self, logs, tokenizer, max_length=128, mask_prob=0.15):
        """
        Dataset for masked language modeling (MLM) inference with BERT.
        
        Args:
        logs (list of str): List of log messages.
        tokenizer (BertTokenizer): BERT tokenizer.
        max_length (int): Maximum sequence length.
        mask_prob (float): Probability of masking a token during inference.
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

        # Create labels and mask some tokens
        labels = input_ids.clone()
        mask = torch.rand(input_ids.shape).lt(self.mask_prob) * (input_ids != self.tokenizer.cls_token_id) * \
               (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask] = self.tokenizer.mask_token_id  # Replace tokens with [MASK]

        return input_ids, attention_mask, labels, mask

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_bert_anomaly_detection'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# Load anomalies from the CSV file
anomalies_file = 'loki_logs_last_hour_anomalies.csv'
anomalies_df = pd.read_csv(anomalies_file)
anomalies_logs = anomalies_df['log'].tolist()

# Create a dataset and dataloader for inference
inference_dataset = LogInferenceDataset(anomalies_logs, tokenizer, max_length=128, mask_prob=0.15)
inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)  # Batch size of 1 for individual loss

# Perform inference and calculate loss
# Training loop
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available.")
else:
    print("MPS device is not available. Using CPU.")
    device = torch.device("cpu")

model.to(device)
model.eval()

criterion = torch.nn.CrossEntropyLoss()

print("Calculating loss for each line in anomalies...")

with torch.no_grad():
    for input_ids, attention_mask, labels, mask in inference_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction_scores = outputs.logits

        # Calculate reconstruction error only for masked tokens
        masked_labels = labels[mask]
        masked_predictions = prediction_scores[mask]

        # Calculate loss
        loss = criterion(masked_predictions.view(-1, model.config.vocab_size), masked_labels.view(-1))
        
        print(f"Loss: {loss.item()}") if loss.item() < 2 else None

print("Loss calculation completed.")
