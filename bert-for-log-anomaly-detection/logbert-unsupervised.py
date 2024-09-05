import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

csv_file = "HDFS_2k.log_structured.csv"  # Replace with the path to your CSV file
log_data = pd.read_csv(csv_file)

# Initialize the BERT tokenizer and model
model_name = "bert-base-uncased"  # Choose the BERT model you prefer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Step 1: Tokenize and preprocess the log data
tokenized_logs = []
attention_masks = []
lng=[]
for log_text in log_data["EventTemplate"]:

    # Tokenize the log message
    tokens = tokenizer.tokenize(log_text)
    print(tokens)
    lng.append(tokens)
    # Add special tokens and apply padding
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    # Optionally, truncate or pad to a fixed length
    max_length = 100  # Set your desired maximum length
    input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_ids))

    tokenized_logs.append(input_ids)

    # Create attention mask
    attn_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
    attention_masks.append(attn_mask)

# Convert tokenized logs to PyTorch tensors
log_tensors = torch.tensor(tokenized_logs)
attention_masks = torch.tensor(attention_masks)

log_embeddings = []
count=0

for log_tensor, attn_mask in zip(log_tensors, attention_masks):
    # Convert log tensor and attention mask to PyTorch
    count+=1
    log_tensor = log_tensor.unsqueeze(0)
    attn_mask = attn_mask.unsqueeze(0)

    # Pass the log tensor and attention mask through the BERT model to obtain embeddings
    with torch.no_grad():
        outputs = model(log_tensor, attention_mask=attn_mask)

    # Extract the embedding for [CLS] token (outputs[0][:, 0, :])
    log_embedding = torch.mean(outputs[0][:, 0, :], dim=0).numpy()
    log_embeddings.append(log_embedding)
    print(count)

# Step 3: Anomaly Detection
# Calculate the mean (centroid) of all log embeddings
if len(log_embeddings) > 0:
    all_logs_centroid = np.mean(log_embeddings, axis=0)
else:
    # Handle the case where there are no logs
    all_logs_centroid = None

threshold = 0.977 # Adjust the threshold as needed
logs_anomal=[]
# Compare each log with the mean log centroid using cosine similarity
for i, log_embedding in enumerate(log_embeddings):
    if all_logs_centroid is not None:
        similarity_score = cosine_similarity([log_embedding], [all_logs_centroid])[0][0]
        # Compare similarity score with the threshold
        if similarity_score < threshold:
            print(f"Anomaly detected: Log {i}")
            logs_anomal.append(i)