import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

# Define the path to the saved model
model_save_path = "fine_tuned_bert_model.pth"

# Initialize the model (use the same configuration as the original model)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load the state dictionary into the model
model.load_state_dict(torch.load(model_save_path))

# Set the model to evaluation mode (important for inference)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Interactive loop for inference
print("Enter 'quit' to exit the loop.")
while True:
    # Ask for user input
    user_input = input("Please enter a sentence for classification: ")
    
    # Exit condition
    if user_input.lower() == 'quit':
        break

    # Tokenize the input text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits (output before applying activation function)
    logits = outputs.logits

    # Apply softmax to get probabilities (optional)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print the result
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}\n")