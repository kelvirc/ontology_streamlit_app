#from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch

with open('data/level_3_model/parent_to_label.json', 'r') as f:
    parent_to_label = json.load(f)


def predict_placement(model, tokenizer, term ,device):
    # Invert parent_to_label to map index to parent term
    idx_to_label = {idx: label for label, idx in parent_to_label.items()}

    with torch.no_grad():

        term = term#"theory of artefacts"

        # Tokenize the term
        encoding = tokenizer.encode_plus(
            term,
            add_special_tokens=True,
            max_length=80,  # Adjust based on your data
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Make prediction
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).item()


    predicted_label_mapped = [idx_to_label.get(predicted_label_idx, 'UNKNOWN')]

    # Create a list of all labels from parent_to_label
    all_labels = list(parent_to_label.keys())

    return predicted_label_mapped
    #print(f"Term: {term}\nPredicted Level 3 Placement: {predicted_label_mapped}\n")