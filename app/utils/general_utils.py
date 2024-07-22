import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertModel, BertTokenizer
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch    

@st.cache_data
def load_embeddings(filepath):
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer

def get_embeddings(terms, model, tokenizer, max_length=50): #512
    embeddings = {}
    for term in terms:
        if isinstance(term, list):
            term = " ".join(term)

        inputs = tokenizer(
            term,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            inputs = {key: val.to(model.device) for key, val in inputs.items()}
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings[term] = embedding[0]

    return embeddings

def get_most_similar(new_term_embds, overall_embeddings, top_n=5):
    similarity_scores = {}
    for term, embedding in overall_embeddings.items():
        similarity = cosine_similarity(new_term_embds.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        similarity_scores[term] = similarity

    sorted_similarity_scores = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))
    most_similar_words = list(sorted_similarity_scores.keys())[:top_n]
    return most_similar_words, sorted_similarity_scores

def is_synonym(new_term, model, tokenizer, overall_embeddings, threshold=0.7267):
    new_term_embds = get_embeddings([new_term], model, tokenizer)
    similarity_scores = {}
    for term, embedding in overall_embeddings.items():
        similarity = cosine_similarity(new_term_embds[new_term].reshape(1, -1), embedding.reshape(1, -1))[0][0]
        similarity_scores[term] = similarity
    sorted_similarity_scores = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))
    if sorted_similarity_scores and list(sorted_similarity_scores.values())[0] > threshold:
        top_synonym = list(sorted_similarity_scores.keys())[0]
        return f"is synonym to {top_synonym} with a similarity score of {list(sorted_similarity_scores.values())[0]}"
    else:
        return "No close synonyms found in the list."

def load_predict_placement_model(filepath):
    model_name = filepath  # Path to the saved model directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model =AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def save_embeddings_to_pickle(embeddings, filename):
    """
    Saves embeddings to a pickle file.

    Args:
        embeddings: A dictionary mapping terms to their corresponding embedding vectors.
        filename: The name of the pickle file to save the embeddings.
    """
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
        

def load_pickle_model(file_path):
    """
    Load a pickle model from the specified file path.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The loaded model object.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def ontology_parent_lookup(df, entity_id):
    """
    Function to find the parent value from a DataFrame given a category (cat_3).

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    cat_3 (str): The category to search for in the 'ent_id' column.

    Returns:
    str: The corresponding parent value or 'UNKNOWN' if no match is found.
    """
    # Check for the match in the 'ent_id' column and return the 'parent' value or 'UNKNOWN'
    parent_value = df['level_3'][df['ent_id'] == entity_id].values
    return parent_value[0] if parent_value else "UNKNOWN"
    