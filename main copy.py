import streamlit as st
from app.utils import load_embeddings, get_embeddings, get_most_similar, load_model_and_tokenizer, is_synonym
import torch

# Load precomputed embeddings
embeddings_ontology = load_embeddings('data/embeddings_ontology.pkl')
embeddings_synonyms = load_embeddings('data/embeddings_synonyms.pkl')

# Merge all embeddings
overall_embeddings = {**embeddings_ontology, **embeddings_synonyms}

# Load SciBERT model and tokenizer with caching
model_name = "allenai/scibert_scivocab_uncased"
model, tokenizer = load_model_and_tokenizer(model_name)

st.title("Word Similarity Finder")

# User input for model selection and word
model_choice = st.selectbox("Select a model:", options=["Ontology", "Synonyms", "Both"])
word = st.text_input("Enter a word:")

if word:
    if model_choice == "Ontology":
        selected_embeddings = embeddings_ontology
    elif model_choice == "Synonyms":
        selected_embeddings = embeddings_synonyms
    else:
        selected_embeddings = overall_embeddings
    
    new_term_embds = get_embeddings([word], model, tokenizer)

    # Find the most similar words
    similar_words, similarity_scores = get_most_similar(new_term_embds[word], selected_embeddings)

    if similar_words:
        st.write(f"Top 5 words similar to '{word}':")
        for i, similar_word in enumerate(similar_words, 1):
            st.write(f"{i}. {similar_word} (Similarity: {similarity_scores[similar_word]:.4f})")
    else:
        st.write(f"No embeddings found for the word '{word}'.")

    # Check if the word is a synonym
    synonym_result = is_synonym(word, model, tokenizer, selected_embeddings)
    st.write(synonym_result)
