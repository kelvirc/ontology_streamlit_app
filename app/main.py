import streamlit as st
from utils.general_utils import load_embeddings, get_most_similar, load_model_and_tokenizer, is_synonym, get_embeddings, ontology_parent_lookup
from utils.placement_lvl3 import predict_placement
from utils.general_utils import load_predict_placement_model
import torch
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.clustering import predict_parent_cluster
import ast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_synonyms(file_path):
    return load_embeddings(file_path)

@st.cache_data
def load_ontology(file_path):
    return load_embeddings(file_path)

@st.cache_data
def load_placement_classifier(file_path):
    return load_predict_placement_model(file_path)  #'app/data/level_3_model'

@st.cache_data
def load_ontology_parent():
    df = pd.read_csv('data/ontology_parent_id.csv')
    return df

# Function to find the parent value based on cat_3
def find_parent(df, cat_3):
    parent_value = df['level_3'][df['ent_id'] == cat_3].values
    return parent_value[0] if parent_value else "UNKNOWN"

# Apply custom CSS for full-screen container
st.markdown(
    """
    <style>
        /* Make the main container use the full width and height */
        .main .block-container {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if 'setup_complete' not in st.session_state:
    with st.spinner('Wait for it...'):
        # Load precomputed embeddings
        embeddings_ontology = load_ontology('data/embeddings_ontology.pkl')
        embeddings_synonyms = load_synonyms('data/embeddings_synonyms.pkl')
        # Merge all embeddings
        overall_embeddings = {**embeddings_ontology, **embeddings_synonyms}
        
        # Load SciBERT model and tokenizer with caching
        model_name = "allenai/scibert_scivocab_uncased"
        model, tokenizer = load_model_and_tokenizer(model_name)

        # Load Classifier model
        predict_cmodel, predict_c_tokenizer = load_predict_placement_model('data/level_3_model')
        device = torch.device('cpu')

        # Save to session state
        st.session_state.embeddings_ontology = embeddings_ontology
        st.session_state.embeddings_synonyms = embeddings_synonyms
        st.session_state.overall_embeddings = overall_embeddings
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.predict_cmodel = predict_cmodel
        st.session_state.predict_c_tokenizer = predict_c_tokenizer
        st.session_state.device = device
        st.session_state.setup_complete = True

        st.success(f'Setup complete: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

st.title("Automated Ontology Enhancement")
st.markdown("This prototype compares user's prompt with synonyms in the ontology to check if a new term could be a candidate for the ontology or if it is a synonym.")

# User input for model selection and word
model_choice = st.selectbox("Select a model:", options=["Ontology", "Synonyms", "Both"])
word = st.text_input("Enter the term:")

if word and 'similar_words' not in st.session_state:
    with st.spinner('Wait for it...'):
        if model_choice == "Ontology":
            selected_embeddings = st.session_state.embeddings_ontology
        elif model_choice == "Synonyms":
            selected_embeddings = st.session_state.embeddings_synonyms
        else:
            selected_embeddings = st.session_state.overall_embeddings
        
        new_term_embds = get_embeddings([word], st.session_state.model, st.session_state.tokenizer)
        logger.info(f"Generated embeddings for the word: {word}")

        # Find the most similar words
        similar_words, similarity_scores = get_most_similar(new_term_embds[word], selected_embeddings)
        logger.info(f"Found most similar words for the word: {word}")

        # Save results to session state
        st.session_state.similar_words = similar_words
        st.session_state.similarity_scores = similarity_scores
        st.session_state.word = word

if 'similar_words' in st.session_state:
    similar_words = st.session_state.similar_words
    similarity_scores = st.session_state.similarity_scores
    word = st.session_state.word

    st.write(f"Top 5 words similar to '{word}':")
    for i, similar_word in enumerate(similar_words, 1):
        st.write(f"{i}. {similar_word} (Similarity: {similarity_scores[similar_word]:.4f})")
    
    st.success(f'Synonym analysis complete: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Check if the word is a synonym
    with st.spinner('Wait for it...'):
        if model_choice == "Ontology":
            selected_embeddings = st.session_state.embeddings_ontology
        elif model_choice == "Synonyms":
            selected_embeddings = st.session_state.embeddings_synonyms
        else:
            selected_embeddings = st.session_state.overall_embeddings
            
        synonym_result = is_synonym(word, st.session_state.model, st.session_state.tokenizer, selected_embeddings)
        st.write(synonym_result)

        st.write('Potential Level 3 Placement')
        cat_3 = predict_placement(model=st.session_state.predict_cmodel, term=word, tokenizer=st.session_state.predict_c_tokenizer, device=st.session_state.device)
        st.write(f"Potential level 3 placement: '{cat_3}'")

        # Save category to session state
        st.session_state.cat_3 = cat_3
    
    st.success(f'Found potential level 3 for placement: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Ask the user for a definition
definition = st.text_area("Enter a definition for the term:")
lookup_df = load_ontology_parent()

if definition:
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('looking for parent')
    with st.spinner('Wait for it...'):
        # Find the parent for cat_3
        new_entities = pd.DataFrame({
        'ent_id': [st.session_state.cat_3],
        'definition': [definition]})
        ppc = predict_parent_cluster(new_entities)
        #st.write(ppc)
        predicted_cluster = ppc['predicted_cluster_name'].iloc[0] 
        print(predicted_cluster)
        expected_entid =  ppc['ent_id'].iloc[0]

        cluster_category = find_parent(lookup_df, predicted_cluster)

        print(cluster_category)
        print(st.session_state.cat_3)
        
        if cluster_category == st.session_state.cat_3:
             st.write(f"Your entity should be under category level 3 '{st.session_state.cat_3}' and possibly  the parent  '{predicted_cluster}'")
            
        else:
            st.write(f"Your entity should be located in the following level 3 category: {st.session_state.cat_3}")
 
    st.success(f'Found potential parent: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


