import streamlit as st
from utils.general_utils import load_embeddings, get_embeddings, get_most_similar, load_model_and_tokenizer, is_synonym
from utils.placement_lvl3 import predict_placement
from utils.general_utils import load_predict_placement_model
import torch
import logging
from datetime import datetime


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

# Load precomputed embeddings
embeddings_ontology = load_ontology('data/embeddings_ontology.pkl')
embeddings_synonyms = load_synonyms('data/embeddings_synonyms.pkl')
#logger.info("Loaded precomputed embeddings.")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Loaded precomputed embeddings.")

# Merge all embeddings
overall_embeddings = {**embeddings_ontology, **embeddings_synonyms}
#logger.info("Merged all embeddings.")
print("Merged all embeddings.")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Load SciBERT model and tokenizer with caching
model_name = "allenai/scibert_scivocab_uncased"
model, tokenizer = load_model_and_tokenizer(model_name)
#logger.info(f"Loaded model and tokenizer: {model_name}")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"Loaded model and tokenizer: {model_name}")



# Load Classifier model
predict_cmodel, predict_c_tokenizer = load_predict_placement_model('data/level_3_model')
device = torch.device('cpu')
#logger.info(f"Loaded model and tokenizer: predict classifier model")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Loaded model and tokenizer: predict classifier model")


st.title("Automated Ontology Enhancement")

st.markdown("This prototype compares user's prompt with synonyms in the ontology to check if a new term could be a candidate for the ontology or if it is a synonym.")

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
    logger.info(f"Generated embeddings for the word: {word}")

    # Find the most similar words
    similar_words, similarity_scores = get_most_similar(new_term_embds[word], selected_embeddings)
    logger.info(f"Found most similar words for the word: {word}")

    if similar_words:
        st.write(f"Top 5 words similar to '{word}':")
        for i, similar_word in enumerate(similar_words, 1):
            st.write(f"{i}. {similar_word} (Similarity: {similarity_scores[similar_word]:.4f})")
    else:
        st.write(f"No embeddings found for the word '{word}'.")

    # Check if the word is a synonym
    synonym_result = is_synonym(word, model, tokenizer, selected_embeddings)
    st.write(synonym_result)

    st.write('Potential Level 3 Placement')
    cat_3 = predict_placement(model=predict_cmodel,
                              term=word,
                              tokenizer=predict_c_tokenizer,
                              device= device)
    st.write(f"Potential level 3 placement'{cat_3}'.")

