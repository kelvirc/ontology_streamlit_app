import os
import streamlit as st
from app.utils import load_embeddings, get_embeddings, get_most_similar, load_model_and_tokenizer, is_synonym
from app.tensorboard_utils import prepare_tensorboard_embeddings
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        /* Full-screen iframe */
        .tensorboard-iframe {
            position: relative;
            width: 100%;
            height: 0;
            padding-bottom: 75%; /* Aspect ratio */
        }
        .tensorboard-iframe iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load precomputed embeddings
embeddings_ontology = load_embeddings('data/embeddings_ontology.pkl')
embeddings_synonyms = load_embeddings('data/embeddings_synonyms.pkl')
logger.info("Loaded precomputed embeddings.")

# Merge all embeddings
overall_embeddings = {**embeddings_ontology, **embeddings_synonyms}
logger.info("Merged all embeddings.")

# Load SciBERT model and tokenizer with caching
model_name = "allenai/scibert_scivocab_uncased"
model, tokenizer = load_model_and_tokenizer(model_name)
logger.info(f"Loaded model and tokenizer: {model_name}")

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

# Display TensorBoard for embeddings
if st.button('Launch TensorBoard for Synonyms'):
    prepare_tensorboard_embeddings(embeddings_synonyms)
    log_dir = os.path.abspath('logs/synonym_embeddings')
    logger.info(f"Launching TensorBoard with logdir: {log_dir}")
    subprocess.Popen(['tensorboard', '--logdir', log_dir, '--host', '0.0.0.0', '--port', '6006'])
    time.sleep(5)  # Give TensorBoard a few seconds to start
    st.markdown(
        """
        <div class="tensorboard-iframe">
            <iframe src="http://localhost:6006/#projector" frameborder="0"></iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
    logger.info("TensorBoard launched and iframe created.")
