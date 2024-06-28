import os
import shutil
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_tensorboard_embeddings(embeddings, log_dir='logs/synonym_embeddings'):
    abs_log_dir = os.path.abspath(log_dir)
    if os.path.exists(abs_log_dir):
        shutil.rmtree(abs_log_dir)
        logger.info(f"Cleared existing log directory: {abs_log_dir}")
    os.makedirs(abs_log_dir)
    logger.info(f"Created log directory: {abs_log_dir}")

    embedding_values = []
    metadata = []

    for term, embedding in embeddings.items():
        embedding_values.append(embedding)
        metadata.append(term)

    embedding_values = np.array(embedding_values)
    logger.info(f"Prepared {len(embedding_values)} embeddings and metadata.")

    embedding_var = tf.Variable(embedding_values, name='synonym_embeddings')
    logger.info("Created TensorFlow variable for embeddings.")

    metadata_path = os.path.join(abs_log_dir, 'metadata.tsv')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for term in metadata:
            f.write(f"{term}\n")
    logger.info(f"Metadata written to {os.path.abspath(metadata_path)}")

    if os.path.exists(metadata_path):
        logger.info(f"Metadata file exists: {os.path.abspath(metadata_path)}")
    else:
        logger.error(f"Metadata file does NOT exist: {os.path.abspath(metadata_path)}")

    checkpoint_prefix = os.path.join(abs_log_dir, "embedding.ckpt")
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
    logger.info(f"Checkpoint created and saved to {os.path.abspath(checkpoint_path)}")

    # Check if checkpoint files exist
    checkpoint_files = [f"{checkpoint_prefix}-1.index", f"{checkpoint_prefix}-1.data-00000-of-00001"]
    for file in checkpoint_files:
        if os.path.exists(file):
            logger.info(f"Checkpoint file exists: {os.path.abspath(file)}")
        else:
            logger.error(f"Checkpoint file does NOT exist: {os.path.abspath(file)}")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.abspath(metadata_path)
    logger.info(f"Projector config set up with metadata path: {os.path.abspath(metadata_path)}")

    projector_config_path = os.path.join(abs_log_dir, 'projector_config.pbtxt')
    with open(projector_config_path, 'w') as f:
        f.write(str(config))
    logger.info(f"Projector config written to {os.path.abspath(projector_config_path)}")

    summary_writer = tf.summary.create_file_writer(abs_log_dir)
    with summary_writer.as_default():
        tf.summary.scalar('dummy', 0, step=0)
        summary_writer.flush()
    logger.info(f"Summary writer logs written to {os.path.abspath(abs_log_dir)}")

    projector.visualize_embeddings(abs_log_dir, config)
    logger.info(f"TensorBoard logs written to {os.path.abspath(abs_log_dir)}")

    # Print paths TensorBoard is trying to access
    logger.info(f"TensorBoard metadata path: {os.path.abspath(metadata_path)}")
    logger.info(f"TensorBoard checkpoint path: {os.path.abspath(checkpoint_prefix)}")
