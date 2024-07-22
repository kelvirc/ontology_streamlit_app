from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, maxdists
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import ast
from utils.general_utils import load_pickle_model
from utils.general_utils import load_model_and_tokenizer


#Apply BERT-tokenization
#scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
#model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model_name = "allenai/scibert_scivocab_uncased"
model, scibert_tokenizer = load_model_and_tokenizer(model_name)
scaler = StandardScaler()

def load_train(filepath):
    train_df = pd.read_csv(filepath)
    train_df["embeddings"] = train_df["embeddings"].apply(lambda x: ast.literal_eval(x))
    train_df["embeddings"] = train_df["embeddings"].apply(lambda x: np.array(x))
    embeddings_array = np.vstack(train_df["embeddings"].to_numpy())
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    Z = load_pickle_model('data/hierarchical_clustering/Z_sci.pkl')
    return embeddings_scaled,Z, train_df,  

embeddings_scaled, Z, train_df =  load_train('data/hierarchical_clustering/fv_train_df.csv')

embeddings_array = np.vstack(train_df["embeddings"].to_numpy())
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_array)


#train_df = pd.read_csv('app/data/hierarchical_clustering/fv_train_df.csv')


# Convert string representation of lists back to actual lists
#train_df["embeddings"] = train_df["embeddings"].apply(lambda x: ast.literal_eval(x))
#train_df["embeddings"] = train_df["embeddings"].apply(lambda x: np.array(x))



def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


from collections import Counter

def assign_cluster_names_frequent_value(df, clusters='cluster', parent='parent'):
    """
    Assign cluster names based on the most frequent parent in the cluster

    Parameters:
    - df: DataFrame containing the cluster and parent columns
    - cluster: column name for the cluster column
    - parent: column name for the parent column

    Returns:
    - DataFrame with the cluster name assigned to each row
    """

    cluster_names = {}
    for cluster in df[clusters].unique():
        parents = df[df[clusters] == cluster][parent]
        parent_counts = Counter(parent)
        most_common = parent_counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            cluster_names[cluster] = np.random.choice(parents)
        else:
            cluster_names[cluster] = most_common[0][0]
    df['cluster_name'] = df[clusters].map(cluster_names)
    return df



def similar_parent(centroid, parents, parent_embeddings):
    """
    Calculate parent embedding most similar to the cluster centroid

    Parameters:
    - centroid: centroid of the cluster
    - parents: a list or series of parent names
    - parent_embeddings: dictionary with parent names as keys and their corresponding embeddings as values

    Returns:
    - The parent with the highest similarity to the centroid in the cluster
    """
    unique_parents = parents.unique()
    # Reshape centroid to a 2D array if it's 1D
    if centroid.ndim == 1:
        centroid = centroid.reshape(1, -1)

    sim_score = [cosine_similarity(np.array(parent_embeddings[parent]).reshape(1,-1),centroid) for parent in unique_parents] # Convert list to NumPy array before reshaping

    best_parent_index = np.argmax(sim_score)
    return unique_parents[best_parent_index]


def predict_parent(new_entities, embeddings_scaled, Z, df, cluster_column_name='cluster'):
  """
  Predict the parent of a new entity
  Parameters:
  - new_entities: DataFrame containing ent_id and definition columns
  - df: DataFrame containing ent_id, parent, and cluster columns
  - cluster_column_name: The name of the column containing cluster assignments. Defaults to 'cluster'.

    Returns:
    - prediction of parent for each new entity
  """
  # Compute BERT embeddings for new definitions
  new_entities['embeddings'] = new_entities['definition'].apply(lambda x: get_bert_embeddings(x, model=model, tokenizer=scibert_tokenizer))
  new_embeddings = np.stack(new_entities['embeddings'].values)

  # Use the original embeddings and clusters from the previous clustering
  max_clusters = df["ent_id"].nunique()
  optimal_clusters = round(max_clusters/2)
  original_clusters = fcluster(Z, optimal_clusters, criterion='maxclust')
  df['cluster'] = original_clusters

  # Scale the new embeddings
  new_embeddings_scaled = scaler.transform(new_embeddings)

  # Find the nearest cluster for each new embedding
  distances = cdist(new_embeddings_scaled, embeddings_scaled, metric='cosine')
  nearest_clusters = original_clusters[np.argmin(distances, axis=1)]
  new_entities['nearest_cluster'] = nearest_clusters

  # Pass the cluster column name to assign_cluster_names_frequent_value
  df_most_frequent = assign_cluster_names_frequent_value(df, clusters=cluster_column_name)

  # Map the nearest clusters to cluster names
  cluster_names = df_most_frequent[[cluster_column_name, 'cluster_name']].drop_duplicates().set_index(cluster_column_name).to_dict()['cluster_name']
  new_entities['predicted_cluster_name'] = new_entities['nearest_cluster'].map(cluster_names)
  #ensuring parent name
  #Create the embedding dictionary here, ensuring it includes 'methodological entity'
  embedding_dict = {parent: get_bert_embeddings(parent, model=model, tokenizer=scibert_tokenizer ) for parent in df_most_frequent["cluster_name"].unique()}
  for i in range(len(new_entities)):
     if pd.isna(new_entities.loc[i, 'predicted_cluster_name']):
         new_parent = similar_parent(new_embeddings_scaled[i], df_most_frequent["cluster_name"], embedding_dict)
         new_entities.at[i, 'predicted_cluster_name'] = new_parent


  return(new_entities[['ent_id', 'definition', "nearest_cluster",'predicted_cluster_name']])

def predict_parent_cluster(new_ent):
    parent = predict_parent(new_entities= new_ent,
                    embeddings_scaled=embeddings_scaled, 
                    Z=Z,
                    df=train_df,
                    cluster_column_name='cluster')
    return parent


