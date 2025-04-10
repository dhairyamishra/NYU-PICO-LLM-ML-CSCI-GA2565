# utils/MonosemanticInit.py (new file you can add)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # or however many physical cores you have
import torch
from sklearn.cluster import KMeans

def prepare_monosemantic_info(model, num_clusters=100, device="cpu"):
    """
    Extract embedding matrix from model and cluster it using K-Means.
    """
    with torch.no_grad():
        embedding_matrix = model.embedding.weight.detach().to("cpu")  # (vocab_size, embed_dim)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embedding_matrix.numpy())

    return {
        "embedding_matrix": embedding_matrix,
        "kmeans": kmeans
    }
