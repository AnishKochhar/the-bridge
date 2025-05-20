# Anish Kochhar, The Bridge, May 2025

""" 9-cluster k means over node mebeddings. Exposes `select_anchors()` """

import pickle, random
from pathlib import Path
from typing import Dict, Tuple, List

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans  

DOMAIN_LABELS = [
    "Bioinspired Mechanics",
    "Optical Nanomaterials",
    "Biomimetic Scaffolds",
    "Materials Mechanics",
    "Tissue Engineering & Angiogenesis",
    "Nanoscale Assembly & Plasmonics",
    "Bone Biomaterials",
    "Functional Biochemistry",
    "Hydrogels & Soft Biomaterials",
]

DATA_DIR = Path("data/GRAPHDATA")
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True)
GRAPH_FN  = DATA_DIR / "graph.graphml"
EMB_FN    = DATA_DIR / "node_embeddings.pkl"
CLUSTER_PKL = CACHE_DIR / "kmeans_9clusters.pkl"
MAPPING_PKL = CACHE_DIR / "node2cluster.pkl"

def compute_and_cache_clusters(n_clusters: int = 9) -> None:
    """ K-means on embeddings -> cache centroids + mapping (node -> cluster) """
    if CLUSTER_PKL.exists() and MAPPING_PKL.exists():
        print("· Clusters already cached")
        return

    print("Loading graph and embeddings..")
    G = nx.read_graphml(GRAPH_FN)
    with open(EMB_FN, "rb") as f:
        node_embeddings: Dict[str, np.ndarray] = pickle.load(f)
    
    X = np.vstack(list(node_embeddings.values()))
    print(X.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)

    mapping = {node: int(label) for node, label in zip(node_embeddings.keys(), labels)}
    with open(CLUSTER_PKL, "wb") as f: pickle.dump(kmeans, f)
    with open(MAPPING_PKL, "wb") as f: pickle.dump(mapping, f)
    print("Saved cluster model & mapping.")

def _load_clusters() -> Tuple[KMeans, Dict[str,int]]:
    if not CLUSTER_PKL.exists(): compute_and_cache_clusters()
    with open(CLUSTER_PKL, "rb") as f: km = pickle.load(f)
    with open(MAPPING_PKL, "rb") as f: mapping = pickle.load(f)
    return km, mapping



def select_anchors(domain1: str, domain2: str) -> Tuple[str, str]:
    """ Choose anchor term from each domain label """
    km, mapping = _load_clusters()
    domain_map = {label: i for i, label in enumerate(DOMAIN_LABELS)}
    clust1, clust2 = domain_map[domain1], domain_map[domain2]

    nodes_by_cluster: Dict[int, List[str]] = {}
    for n, cid in mapping.items():
        nodes_by_cluster.setdefault(cid, []).append(n)
    
    # print(f"{domain1} nodes: {len(nodes_by_cluster.get(clust1, list(mapping.keys())))}, {domain2} nodes: {len(nodes_by_cluster.get(clust2, list(mapping.keys())))}")

    anchor1 = random.choice(nodes_by_cluster.get(clust1, list(mapping.keys())))
    anchor2 = random.choice(nodes_by_cluster.get(clust2, list(mapping.keys())))
    return anchor1, anchor2


if __name__ == "__main__":
    compute_and_cache_clusters()
    dom1, dom2 = random.choice(DOMAIN_LABELS), random.choice(DOMAIN_LABELS)
    a, b = select_anchors(dom1, dom2)

    print(f"Domains:, {dom1} | {dom2}")
    print(f"Anchors: {a} | {b}")
