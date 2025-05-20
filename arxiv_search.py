# Anish Kochhar, May 2025

""" Hybrid search: Chroma (offline) + arXiv API fallback """

import re, arxiv, chromadb, os, shutil
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.documents import Document as LC_Doc
from chromadb.config import Settings
from chromadb import PersistentClient

DOMAIN_QUERIES = {
    "Bioinspired Mechanics":
        ("bioinspired AND mechanics OR 'structural hierarchy'", ["cond-mat.mtrl-sci", "cond-mat.soft"]),
    "Optical Nanomaterials":
        ("plasmonic OR photonic crystals OR metasurfaces", ["physics.optics", "cond-mat.mtrl-sci"]),
    "Biomimetic Scaffolds":
        ("biomimetic scaffold OR electrospun nanofiber", ["q-bio.TO", "cond-mat.mtrl-sci"]),
    "Materials Mechanics":
        ("fracture toughness composite OR multiscale deformation", ["cond-mat.mtrl-sci"]),
    "Tissue Engineering & Angiogenesis":
        ("vascularization OR angiogenesis scaffold", ["q-bio.TO", "physics.bio-ph"]),
    "Nanoscale Assembly & Plasmonics":
        ("DNA origami plasmonic OR nanoscale self-assembly", ["cond-mat.mes-hall", "physics.optics"]),
    "Bone Biomaterials":
        ("hydroxyapatite OR collagen bone composite", ["q-bio.TO", "cond-mat.mtrl-sci"]),
    "Functional Biochemistry":
        ("enzyme-mimic nanomaterial OR catalytic biomimicry", ["q-bio.BM", "q-bio.BQ"]),
    "Hydrogels & Soft Biomaterials":
        ("hydrogel OR soft biomaterial OR double-network hydrogel", ["cond-mat.soft", "physics.bio-ph"])
}

load_dotenv()

EMBED = OpenAIEmbeddings(model="text-embedding-3-small")
DOMAIN_DIR = Path("domain_kb")
DOMAIN_DIR.mkdir(exist_ok=True)
CHROMA_DIR = DOMAIN_DIR / "chroma_store"
COLLECTION_NAME = "arxiv_kb"

# arXiv helper
def _arxiv_search(query: str, cats: List[str], max_n: int = 15) -> List[Dict[str, str]]:
    q_full = f"({query}) AND cat:{' OR cat:'.join(cats)}"
    client = arxiv.Client(page_size=max_n)
    results = client.results(arxiv.Search(query=q_full, max_results=max_n, sort_by=arxiv.SortCriterion.Relevance))

    out = []
    for res in results:
        out.append({"title": res.title, "abstract": re.sub(r"\s+"," ", res.summary)[:1500], "src": "arxiv"})
    return out

# Offline knowledge base builder / retriever
def build_chroma(force = False):
    if CHROMA_DIR.exists():
        if force:
            print("Force deleting old knowledge base...")
            shutil.rmtree(CHROMA_DIR)
        else:
            return

    from tqdm import tqdm
    client = PersistentClient(path=str(CHROMA_DIR))
    vectordb = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR), embedding_function=EMBED, client=client)

    for domain, (query, cats) in tqdm(DOMAIN_QUERIES.items()):
        docs = _arxiv_search(query, cats, 50)
        ldocs = [Document(page_content=f"{d['title']}. {d['abstract']}",
                          metadata={"domain": domain, "title": d['title'], "src": "arxiv-cache"}) for d in docs]
        vectordb.add_documents(ldocs)

def load_retriever(domain: str, k: int = 4):
    vectordb = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR), embedding_function=EMBED)
    return vectordb.as_retriever(search_kwargs={"k": k, "filter": {"domain": domain}})


# Hybrid knowledge base / arXiv search
def hybrid_search(query: str, domain: str, k: int = 4) -> List[LC_Doc]:
    retriever = load_retriever(domain)
    results = retriever.invoke(query)

    if len(results) < k:
        print("Searching arXiv for results..")
        q, cats = DOMAIN_QUERIES[domain]
        live = _arxiv_search(query + " " + q, cats, 10)
        live_docs = [Document(page_content=f"{d['title']}. {d['abstract']}",
                     metadata={"domain": domain, "title": d['title'], "src": "arxiv-live"}) for d in live[:k]]

        # Insert into store
        load_retriever(domain).vectorstore.add_documents(live_docs)
        results += live_docs

    print(f"- arXiv search results for {query} -")
    for doc in results:
        print("•", doc.metadata.get("title", "[No title]"))
    return results[:k]


if __name__=="__main__":
    build_chroma(force=False)
    docs = hybrid_search("freeze-casting hydrogels", "Hydrogels & Soft Biomaterials")
    for d in docs:
        print(d.metadata["src"],"::",d.metadata["title"])