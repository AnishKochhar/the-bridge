# Anish Kochhar, The Bridge, May 2025

""" Build Chroma vector DB over 9 DOMAIN_LABELS + expose `load_retriever()` """

import json, os, re, arxiv
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma

from cluster import DOMAIN_LABELS

load_dotenv()

EMBED = OpenAIEmbeddings(model="text-embedding-3-small")
DOMAIN_DIR = Path("domain_kb")
DOMAIN_DIR.mkdir(exist_ok=True)
CHROMA_DIR = DOMAIN_DIR / "chroma_store"

def fetch_arxiv_bulk(domain_query: str, n: int = 50) -> List[Dict[str, str]]:
    client = arxiv.Client()
    search = arxiv.Search(query=domain_query, max_results=n, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = []
    for res in client.results(search):
        print(res.title)
        results.append({"title": res.title, "abstract": re.sub(r"\s+", " ", res.summary)})
    return results


def build_offline_kb(domains: List[str], force=False):
    if CHROMA_DIR.exists() and not force:
        print("Offline KB already exists")
        return
    
    client = chromadb.Client(Settings(allow_reset=True))
    client.reset()
    vectordb = Chroma(
        client=client,
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBED,
        collection_name="domain_kb",
    )

    for domain in domains:
        jfile = DOMAIN_DIR / f"{domain}.json"
        if not jfile.exists() or force:
            data = fetch_arxiv_bulk(domain, 50)
            with open(jfile, "w") as f: json.dump(data, f, indent=2)
        else:
            data = json.load(open(jfile))

        # Trn into Document
        docs = [Document(
            page_content=f"{d['title']}. {d['abstract']}", 
            metadata={"domain": domain, "title": d["title"], "src": "kb"}
        ) for d in data]
        if not docs:
            print(f"[WARN] No docs for {domain}")
        vectordb.add_documents(docs)
        print(f"Added {len(docs)} docs for {domain}")
    print("Document count:", vectordb._collection.count())
    print("Chroma vector store built")
    
def load_retriever(domain: str, k: int = 4):
    vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=EMBED, collection_name="domain_kb")
    # Filter by metadata domain
    retriever = vectordb.as_retriever(search_kwargs={"k": k, "filter": {"domain": domain}})
    return retriever


if __name__ == "__main__":
    build_offline_kb(DOMAIN_LABELS)
    ret = load_retriever("Hydrogels & Soft Biomaterials")
    docs = ret.invoke("freeze-casting hydrogels")
    print(f"Retrieved {len(docs)} docs")
    for doc in docs:
        print("-", doc.metadata["title"])
