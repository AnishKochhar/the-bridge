# Anish Kochhar, May 2025

""" Live API call to arXiv to supplement knowledge base """

from kb_builder import load_retriever, fetch_arxiv_bulk

from langchain.docstore.document import Document

def hybrid_search(query: str, domain: str, k: int = 2):
    retriever = load_retriever(domain)
    results = retriever.similarity_search(query, k=3)

    if len(results) < k:
        print("Searching arXiv for results..")
        new_docs = [Document(
            page_content=f"{d['title']}. {d['abstract']}",
            metadata={"domain": domain, "title": d["title"], "src": "arxiv"}
        ) for d in fetch_arxiv_bulk(query, 6)]

        # Insert into store
        retriever.vectorstore.add_documents(new_docs)
        results += new_docs[:k - len(results)]
    return results[:k]