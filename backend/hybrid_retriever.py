from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from backend.chroma_db import setup_chroma, load_drug_data

def build_hybrid_retriever(k_dense=2, k_sparse=2):
    # ✅ Use dense retriever from Chroma
    vectorstore = setup_chroma()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k_dense})

    # ✅ Use raw CSV drug data for sparse retriever
    drug_data = load_drug_data()
    bm25_docs = []

    for entry in drug_data:
        content = "\n".join([
            f"{k}: {str(v).strip()}"
            for k, v in entry.items()
            if k in ["drug_name", "generic_name", "medical_condition", "side_effects"]
        ])
        bm25_docs.append(Document(page_content=content, metadata={"drug_name": entry.get("drug_name", "")}))

    sparse_retriever = BM25Retriever.from_documents(bm25_docs)
    sparse_retriever.k = k_sparse

    def hybrid_search(query: str):
        dense_results = dense_retriever.get_relevant_documents(query)
        sparse_results = sparse_retriever.get_relevant_documents(query)

        # Deduplicate by content
        seen = set()
        combined = []
        for doc in dense_results + sparse_results:
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)
        return combined

    return hybrid_search
