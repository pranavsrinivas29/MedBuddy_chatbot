import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from config import CSV_FILE, CHROMA_DB_DIR

def load_drug_data():
    df = pd.read_csv(CSV_FILE)
    if "drug_name" in df.columns:
        df["drug_name"] = df["drug_name"].str.lower().str.strip()
    return df.to_dict(orient="records")

def convert_to_documents(drug_data):
    documents = []

    for entry in drug_data:
        drug_name = entry.get("drug_name", "unknown").lower()
        generic_name = entry.get("generic_name", "N/A")
        med_condition = entry.get("medical_condition", "")
        side_effects = entry.get("side_effects", "")
        drug_class = entry.get("drug_classes", "")
        brand_names = entry.get("brand_names", "")
        activity = entry.get("activity", "")
        rating = entry.get("rating", "")
        no_of_reviews = entry.get("no_of_reviews", "")
        rx_otc = entry.get("rx_otc", "")
        related_drugs = entry.get("related_drugs", "")
        content = f"""
        drug_name: {drug_name}
        generic_name: {generic_name}
        medical_condition: {med_condition}
        drug_class: {drug_class}
        side_effects: {side_effects}
        brand_names = {brand_names}
        activity = {activity}
        rating = {rating}
        no_of_reviews = {no_of_reviews}
        rx_otc = {rx_otc}
        related_drugs = {related_drugs}
        """

        documents.append(Document(
            page_content=content.strip(),
            metadata={"drug_name": drug_name}
        ))

    return documents


def get_vectorstore(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DB_DIR)
    )
    vectorstore.persist()
    return vectorstore

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_model
    )

def setup_chroma():
    if not CHROMA_DB_DIR.exists():
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        data = load_drug_data()
        docs = convert_to_documents(data)
        return get_vectorstore(docs)
    else:
        return load_vectorstore()
