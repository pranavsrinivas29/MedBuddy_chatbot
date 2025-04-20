from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document

from backend.llm_ollama import get_local_llm

from backend.chroma_db import setup_chroma, load_drug_data
import re
from backend.hybrid_retriever import build_hybrid_retriever

# Prompt
prompt_template = """
You are MedBuddy, a helpful and trustworthy AI medical assistant.

Based on the context below, answer the user's question clearly, concisely, and in your own words. 
Summarize any symptoms or side effects neatly and avoid copying raw text. 
If the answer is not present in the context, respond with:
"The information is not available in the current data."

Context:
{context}

Question:
{question}

Answer:
"""
COMPARISON_PROMPT = PromptTemplate(
    input_variables=["drug1", "drug2", "context"],
    template="""
You are MedBuddy, a helpful and concise AI medical assistant.

Compare the following two drugs based on the medical context below.

Drugs: {drug1} and {drug2}

Instructions:
- Mention common and differing uses.
- List key side effects, risks, or warnings.
- Highlight any significant pros or cons.
- Be clear and structured (use markdown if needed).

Context:
{context}

Answer:
"""
)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Load drug names dynamically for metadata filtering
drug_names = {entry["drug_name"] for entry in load_drug_data() if "drug_name" in entry}

def extract_drug_name(query: str) -> str:
    query = query.lower()
    for drug in drug_names:
        if drug in query:
            return drug
    return None

def create_qa_chain(user_query: str):
    llm = get_local_llm()
    vectorstore = setup_chroma()

    drug_name = extract_drug_name(user_query)
    print(f"[DEBUG] Extracted drug name for filter: {drug_name}")

    search_kwargs = {"k": 1}
    if drug_name:
        search_kwargs["filter"] = {"drug_name": drug_name}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Retrieve top-k documents
    docs = retriever.get_relevant_documents(user_query)
    print(f"[DEBUG] Retrieved {len(docs)} document(s)")
    for i, doc in enumerate(docs):
        print(f"[DOC {i+1}]:\n{doc.page_content[:300]}")

    # Truncate context for safety
    context = "\n\n".join([doc.page_content.strip() for doc in docs])[:1500]


    # Define the manual chain
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)
    qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    return qa_chain, docs

def compare_two_drugs(drug1: str, drug2: str):
    vectorstore = setup_chroma()

    # Retrieve documents for each drug
    retriever = vectorstore.as_retriever()
    docs1 = retriever.get_relevant_documents(drug1)
    docs2 = retriever.get_relevant_documents(drug2)
    combined_docs = docs1 + docs2

    # Combine and truncate context
    context = "\n\n".join(doc.page_content for doc in combined_docs)[:3000]

    # Create chain
    llm = get_local_llm()
    chain = LLMChain(llm=llm, prompt=COMPARISON_PROMPT)

    # Run chain
    response = chain.run(drug1=drug1, drug2=drug2, context=context)
    return response

EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["term"],
    template="""
You are MedBuddy, a medical assistant who explains complex medical terminology in a clear, simple, and accurate way.

Explain the following medical term or side effect in a patient-friendly way:

Term: {term}

Make it concise, helpful, and medically accurate.
"""
)
def create_explanation_chain(user_query: str):
    """
    If user_query is a short definition/explanation request (like 'what is photophobia'),
    return a temporary chain that answers it directly.
    """
    cleaned = user_query.lower()
    for prefix in ["what is", "explain", "meaning of"]:
        cleaned = cleaned.replace(prefix, "")
    term = cleaned.strip(" ?")

    llm = get_local_llm()
    chain = LLMChain(llm=llm, prompt=EXPLAIN_PROMPT)

    # Fake document list to keep app.py pipeline compatible
    return chain, [Document(page_content=term)]

def create_hybrid_qa_chain(user_query: str):
    llm = get_local_llm()
    hybrid_retriever = build_hybrid_retriever()
    docs = hybrid_retriever(user_query)

    context = "\n\n".join([doc.page_content.strip() for doc in docs])[:1500]

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)
    qa_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    return qa_chain, docs

def suggest_drugs_for_symptoms(symptom_query: str):
    from backend.chroma_db import setup_chroma
    vectorstore = setup_chroma()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(symptom_query)

    context = "\n\n".join(doc.page_content for doc in docs)[:3000]

    suggestion_prompt = PromptTemplate(
        input_variables=["context", "symptoms"],
        template="""
You are MedBuddy, an AI medical assistant.

A user described their symptoms as: "{symptoms}".

Based on the context below (which includes drug descriptions), suggest appropriate medications:
- Summarize why the drug may be suitable.
- Mention relevant conditions or benefits.
- Avoid giving definitive prescriptions.

Context:
{context}

Answer:
"""
    )

    llm = get_local_llm()
    chain = LLMChain(llm=llm, prompt=suggestion_prompt)
    return chain.run(symptoms=symptom_query, context=context)
