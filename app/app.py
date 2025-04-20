import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils import format_response_markdown, recommend_drugs_multi_symptoms
from backend.langchain_agent import (
    create_qa_chain,
    extract_drug_name,
    create_explanation_chain,
    compare_two_drugs,
    create_hybrid_qa_chain
)
from backend.chroma_db import load_drug_data

# Page config
st.set_page_config(page_title="MedBuddy", page_icon="💊")
st.title("💊 MedBuddy – Your Drug Assistant")

# Tabs for Chat and Compare
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Compare Drugs"])

# =========================================
# 💬 TAB 1: Chatbot UI
# =========================================
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_query" in st.session_state:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query
        st.session_state.input_box = ""
    else:
        user_query = st.text_input(
            "Ask a question about any drug:",
            placeholder="e.g. What is Compoz used for?",
            key="input_box",
        ).strip().lower()

    if user_query:
        original_query = user_query
        drug_mentioned = extract_drug_name(user_query)
        if drug_mentioned:
            st.session_state.last_drug = drug_mentioned

        if any(word in user_query.split() for word in ["its", "it", "this", "that"]):
            last_drug = st.session_state.get("last_drug")
            if last_drug:
                for vague in ["its", "it", "this", "that", "It"]:
                    user_query = user_query.replace(vague, last_drug)
                st.markdown(f"💡 MedBuddy: Interpreting Contextual Retrieval as referring to **{last_drug.capitalize()}**.")

        if ("user", original_query) not in st.session_state.chat_history:
            st.session_state.chat_history.append(("user", original_query))

        has_drug = extract_drug_name(user_query) is not None
        is_explanation_query = (
            not has_drug and
            len(user_query.split()) <= 8 and
            any(kw in user_query.lower() for kw in ["what is", "explain", "meaning of"])
        )
        is_fuzzy_query = (
            len(user_query.split()) < 5 or
            not any(keyword in user_query for keyword in ["what", "how", "can", "why", "?"])
        )
        is_symptom_query = not has_drug and not is_explanation_query

        response = None

        with st.spinner("MedBuddy is thinking..."):
            if is_explanation_query:
                qa_chain, docs = create_explanation_chain(user_query)
                response = qa_chain.run(term=user_query)

            elif is_fuzzy_query:
                qa_chain, docs = create_hybrid_qa_chain(user_query)
                response = qa_chain.run(input_documents=docs, question=user_query)

            elif is_symptom_query:
                st.markdown("🔍 Detecting potential drugs based on symptoms...")
                suggestions = recommend_drugs_multi_symptoms(user_query, pd.DataFrame(load_drug_data()))
                response = "**🤖 MedBuddy: 🧠 Suggested Drugs Based on Symptoms:**\n"
                for drug in suggestions:
                    response += f"- **{drug['drug_name']}** _(Condition: *{drug['condition']}*)_\n"
            else:
                qa_chain, docs = create_qa_chain(user_query)
                response = qa_chain.run(input_documents=docs, question=user_query)

        if response:
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()

            response = format_response_markdown(response)

            if ("bot", response) not in st.session_state.chat_history:
                st.session_state.chat_history.append(("bot", response))

        last_drug = st.session_state.get("last_drug")
        if last_drug:
            st.write("#### 💡 You can also ask:")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(f"Dosage for {last_drug}"):
                    st.session_state.pending_query = f"What is the dosage for {last_drug}?"
                    st.rerun()

            with col2:
                if st.button(f"Interactions with {last_drug}"):
                    st.session_state.pending_query = f"Are there any drug interactions with {last_drug}?"
                    st.rerun()

            with col3:
                if st.button(f"Alternatives to {last_drug}"):
                    st.session_state.pending_query = f"What are alternatives to {last_drug}?"
                    st.rerun()

    st.write("---")
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**👤 You:** {msg}")
        else:
            st.markdown(f"**🤖 MedBuddy:** {msg}")

# =========================================
# 🔍 TAB 2: Compare Drugs UI
# =========================================
with tab2:
    st.subheader("🔍 Compare Two Drugs")

    drug_data = load_drug_data()
    df = pd.DataFrame(drug_data)

    unique_conditions = sorted(df["medical_condition"].dropna().unique())
    selected_condition = st.selectbox("Select a medical condition", unique_conditions)

    condition_drugs = df[df["medical_condition"] == selected_condition]["drug_name"].dropna().unique()

    selected_drugs = st.multiselect(
        "Select two drugs to compare",
        options=condition_drugs,
        max_selections=2,
        disabled=not selected_condition
    )

    compare_triggered = st.button("Compare", disabled=len(selected_drugs) != 2)

    if compare_triggered and len(selected_drugs) == 2:
        st.markdown("🧠 Comparing drugs...")
        raw_output = compare_two_drugs(selected_drugs[0], selected_drugs[1])
        formatted_output = format_response_markdown(raw_output)
        st.markdown(formatted_output)
