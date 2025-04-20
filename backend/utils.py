import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.chroma_db import load_drug_data
import pandas as pd

def format_response_markdown(text: str) -> str:
    text = text.strip()

    # Highlight critical warnings
    text = re.sub(
        r"(birth defects|do not.*pregnant|consult.*doctor|serious allergic reactions|seek.*immediate.*attention)",
        r"\n\n⚠️ **\1**", text, flags=re.IGNORECASE
    )

    # Bullet common phrases for side effects
    bullet_patterns = [
        r"(side effects (can|may|might)? (include|cause|involve|be).*?):\s*(.+?)(?:\.\s|$)",
        r"(may (include|cause|lead to))\s+(.+?)(?:\.\s|$)",
        r"(common (effects|symptoms) (are|include))\s+(.+?)(?:\.\s|$)",
        r"(this drug (can|may|might) (cause|lead to|result in))\s+(.+?)(?:\.\s|$)",
    ]

    for pattern in bullet_patterns:
        text = re.sub(
            pattern,
            lambda m: f"**{m.group(1).capitalize()}**:\n" +
                      ''.join(f"- {item.strip()}\n" for item in m.groups()[-1].split(",") if item.strip()),
            text,
            flags=re.IGNORECASE
        )

    # Bold headings like "Uses:", "Dosage:", "Interactions:"
    text = re.sub(r"(^|\n)([A-Z][^\n:]{3,40}):", r"\1**\2:**", text)

    # Convert plain URLs to clickable markdown
    text = re.sub(r"(https?://\S+)", r"<\1>", text)

    return text.strip()

def recommend_drugs_multi_symptoms(symptoms: str, df: pd.DataFrame, top_k: int = 3):
    symptoms_list = [s.strip().lower() for s in symptoms.split(" and ") if s.strip()]
    drug_suggestions = []

    for symptom in symptoms_list:
        df_clean = df.dropna(subset=["medical_condition", "drug_name"])
        conditions = df_clean["medical_condition"].astype(str).str.lower()
        
        vectorizer = TfidfVectorizer()
        condition_vectors = vectorizer.fit_transform(conditions)
        symptom_vec = vectorizer.transform([symptom])

        similarities = cosine_similarity(symptom_vec, condition_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        top_drugs = df_clean.iloc[top_indices][["drug_name", "medical_condition"]]

        drug_suggestions.extend(
            {
                "drug_name": row["drug_name"],
                "condition": row["medical_condition"],
                "symptom": symptom
            }
            for _, row in top_drugs.iterrows()
        )

    return drug_suggestions
