# 💊 MedBuddy – AI-Powered Drug Assistant

**MedBuddy** is a conversational medical assistant built with **LangChain**, **Streamlit**, and **LLMs (Zephyr/Ollama)**. It helps users interactively explore drug-related information, compare medications, understand medical terms, and even suggest drugs based on symptoms.

---

## 📋 Table of Contents

- [🚀 Features](#-features)
- [🛠️ Installation](#️-installation)
- [🧠 Architecture Overview](#-architecture-overview)
- [💬 Chatbot Capabilities](#-chatbot-capabilities)
- [🔍 Drug Comparison Tool](#-drug-comparison-tool)
- [🧾 Explanation Queries](#-explanation-queries)
- [🩺 Symptom-Based Drug Suggestions](#-symptom-based-drug-suggestions)

---

## 🚀 Features

- ✅ Ask about drug usage, side effects, and interactions.
- ✅ Compare two drugs side by side.
- ✅ Understand complex medical terms (e.g., “What is teratogenic?”).
- ✅ Get drug suggestions based on symptoms (e.g., "I have insomnia and depression").
- ✅ Follow-up suggestions for interactive conversations.
- ✅ Hybrid retrieval: dense (Chroma) + sparse (BM25) for fuzzy queries.
- ✅ Contextual memory (resolve “it”, “this”, etc.).
- ✅ Fully local deployment with **Ollama** and **Mistral**.

---

## 🛠️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/pranavsrinivas29/MedBuddy_chatbot.git
cd MedBuddy_chatbot
```

Step 2: Set up virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

Step 4: Install Ollama (for local model usage)
```bash
# macOS
brew install ollama

# OR via curl
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
Step 5: Pull the Mistral model locally
ollama pull zephyr:7b-beta
```

```bash
Step 6: Run the Streamlit app
streamlit run app/app.py
```

## 🧠 Architecture Overview

```text
User Query
   ↓
Streamlit Frontend
   ↓
LangChain Agent (Decision Logic)
   ├── Explanation Chain (LLM only)
   ├── Chroma QA Chain (Dense Vector Retrieval)
   ├── Hybrid Chain (BM25 + Dense for Fuzzy Queries)
   └── Symptom-Based Chain (TF-IDF Similarity Search)
         ↓
       LLM (via Ollama / Zephyr)
         ↓
     Response Display (Streamlit UI)
```

## 💬 Chatbot Capabilities
- Direct Questions: What is Accutane used for?
- Vague References: What are its side effects? → resolves “its” to last drug
- Explanation: Explain serotonin syndrome
- Fuzzy Queries: Help for nausea → hybrid dense + sparse search
- Follow-ups: Dosage, interactions, alternatives

## 🔍 Drug Comparison Tool
- Available under "Compare Drugs" tab:
- Select condition → Filter available drugs
- Pick two drugs → Click Compare
- Summarized output via LLM with pros/cons

## 🧾 Explanation Queries
- Uses LLM directly without Chroma.
Examples:
- What is neurotoxicity?
- Explain hepatotoxic

## 🩺 Symptom-Based Drug Suggestions
- TF-IDF based similarity match with dataset
- Query: I have insomnia and anxiety
Output:
🤖 MedBuddy: Suggested Drugs Based on Symptoms:
- **Zolpidem** (Condition: *Insomnia*)
- **Sertraline** (Condition: *Depression*, matched: *anxiety*)

