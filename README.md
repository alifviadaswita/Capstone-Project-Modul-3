# 🧾 RESUME Recommendation Agent

AI-powered Resume Search & Recommendation using **LangChain + Qdrant + OpenAI**

## 🚀 Features
- Search resumes by **Category** or **Skills**
- Recommend **similar candidates**
- Retrieve resume by **Candidate ID**
- Full **RAG-based agent** with tool reasoning
- Streamlit Chat UI + Token & Cost tracking

## 🧠 Tech Stack
- **LLM**: GPT-4o Mini (OpenAI)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: Qdrant Cloud
- **Framework**: LangChain + LangGraph
- **UI**: Streamlit

## ⚙️ Setup
```bash
pip install -r requirements.txt
streamlit run app.py