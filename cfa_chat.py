import streamlit as st
import json
import os
from datetime import datetime
from openai import OpenAI
from pymongo import MongoClient

# Configuration — reads from Streamlit secrets (cloud) or .env (local)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
FINE_TUNED_MODEL = st.secrets.get("FINE_TUNED_MODEL", os.environ.get("FINE_TUNED_MODEL", "ft:gpt-4o-mini-2024-07-18:northstar:cfa-expert-v2:DIC778WZ"))
MONGO_URI = st.secrets.get("MONGO_URI", os.environ.get("MONGO_URI", ""))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_LOG = os.path.join(BASE_DIR, "questions_log.json")

client = OpenAI(api_key=OPENAI_API_KEY)

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["cfa_app"]
questions_collection = db["question_counts"]


def load_questions_log():
    if os.path.exists(QUESTIONS_LOG):
        with open(QUESTIONS_LOG, "r") as f:
            return json.load(f)
    return []


def save_question(question, answer):
    log = load_questions_log()
    log.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
    })
    with open(QUESTIONS_LOG, "w") as f:
        json.dump(log, f, indent=2)

    # Update question count in MongoDB
    questions_collection.update_one(
        {"_id": "global_counter"},
        {
            "$inc": {"total_questions": 1},
            "$set": {"last_question_at": datetime.now().isoformat()},
        },
        upsert=True,
    )


def get_question_count():
    doc = questions_collection.find_one({"_id": "global_counter"})
    return doc["total_questions"] if doc else 0


def ask_cfa_model(question):
    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": "You are a CFA (Chartered Financial Analyst) expert. Provide accurate, detailed answers to CFA-related questions."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# Streamlit UI
st.set_page_config(page_title="CFA Expert Q&A", page_icon="📊", layout="wide")
st.title("📊 CFA Expert Q&A")
st.caption("Powered by a fine-tuned GPT model trained on CFA material")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a CFA question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_cfa_model(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save to local log
    save_question(question, answer)

# Sidebar: question history
with st.sidebar:
    st.header("Question History")
    log = load_questions_log()
    if log:
        for i, entry in enumerate(reversed(log)):
            with st.expander(f"{entry['question'][:60]}...", expanded=False):
                st.caption(entry["timestamp"])
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
        if st.button("Clear History"):
            with open(QUESTIONS_LOG, "w") as f:
                json.dump([], f)
            st.rerun()
    else:
        st.info("No questions asked yet.")
