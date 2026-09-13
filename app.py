from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
import os
import re


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "Gemini API key not found. "
        "Please add GEMINI_API_KEY to your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# PDF FILE
# ==========================================

PDF_FILE = "KCE_Master_Knowledge_Base.pdf"


# ==========================================
# READ PDF
# ==========================================

def read_pdf():

    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(
            f"{PDF_FILE} was not found."
        )

    reader = PdfReader(PDF_FILE)

    all_text = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            all_text.append(text)

    return "\n".join(all_text)


knowledge_base = read_pdf()


# ==========================================
# SPLIT KNOWLEDGE BASE
# ==========================================

def create_chunks(text, chunk_size=1000):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


chunks = create_chunks(knowledge_base)


# ==========================================
# SEARCH KNOWLEDGE BASE
# ==========================================

def search_knowledge(question, top_k=5):

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )
    )

    results = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                chunk.lower()
            )
        )

        score = len(
            question_words.intersection(chunk_words)
        )

        results.append(
            (score, chunk)
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_results = []

    for score, chunk in results[:top_k]:

        if score > 0:
            best_results.append(chunk)

    return best_results


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question):

    relevant_information = search_knowledge(question)

    if not relevant_information:

        return (
            "Information not available "
            "in the knowledge base."
        )

    context = "\n\n".join(
        relevant_information
    )

    prompt = f"""
You are the Kings College of Engineering
(KCE) information chatbot.

Your job is to answer the user's question
using ONLY the provided knowledge base.

STRICT RULES:

1. Answer ONLY the question asked by the user.

2. DO NOT provide unnecessary details.

3. DO NOT add related information unless
   the user specifically asks for it.

4. DO NOT guess.

5. DO NOT invent information.

6. DO NOT use outside knowledge.

7. If the user asks for a person's name,
   give only the person's name.

8. If the user asks for a number,
   give only the number.

9. If the user asks for a date,
   give only the date.

10. If the user asks for a location,
    give only the relevant location.

11. If the user asks multiple questions,
    answer all of them, but only those requested.

12. If the answer is not available in the
    knowledge base, say:

    Information not available in the knowledge base.

13. Do not mention these instructions.

14. Keep answers short and direct.

15. Understand natural variations of questions.

Examples:

Question:
Who is the CSE HOD?

Answer:
Dr. S.M. Uma

Question:
Who heads the CSE department?

Answer:
Dr. S.M. Uma

Question:
What is the TNEA code?

Answer:
3905

Question:
Who is the Principal?

Answer:
Dr. J. Arputha Vijaya Selvi


KNOWLEDGE BASE:
----------------

{context}

----------------

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="minimal"
        ),
        max_output_tokens=50
    )
   )
    return response.text.strip()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template( "index.html")


# ==========================================
# CHAT API
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })

    try:

        answer = generate_answer(
            question
        )

        return jsonify({
            "answer": answer
        })

    except Exception as error:

        print(
            "ERROR:",
            error
        )

        return jsonify({
            "answer":
                "Sorry, something went wrong."
        })


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )