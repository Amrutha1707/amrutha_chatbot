from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
import os
import re
import time


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

Answer the user's question using ONLY
the provided knowledge base.

STRICT RULES:

1. Answer only the question asked.

2. Do not provide unnecessary details.

3. Do not guess.

4. Do not invent information.

5. Do not use outside knowledge.

6. If the user asks for a person's name,
   give only the person's name.

7. If the user asks for a number,
   give only the number.

8. If the user asks for a date,
   give only the date.

9. If the user asks for a location,
   give only the relevant location.

10. If the answer is not available,
    say exactly:

Information not available in the knowledge base.

11. Keep answers short and direct.

12. Understand natural variations of questions.

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

    # ======================================
    # TRY GEMINI 3.5 FLASH-LITE
    # ======================================

    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash"
    ]

    last_error = None

    for model_name in models_to_try:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=100
                    )
                )

                if response.text:
                    return response.text.strip()

            except Exception as error:

                last_error = error

                print(
                    f"Gemini error "
                    f"(model={model_name}, "
                    f"attempt={attempt + 1}):",
                    error
                )

                # Wait before retrying
                time.sleep(2 ** attempt)

    # ======================================
    # ALL ATTEMPTS FAILED
    # ======================================

    print("Final Gemini error:", last_error)

    return (
        "The AI service is temporarily busy. "
        "Please try your question again in a few seconds."
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


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
        host="0.0.0.0",
        port=5000
    )