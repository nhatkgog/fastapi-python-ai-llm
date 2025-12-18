import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
import fitz
import pymupdf4llm
import asyncio
from dotenv import load_dotenv
import httpx
import json
import os
from google import genai
from openai import OpenAI
from typing import Optional

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

hg_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

LAST_PARSED_CV: Optional[dict] = None
history = []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="FastAPI",
    description="FastAPI",
    version="1.0.0",
)

def clean_json_string(s: str) -> str:
    """
    Extract the first top-level JSON object from a string.
    Handles nested braces properly.
    """
    brace_count = 0
    start_idx = None

    for idx, char in enumerate(s):
        if char == '{':
            if brace_count == 0:
                start_idx = idx
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                # Extract substring and return
                return s[start_idx:idx+1]

    # Fallback: return original string if no JSON found
    return s

async def parse_cv_to_json(cv_text: str) -> dict:
    prompt = f"""
You are a CV parsing assistant.

You will receive the text of a CV. Some personal information like names, emails, phone numbers, and addresses has already been anonymized. 

Your task is to extract structured information from the CV and output it as a JSON object with the following format:

{{
  "apply_for": {{"job_title": ""}},
  "skills": [],
  "languages": [],
  "experiences": [],
  "certifications": []
}}

Rules:
1. Only include information explicitly mentioned in the CV.
2. Preserve anonymized placeholders as-is.
3. Organize experiences in chronological order, most recent first.
4. If a field is not available, leave it empty or as an empty array/object.
5. Make JSON valid.
6. Output JSON only. No text outside JSON.
7. Do not add explanations or commentary.

Here is the CV text:

\"\"\"{cv_text}\"\"\"
"""

    response = gemini_client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    structured_json = json.loads(clean_json_string(response.text))

    return structured_json

# async def generate_interview_questions(cv_text: str):
#     """
#     A background task to generate interview questions based on the CV.
#     """
#     prompt = f"""
# You are an interview question generator.
#
# Based on the following CV text, generate 5–10 relevant interview questions tailored to the candidate's skills, experience, and background.
#
# Return the result as a **JSON array of strings only**, with no additional text, explanation, or wrapping.
#
# Example format:
# {{
#   "questions": [
#     {{
#       "id": "q_001",
#       "type": "type1",
#       "difficulty": "difficulty1",
#       "target_skill": "skill1",
#       "experience_ref": "company1",
#       "question": "question1",
#       "answer_guide": "answer_guide1"
#     }},
#     {{
#       "id": "q_002",
#       "type": "type2",
#       "difficulty": "difficulty2",
#       "target_skill": "skill2",
#       "experience_ref": "company2",
#       "question": "question2",
#       "answer_guide": "answer_guide2"
#     }},
#     {{
#       "id": "...",
#       "type": "...",
#       "difficulty": "...",
#       "target_skill": "...",
#       "experience_ref": "...",
#       "question": "...",
#       "answer_guide": "..."
#     }}
#   ]
# }}
#
# CV Text:
# \"\"\"{cv_text}\"\"\"
#
# 1. Make JSON valid (use double quotes and proper syntax).
# 2. **Output JSON only. Do not include any text outside the JSON object.**
# 3. Do not add explanations, notes, or commentary.
# """
#     try:
#         response = await chat_with_fallback(prompt)
#         response = clean_json_string(response)
#         # For now, we just log the questions. In a real app, we might save them to a database.
#         logging.info(f"[BackgroundTask] Generated Interview Questions:\n{response}")
#     except Exception as e:
#         logging.error(f"[BackgroundTask] Error generating interview questions: {e}")

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Async wrapper for PyMuPDF Layout parser."""
    return await asyncio.to_thread(_extract_text_layout_aware, pdf_content)


def _extract_text_layout_aware(pdf_content: bytes) -> str:
    """Extracts structured text (headings, paragraphs) using PyMuPDF-Layout."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)  # Markdown preserves headings, bold, italic, lists
    doc.close()
    return md

@app.post("/api/extract-cv")
async def extract_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global LAST_PARSED_CV
    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = await extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")

    try:
        cv_json = await parse_cv_to_json(text)
        LAST_PARSED_CV = cv_json
        # Add background task to run after the response is sent
        # background_tasks.add_task(generate_interview_questions, cv_json)
    except Exception as e:
        logging.error(f"Failed to parse CV to JSON: {e}")
        cv_json = {"error": f"Failed to parse CV to JSON"}
        LAST_PARSED_CV = None

    return cv_json

@app.get("/api/last-cv")
async def last_cv():
    if LAST_PARSED_CV is None:
        return {"error": "No CV has been parsed yet"}
    return LAST_PARSED_CV

@app.get("/api/history")
async def interview_history():
    if not history:
        return {"error": "No interview is ongoing yet"}
    return history

@app.post("/api/question")
async def questioning(user_answer=None):

    prompt = f"""
    You are an IT interviewer.

    Use the CV below as the **only source** for questions. Only ask about information that is present and non‑empty.

    <CV>
    {LAST_PARSED_CV}
    </CV>

    Before asking a question:
    - If a field (e.g., skill, experience description) is missing or empty, DO NOT ask about it.
    - Only generate a question that refers to data that is actually present in the CV.

    Interview structure (strict):
    - Self intro: 10%
    - CV & projects: 25%
    - Technical & problem‑solving: 55%
    - Mindset & teamwork: 10%

    Rules:
    - Ask **one question per turn**
    - Do NOT include evaluation/instructions in the output
    - Output ONLY the next interview question
    - Questions must reference the CV
    - Skip any fields that are missing or empty
    - Increase difficulty gradually

    If this is the first turn, ask the candidate to briefly introduce themselves.
    Otherwise, generate the next question based on the prior transcript and the available fields.

    Output format:
    Your next question here?
    """
    if not history and user_answer is None:
    # first prompt only
        history.append({"role": "user", "content": prompt})

    # if we already have interview started and have user answer
    elif user_answer is not None:
    # add the candidate's answer
        history.append({"role": "user", "content": user_answer})
    response = hg_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=history
    )
    assistant_text = response.choices[0].message.content
    role = response.choices[0].message.role

    history.append({"role": role, "content": assistant_text})
    return {
        "question": assistant_text
    }

@app.post("/api/history/clear")
async def clear_history():
    global history
    history = []
    return {"status": "success", "message": "Interview history cleared."}

@app.get("/api/data")
def get_sample_data():
    return {
        "data": [
            {"id": 1, "name": "Sample Item 1", "value": 100},
            {"id": 2, "name": "Sample Item 2", "value": 200},
            {"id": 3, "name": "Sample Item 3", "value": 300}
        ],
        "total": 3,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/api/items/{item_id}")
def get_item(item_id: int):
    return {
        "item": {
            "id": item_id,
            "name": "Sample Item " + str(item_id),
            "value": item_id * 100
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI</title>
        <link rel="icon" type="image/x-icon" href="/favicon.ico">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background-color: #000000;
                color: #ffffff;
                line-height: 1.6;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            header {
                border-bottom: 1px solid #333333;
                padding: 0;
            }
            
            nav {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                padding: 1rem 2rem;
                gap: 2rem;
            }
            
            .logo {
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
                text-decoration: none;
            }
            
            .nav-links {
                display: flex;
                gap: 1.5rem;
                margin-left: auto;
            }
            
            .nav-links a {
                text-decoration: none;
                color: #888888;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.2s ease;
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            .nav-links a:hover {
                color: #ffffff;
                background-color: #111111;
            }
            
            main {
                flex: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 4rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .hero {
                margin-bottom: 3rem;
            }
            
            .hero-code {
                margin-top: 2rem;
                width: 100%;
                max-width: 900px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }
            
            .hero-code pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: left;
                grid-column: 1 / -1;
            }
            
            h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(to right, #ffffff, #888888);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .subtitle {
                font-size: 1.25rem;
                color: #888888;
                margin-bottom: 2rem;
                max-width: 600px;
            }
            
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                width: 100%;
                max-width: 900px;
            }
            
            .card {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                transition: all 0.2s ease;
                text-align: left;
            }
            
            .card:hover {
                border-color: #555555;
                transform: translateY(-2px);
            }
            
            .card h3 {
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #ffffff;
            }
            
            .card p {
                color: #888888;
                font-size: 0.875rem;
                margin-bottom: 1rem;
            }
            
            .card a {
                display: inline-flex;
                align-items: center;
                color: #ffffff;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                padding: 0.5rem 1rem;
                background-color: #222222;
                border-radius: 6px;
                border: 1px solid #333333;
                transition: all 0.2s ease;
            }
            
            .card a:hover {
                background-color: #333333;
                border-color: #555555;
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background-color: #0070f3;
                color: #ffffff;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-bottom: 2rem;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                background-color: #00ff88;
                border-radius: 50%;
            }
            
            pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 1rem;
                overflow-x: auto;
                margin: 0;
            }
            
            code {
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.5;
                color: #ffffff;
            }
            
            /* Syntax highlighting */
            .keyword {
                color: #ff79c6;
            }
            
            .string {
                color: #f1fa8c;
            }
            
            .function {
                color: #50fa7b;
            }
            
            .class {
                color: #8be9fd;
            }
            
            .module {
                color: #8be9fd;
            }
            
            .variable {
                color: #f8f8f2;
            }
            
            .decorator {
                color: #ffb86c;
            }
            
            @media (max-width: 768px) {
                nav {
                    padding: 1rem;
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .nav-links {
                    margin-left: 0;
                }
                
                main {
                    padding: 2rem 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .hero-code {
                    grid-template-columns: 1fr;
                }
                
                .cards {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <nav>
                <a href="/" class="logo">Vercel + FastAPI</a>
                <div class="nav-links">
                    <a href="/docs">API Docs</a>
                    <a href="/api/data">API</a>
                </div>
            </nav>
        </header>
        <main>
            <div class="hero">
                <h1>Vercel + FastAPI</h1>
                <div class="hero-code">
                    <pre><code><span class="keyword">from</span> <span class="module">fastapi</span> <span class="keyword">import</span> <span class="class">FastAPI</span>

<span class="variable">app</span> = <span class="class">FastAPI</span>()

<span class="decorator">@app.get</span>(<span class="string">"/"</span>)
<span class="keyword">def</span> <span class="function">read_root</span>():
    <span class="keyword">return</span> {<span class="string">"Python"</span>: <span class="string">"on Vercel"</span>}</code></pre>
                </div>
            </div>
            
            <div class="cards">
                <div class="card">
                    <h3>Interactive API Docs</h3>
                    <p>Explore this API's endpoints with the interactive Swagger UI. Test requests and view response schemas in real-time.</p>
                    <a href="/docs">Open Swagger UI →</a>
                </div>
                
                <div class="card">
                    <h3>Sample Data</h3>
                    <p>Access sample JSON data through our REST API. Perfect for testing and development purposes.</p>
                    <a href="/api/data">Get Data →</a>
                </div>
                
            </div>
        </main>
    </body>
    </html>
    """