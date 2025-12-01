import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
import spacy
import fitz
import pymupdf4llm
import io
import re
import asyncio
from dotenv import load_dotenv
import os
import requests
import httpx
import json
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the local spaCy model
SPACY_MODEL_PATH = os.path.join(os.getcwd(), "spacy_model")

_real_spacy_load = spacy.load
def custom_load(path, *args, **kwargs):
    if path == "en_core_web_sm":
        return _real_spacy_load(SPACY_MODEL_PATH)
    return _real_spacy_load(path)

spacy.load = custom_load
# Load the spaCy model from the local directory
nlp = spacy.load(SPACY_MODEL_PATH)

import text_anonymizer.core
text_anonymizer.core.nlp = nlp

from text_anonymizer import anonymize
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(
    title="FastAPI",
    description="FastAPI",
    version="1.0.0",
)

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

MODELS = [
        "tngtech/deepseek-r1t-chimera:free",
        "openai/gpt-oss-20b:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-235b-a22b:free",
        "mistralai/mistral-small-3.2-24b-instruct",
        "tngtech/deepseek-r1t2-chimera:free",
        "qwen/qwen3-4b:free",
        "x-ai/grok-4.1-fast:free",
    ]

WHITELIST = {
    # -------------------------
    # ORIGINAL TERMS
    # -------------------------
    "FPT University", "Coursera", "English", "Japanese", "Web", "API", "OTP",
    "Google", "Servlets", "JSP", "Time", "Business", "Python", "JavaScript",
    "Java", "C++", "C#", "Ruby", "Go", "Swift", "Kotlin", "React", "Angular",
    "Vuejs", "Nodejs", "Django", "Flask", "Spring", "SQL", "NoSQL", "MongoDB",
    "PostgreSQL", "MySQL", "Git", "Docker", "Kubernetes", "AWS", "Azure",
    "Google Cloud", "Terraform", "Jenkins", "CI/CD", "Critical thinking",
    "Teamwork", "Software", "Automation", "Engineering", "deliver", "QA",
    "Quality", "Assurance", "TensorFlow", "PyTorch", "Keras", "GCP", "Linux",
    "GitHub", "Bitbucket", "Slack", "Zoom", "JIRA", "Confluence", "Redis", "Intern",
    "UK", "",

    # -------------------------
    # PROGRAMMING LANGUAGES
    # -------------------------
    "TypeScript", "Rust", "PHP", "Perl", "R", "Scala", "Elixir", "Haskell",
    "Bash", "Shell Script", "MATLAB", "Objective-C", "Fortran", "COBOL",
    "Solidity", "Dart", "Assembly", "VBA", "Prolog", "Scheme", "Lisp",
    "F#", "OCaml", "PowerShell", "Shell",

    # -------------------------
    # FRONTEND
    # -------------------------
    "HTML", "CSS", "TailwindCSS", "Bootstrap", "Sass", "Less",
    "Material UI", "Chakra UI", "Next.js", "Nuxt.js", "Svelte", "SvelteKit",
    "Webpack", "Vite", "Rollup", "Gulp", "Babel", "ESLint", "Prettier",
    "Figma", "Responsive Design", "Accessibility", "A11y", "UI", "UX",
    "Design Systems", "Storybook", "Emotion", "Styled Components",
    "Redux", "MobX", "Pinia", "Zustand", "Jest", "Vitest", "Cypress",
    "Playwright", "WebAssembly",

    # -------------------------
    # BACKEND
    # -------------------------
    "Express.js", "FastAPI", "Laravel", "Symfony", "Rails", "ASP.NET",
    "NestJS", "Hapi", "Koa", "tRPC", "Gin", "Fiber", "Actix", "Phoenix",
    "GraphQL", "REST", "gRPC", "OpenAPI", "JWT", "OAuth", "WebSockets",
    "Rate Limiting", "Load Balancing", "Nginx", "Apache", "IIS",
    "Caching", "Session Management", "Cron", "Queues", "RabbitMQ",
    "Kafka", "ActiveMQ", "ZeroMQ", "Redis Streams", "Event Sourcing",
    "CQRS", "Service Mesh", "Istio", "Linkerd",

    # -------------------------
    # MOBILE DEVELOPMENT
    # -------------------------
    "Android", "iOS", "Flutter", "React Native", "SwiftUI",
    "Jetpack Compose", "Xamarin", "KMM", "Cordova", "Ionic",

    # -------------------------
    # CLOUD SERVICES (AWS)
    # -------------------------
    "EC2", "S3", "RDS", "ECS", "EKS", "Lambda", "DynamoDB", "API Gateway",
    "CloudWatch", "CloudTrail", "Route53", "IAM", "VPC", "Kinesis",
    "SNS", "SQS", "Elastic Beanstalk", "Glue", "Athena", "Redshift",
    "Aurora", "Fargate", "Elasticache", "Lightsail",

    # -------------------------
    # CLOUD SERVICES (GCP)
    # -------------------------
    "Compute Engine", "Cloud Run", "Cloud Functions", "Firestore",
    "BigTable", "BigQuery", "Cloud SQL", "Pub/Sub", "Dataflow",
    "Composer", "Anthos", "GKE",

    # -------------------------
    # CLOUD SERVICES (AZURE)
    # -------------------------
    "Azure Functions", "AKS", "App Services", "Cosmos DB", "Blob Storage",
    "Azure SQL", "Monitor", "DevOps Pipelines", "Service Bus",

    # -------------------------
    # DEVOPS / INFRASTRUCTURE
    # -------------------------
    "CI/CD", "GitOps", "ArgoCD", "FluxCD", "Docker Compose", "Kustomize",
    "Helm", "Prometheus", "Grafana", "ELK", "EFK", "Zabbix", "Nagios",
    "Ansible", "Puppet", "Chef", "SaltStack", "Nomad", "Consul",
    "Vault", "OpenTelemetry", "Jaeger", "Sentry", "PagerDuty",
    "Splunk", "Datadog", "New Relic", "Dynatrace",

    # -------------------------
    # DATABASES — SQL + NoSQL + Graph + Time Series
    # -------------------------
    "MariaDB", "CockroachDB", "Cassandra", "SQLite", "Firebird",
    "Oracle", "DB2", "Teradata", "Snowflake",
    "CouchDB", "RethinkDB", "ArangoDB", "Neo4j", "Dgraph",
    "TigerGraph", "TimescaleDB", "InfluxDB", "Prometheus TSDB",

    # -------------------------
    # AI / ML / DATA SCIENCE
    # -------------------------
    "Machine Learning", "Deep Learning", "Neural Networks", "Transformers",
    "LLM", "RNN", "CNN", "GAN", "Autoencoder", "Diffusion Models",
    "NLP", "Computer Vision", "Reinforcement Learning",
    "XGBoost", "LightGBM", "CatBoost", "Scikit-Learn",
    "OpenCV", "NLTK", "spaCy", "HuggingFace", "FastAI",
    "MLflow", "Kubeflow", "DataRobot", "Vertex AI",
    "Feature Store", "Model Registry", "ONNX", "Quantization",
    "Pruning", "Distillation", "Model Serving", "Inference Optimization",

    # -------------------------
    # CYBERSECURITY
    # -------------------------
    "OWASP", "SQL Injection", "XSS", "CSRF", "RCE", "SSRF", "IDOR",
    "Penetration Testing", "Threat Modeling", "Encryption",
    "Hashing", "AES", "RSA", "TLS", "SSL", "Kerberos",
    "Fuzzing", "Burp Suite", "Nmap", "Metasploit", "Wireshark",
    "Firewall", "WAF", "Antivirus", "EDR", "SIEM", "SOC", "MITRE ATT&CK",
    "Zero Trust", "Identity Access Management", "IAM", "OAuth2", "SAML",

    # -------------------------
    # SYSTEM DESIGN & ARCHITECTURE
    # -------------------------
    "Load Balancer", "Reverse Proxy", "Sharding", "Replication",
    "Partitioning", "CAP Theorem", "Consistency", "Durability",
    "Scalability", "Fault Tolerance", "High Availability",
    "Distributed Systems", "Leader Election", "Consensus",
    "Raft", "Paxos", "Gossip Protocol", "Circuit Breaker Pattern",
    "Bulkhead Pattern", "Rate Limiting", "Throttling",
    "Service Discovery", "API Gateway", "Edge Computing",
    "CDN", "Caching Layer", "Event-Driven Architecture",
    "Message Broker", "Failover", "Autoscaling",

    # -------------------------
    # NETWORKING
    # -------------------------
    "TCP", "UDP", "IP", "DNS", "DHCP", "HTTP", "HTTPS", "FTP", "SSH",
    "Telnet", "VPN", "Subnetting", "Routing", "Switching",
    "Load Balancing", "Proxy", "NAT", "Firewall",

    # -------------------------
    # TESTING & QA
    # -------------------------
    "Unit Testing", "Integration Testing", "System Testing",
    "Acceptance Testing", "Regression Testing", "Smoke Testing",
    "Sanity Testing", "Performance Testing", "Load Testing",
    "Stress Testing", "Pen Testing", "API Testing",
    "Selenium", "Appium", "JUnit", "NUnit", "Mocha", "Chai",
    "TestNG", "Allure", "Postman", "Newman",

    # -------------------------
    # PROJECT / PRODUCT / AGILE
    # -------------------------
    "Agile", "Scrum", "Kanban", "Waterfall", "Lean", "OKRs",
    "KPIs", "Sprints", "Backlog", "User Stories",
    "Acceptance Criteria", "Roadmap", "Stakeholders",
    "Project Management", "Product Management",
    "Requirements Engineering", "Risk Management",

    # -------------------------
    # DEV TOOLS
    # -------------------------
    "VS Code", "IntelliJ", "Eclipse", "Android Studio",
    "Xcode", "PyCharm", "Vim", "Emacs", "Postman",
    "Fiddler", "Insomnia", "Swagger", "K6",

    # -------------------------
    # OPERATING SYSTEMS
    # -------------------------
    "Windows", "Linux", "macOS", "Ubuntu", "Debian", "Fedora",
    "Arch Linux", "CentOS", "Red Hat", "Alpine Linux",

    # -------------------------
    # VIRTUALIZATION / CONTAINERS
    # -------------------------
    "VMware", "VirtualBox", "Hyper-V", "KVM",
    "Docker", "OCI", "runc", "containerd",

    # -------------------------
    # SOFT SKILLS
    # -------------------------
    "Communication", "Leadership", "Problem Solving",
    "Teamwork", "Adaptability", "Creativity",
    "Decision Making", "Analytical Thinking",

    # -------------------------
    # EDUCATIONAL / MISC
    # -------------------------
    "Computer Science", "Information Technology", "Algorithms",
    "Data Structures", "Time Complexity", "Space Complexity",
    "Systems Analysis", "Software Engineering",
    "Quality Engineering", "Scripting", "Automation"
}

async def chat_with_fallback(
    message: str,
    models: List[str],
    max_retries: int = 1,
    timeout: int = 10
) -> str:
    """
    Try multiple models in order until one responds successfully.
    Returns the assistant's message or raises an exception if all fail.
    """
    last_exception = None

    for model in models:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=HEADERS,
                        data=json.dumps({
                            "model": model,
                            "messages": [{"role": "user", "content": message}]
                        }),
                        timeout=timeout
                    )
                response.raise_for_status()
                data = response.json()

                # Extract content (adapt based on API response structure)
                # Some APIs return `choices` array
                # logging.info(data)
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                last_exception = e
                logging.warning(f"Model {model} attempt {attempt+1} failed: {e}")
                # Try again up to max_retries

        logging.info(f"Switching to next model after {max_retries} failed attempts: {model}")

    raise RuntimeError(f"All models failed. Last exception: {last_exception}")

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
    """
    Send anonymized CV text to the LLM and return structured JSON.
    """

    # Construct the prompt
    prompt = f"""
You are a CV parsing assistant.

You will receive the text of a CV. Some personal information like names, emails, phone numbers, and addresses has already been anonymized. 

Your task is to extract structured information from the CV and output it as a JSON object with the following format:

{{
  "summary": "<short professional summary or objective>",
  "skills": ["skill1", "skill2", "..."],
  "languages": ["language1", "language2", "..."],
  "experiences": [
    {{
      "job_title": "<Job title>",
      "company": "<Company name>",
      "description": "<Brief summary of responsibilities and achievements>"
    }}
  ],
  "certifications": ["certification1", "certification2", "..."]
}}

Rules:
1. Only include information explicitly mentioned in the CV. Do not make assumptions.
2. Preserve anonymized placeholders as-is.
3. Organize experiences and education in chronological order, most recent first.
4. If a field is not available, leave it as an empty string or empty array/object.
5. Make JSON valid (use double quotes and proper syntax).
6. **Output JSON only. Do not include any text outside the JSON object.**
7. Do not add explanations, notes, or commentary.

Here is the CV text:

\"\"\"{cv_text}\"\"\"
"""

    # Send request to OpenRouter API
    answer = await chat_with_fallback(prompt, MODELS)
    answer = clean_json_string(answer)
    try:
        structured_json = json.loads(answer)
    except json.JSONDecodeError:
        structured_json = {}  # fallback

    return structured_json

async def generate_interview_questions(cv_text: str):
    """
    A background task to generate interview questions based on the CV.
    """
    prompt = f"""
You are an interview question generator.

Based on the following CV text, generate 5–10 relevant interview questions tailored to the candidate's skills, experience, and background.

Return the result as a **JSON array of strings only**, with no additional text, explanation, or wrapping.

Example format:
{{
  "questions": [
    {{
      "id": "q_001",
      "type": "type1",
      "difficulty": "difficulty1",
      "target_skill": "skill1",
      "experience_ref": "company1",
      "question": "question1",
      "answer_guide": "answer_guide1"
    }},
    {{
      "id": "q_002",
      "type": "type2",
      "difficulty": "difficulty2",
      "target_skill": "skill2",
      "experience_ref": "company2",
      "question": "question2",
      "answer_guide": "answer_guide2"
    }},
    {{
      "id": "...",
      "type": "...",
      "difficulty": "...",
      "target_skill": "...",
      "experience_ref": "...",
      "question": "...",
      "answer_guide": "..."
    }}
  ]
}}

CV Text:
\"\"\"{cv_text}\"\"\"

1. Make JSON valid (use double quotes and proper syntax).
2. **Output JSON only. Do not include any text outside the JSON object.**
3. Do not add explanations, notes, or commentary.
"""
    try:
        response = await chat_with_fallback(prompt, MODELS)
        response = clean_json_string(response)
        # For now, we just log the questions. In a real app, we might save them to a database.
        logging.info(f"[BackgroundTask] Generated Interview Questions:\n{response}")
    except Exception as e:
        logging.error(f"[BackgroundTask] Error generating interview questions: {e}")

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Async wrapper for PyMuPDF Layout parser."""
    return await asyncio.to_thread(_extract_text_layout_aware, pdf_content)


def _extract_text_layout_aware(pdf_content: bytes) -> str:
    """Extracts structured text (headings, paragraphs) using PyMuPDF-Layout."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)  # Markdown preserves headings, bold, italic, lists
    doc.close()
    return md

def anonymize_with_whitelist(text: str, whitelist: set[str]):
    anonymized_text, anonymization_map = anonymize(text)

    for placeholder, original in list(anonymization_map.items()):
        if original in whitelist:
            anonymized_text = anonymized_text.replace(placeholder, original)
            del anonymization_map[placeholder]

    return anonymized_text, anonymization_map

@app.post("/api/extract-cv")
async def extract_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = await extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")

    anonymized_text, anonymization_map = anonymize_with_whitelist(text, WHITELIST)
    sanitized = anonymized_text
    try:
        cv_json = await parse_cv_to_json(sanitized)
    except Exception as e:
        logging.error(f"Failed to parse CV to JSON: {e}")
        cv_json = {"error": f"Failed to parse CV to JSON"}

    # Add background task to run after the response is sent
    background_tasks.add_task(generate_interview_questions, cv_json)

    return cv_json

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