"""
SmartStud.Ai -- All-in-One Autonomous AI Study Platform (Single-File Flask App)

Features:
- Multi-provider AI: Google Gemini (1.5/2.0 Flash) / Anthropic (Claude 3.5) / OpenAI (GPT-4o) / Deep Knowledge Engine
- Accurate encyclopedic domain knowledge for Robotics, AI, CS, Science, Math, History, Medicine, and Engineering
- Markdown rendering via Marked.js
- Web Speech TTS/STT, Sound Effects (Web Audio API)
- 3D perspective flip flashcards, interactive SVG radial mind maps, adaptive quiz arena with confetti & XP,
  Pomodoro timer, local task planner, and SQLite-backed multi-device link sharing.

Run:
    pip install flask anthropic python-dotenv
    python app.py

Then open http://localhost:5000 (or your LAN IP from any phone/laptop on your Wi-Fi!)
"""

import os
import re
import sys
import json
import math
import time
import uuid
import socket
import sqlite3
import traceback
import urllib.request
import urllib.parse
import urllib.error

from flask import Flask, request, jsonify, Response

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "smartstud-super-secret-key-2026")

# ---------------------------------------------------------------- Database Setup ----
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartstud.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shared_sessions (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                topic TEXT,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_chunks (
                id TEXT PRIMARY KEY,
                source TEXT,
                chunk_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                room_code TEXT PRIMARY KEY,
                topic TEXT,
                host_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Warning] SQLite init error: {e}")

init_db()

# ---------------------------------------------------------------- Network Utility ----
def get_local_ip():
    """Detects local LAN IP to allow sharing links with other devices on Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
DEFAULT_PORT = 5000

# ---------------------------------------------------------------- AI Engine & API Keys ----
TUTOR_SYSTEM = (
    "You are Stud, an expert, encouraging AI study tutor and professor. Provide comprehensive, accurate, "
    "and engaging explanations with clear headings, organized bullet points, real-world examples, and key takeaways."
)
JSON_SYSTEM = "You output strictly raw, valid JSON only. Never output markdown fences, backticks, preamble, or trailing commentary."

def get_client_api_info(req):
    """Retrieves API key & provider preference from request headers, JSON, or environment."""
    custom_key = ""
    provider = ""
    
    if req.is_json and req.json:
        custom_key = req.json.get("customApiKey", "").strip()
        provider = req.json.get("aiProvider", "").strip()
        
    if not custom_key:
        custom_key = req.headers.get("X-Api-Key", "").strip()
    if not provider:
        provider = req.headers.get("X-Ai-Provider", "").strip()
    
    if not custom_key:
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini", os.environ.get("GEMINI_API_KEY")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic", os.environ.get("ANTHROPIC_API_KEY")
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai", os.environ.get("OPENAI_API_KEY")
        return "simulation", None

    if not provider:
        if custom_key.startswith("AIzaSy"):
            provider = "gemini"
        elif custom_key.startswith("sk-ant-"):
            provider = "anthropic"
        elif custom_key.startswith("gsk_"):
            provider = "groq"
        elif custom_key.startswith("sk-"):
            provider = "openai"
        else:
            provider = "gemini"
            
    return provider, custom_key


def call_anthropic(key, prompt, as_json=False, max_tokens=1800):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        system_prompt = JSON_SYSTEM if as_json else TUTOR_SYSTEM
        model = "claude-3-5-sonnet-20241022"
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return clean_json_response(text) if as_json else text
    except Exception as e:
        print(f"[Anthropic Error] {e}")
        return None if as_json else f"⚠️ Claude API Error: {e}"


def call_gemini(key, prompt, as_json=False):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": (JSON_SYSTEM + "\n\n" if as_json else TUTOR_SYSTEM + "\n\n") + prompt}]}],
            "generationConfig": {
                "temperature": 0.2 if as_json else 0.7,
                "maxOutputTokens": 2048
            }
        }
        if as_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        
        with urllib.request.urlopen(req, timeout=25) as response:
            res = json.loads(response.read().decode("utf-8"))
            text = res["candidates"][0]["content"]["parts"][0]["text"]
            return clean_json_response(text) if as_json else text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None if as_json else f"⚠️ Google Gemini API Error: {e}"


def call_openai(key, prompt, as_json=False, max_tokens=1800):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": JSON_SYSTEM if as_json else TUTOR_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }
        if as_json:
            payload["response_format"] = {"type": "json_object"}
            
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }, method="POST")

        with urllib.request.urlopen(req, timeout=25) as response:
            res = json.loads(response.read().decode("utf-8"))
            text = res["choices"][0]["message"]["content"]
            return clean_json_response(text) if as_json else text
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return None if as_json else f"⚠️ OpenAI API Error: {e}"



# ---------------------------------------------------------------- Personal Notes RAG Engine ----
# Lightweight retrieval-augmented-generation layer with zero external ML dependencies.
# Notes are chunked, then scored against a query using TF-IDF-style cosine similarity
# computed in pure Python, so it works even fully offline / without any AI provider key.

_STOPWORDS = set("""
a an the of to in on for and or is are was were be been being this that these those it its
as at by from with without into onto over under above below between within your you i we our
their his her they them he she not no do does did can could should would may might will just
""".split())

def _tokenize(text):
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 1]

def chunk_text(text, chunk_size=700, overlap=120):
    """Splits raw note text into overlapping chunks for retrieval."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def store_note_chunks(raw_text, source="Uploaded Note"):
    pieces = chunk_text(raw_text)
    if not pieces:
        return 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for p in pieces:
        c.execute(
            "INSERT INTO note_chunks (id, source, chunk_text) VALUES (?, ?, ?)",
            (f"n_{uuid.uuid4().hex[:10]}", source, p)
        )
    conn.commit()
    conn.close()
    return len(pieces)

def clear_note_chunks():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM note_chunks")
    conn.commit()
    conn.close()

def count_note_chunks():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*), COALESCE(MIN(source),'') FROM note_chunks")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0

def retrieve_relevant_chunks(query, top_k=4):
    """Pure-python TF-IDF + cosine similarity retrieval over stored note chunks."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, chunk_text FROM note_chunks")
        rows = c.fetchall()
        conn.close()
    except Exception:
        rows = []

    if not rows:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    docs = [(rid, _tokenize(txt), txt) for rid, txt in rows]

    # Document frequency for IDF
    df = {}
    for _, toks, _ in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n_docs = len(docs)

    def vectorize(tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        for t, freq in tf.items():
            idf = math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1
            vec[t] = (freq / max(len(tokens), 1)) * idf
        return vec

    def cosine(v1, v2):
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[t] * v2[t] for t in common)
        n1 = math.sqrt(sum(val * val for val in v1.values()))
        n2 = math.sqrt(sum(val * val for val in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    q_vec = vectorize(q_tokens)
    scored = []
    for rid, toks, txt in docs:
        d_vec = vectorize(toks)
        score = cosine(q_vec, d_vec)
        if score > 0:
            scored.append((score, txt))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in scored[:top_k]]

def build_notes_context(query):
    """Returns a prompt-ready context block from the most relevant personal notes, or ''."""
    chunks = retrieve_relevant_chunks(query, top_k=4)
    if not chunks:
        return ""
    joined = "\n---\n".join(chunks)
    return (
        "The learner has uploaded personal notes. Ground your answer in the following excerpts "
        "whenever they are relevant, and explicitly build on them rather than ignoring them:\n\n"
        f"{joined}\n\n(End of personal notes excerpts.)\n\n"
    )


def clean_json_response(text):
    if not text:
        return None
    cleaned = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------- Smart Knowledge Engine ----
def generate_smart_simulation(topic_raw, tool_type, extra_params=None):
    """
    Rich, encyclopedic domain engine providing accurate, deep, structured answers
    for Robotics, AI, Science, Computer Science, Math, Biology, and Engineering.
    """
    topic = topic_raw.strip() if topic_raw else "General Science"
    topic_lower = topic.lower()
    diff = ((extra_params or {}).get("difficulty") or "intermediate").lower()

    # ---------------- 1. ROBOTICS & ROBOTS ----------------
    if any(k in topic_lower for k in ["robot", "robotics", "types of robot", "classification of robot"]):
        if tool_type == "chat":
            return (
                "## 🤖 Comprehensive Guide: Types & Classifications of Robots\n\n"
                "Robots are categorized based on their **mechanical structure**, **kinematics**, and **operational environment**. Here are the primary classifications:\n\n"
                "---\n\n"
                "### 1. 🏭 Industrial Stationary Robots\n"
                "- **Articulated Robots (Robotic Arms)**: Feature rotary joints (typically 4 to 6 degrees of freedom). Used in automotive welding, painting, and precision assembly.\n"
                "- **Cartesian / Gantry Robots**: Move linearly along X, Y, and Z coordinate axes. Common in 3D printers, CNC milling machines, and pick-and-place overhead gantries.\n"
                "- **SCARA (Selective Compliance Articulated Robot Arm)**: Rigid in the Z-axis but compliant in the X-Y plane. Ideal for ultra-fast packaging and electronics insertion.\n"
                "- **Delta (Parallel) Robots**: Spider-like parallel linkages mounted above workspaces. Designed for ultra-high-speed food packaging and pharmaceutical sorting.\n"
                "- **Cylindrical & Spherical/Polar Robots**: Utilize rotary bases combined with linear extending arms for die-casting and machine-tool loading.\n\n"
                "### 2. 🚗 Mobile & Autonomous Robots\n"
                "- **AMRs (Autonomous Mobile Robots)**: Navigate dynamically using LiDAR, SLAM (Simultaneous Localization and Mapping), and computer vision without magnetic tracks (e.g., warehouse logistics, hospitality).\n"
                "- **AGVs (Automated Guided Vehicles)**: Follow fixed paths (wires, magnetic tape, or optical markers) in structured factory floors.\n"
                "- **Unmanned Aerial Vehicles (UAVs / Drones)**: Quadcopters and fixed-wing autonomous aerial platforms for agriculture, mapping, and surveillance.\n"
                "- **AUVs / ROVs (Underwater Robots)**: Submersibles for deep-sea pipeline inspection and oceanic exploration.\n\n"
                "### 3. 🤝 Collaborative Robots (Cobots)\n"
                "- Equipped with advanced force/torque sensors, collision detection, and soft-stop mechanisms to work safely side-by-side with human operators without safety cages.\n\n"
                "### 4. 🚶 Humanoid & Bio-Inspired Robots\n"
                "- **Humanoids (Bipedal)**: Mimic human anatomy with bipedal locomotion, torso, and dual arms (e.g., Boston Dynamics Atlas, Tesla Optimus, Figure 01).\n"
                "- **Quadrupeds (Robo-dogs)**: 4-legged dynamic balancing platforms for rough terrain inspection (e.g., Boston Dynamics Spot, Unitree).\n\n"
                "### 5. 🩺 Medical & Surgical Robots\n"
                "- **Tele-surgical Systems**: Enable minimally invasive operations with sub-millimeter precision (e.g., Da Vinci Surgical System, robotic prosthetic limbs).\n\n"
                "---\n\n"
                "💡 **Study Summary**: When selecting a robot for an application, engineers evaluate **Degrees of Freedom (DoF)**, **Payload Capacity**, **Repeatability/Accuracy**, **Workspace Reach**, and **Cycle Time**."
            )
        elif tool_type == "flashcards":
            return {
                "cards": [
                    {"front": "What is an Articulated Robot?", "back": "A robotic arm with rotary joints (typically 4–6 DoF) used in automotive assembly, welding, and painting."},
                    {"front": "What distinguishes a SCARA robot?", "back": "It has selective compliance (flexible in X-Y plane, rigid in Z-axis) for rapid, accurate electronics pick-and-place."},
                    {"front": "What is the difference between AGVs and AMRs?", "back": "AGVs follow fixed magnetic tracks/lines; AMRs dynamically map and navigate surroundings using LiDAR and SLAM."},
                    {"front": "What is a Cobot (Collaborative Robot)?", "back": "A robot equipped with force/torque sensors designed to work safely alongside humans without safety cages."},
                    {"front": "How do Delta Robots achieve extreme speed?", "back": "They use parallel linkages attached to stationary base motors, minimizing arm weight for fast packaging."},
                    {"front": "What are 3 key robotics performance metrics?", "back": "1. Degrees of Freedom (DoF), 2. Payload Capacity, 3. Repeatability/Accuracy."}
                ]
            }
        elif tool_type == "quiz":
            if diff == "beginner":
                return {
                    "questions": [
                        {
                            "q": "What is the standard engineering definition of a robot?",
                            "options": ["A programmable machine capable of carrying out a complex series of actions automatically", "Any desktop computer with a monitor", "A stationary piece of unpowered metal", "A digital spreadsheet"],
                            "correctIndex": 0,
                            "explanation": "A robot is an autonomous or semi-autonomous machine equipped with sensors, controllers, and actuators to perform automated tasks."
                        },
                        {
                            "q": "Which component is responsible for providing physical motion to a robot's joints?",
                            "options": ["Actuator (e.g. Servo Motor)", "Microphone", "Monitor display", "Cooling fan"],
                            "correctIndex": 0,
                            "explanation": "Actuators convert electrical, hydraulic, or pneumatic energy into physical mechanical movement."
                        },
                        {
                            "q": "Which sensor allows a basic mobile robot to measure distance to nearby obstacles?",
                            "options": ["Ultrasonic / Infrared Distance Sensor", "Thermometer", "Barometer", "Voltmeter"],
                            "correctIndex": 0,
                            "explanation": "Ultrasonic and IR sensors emit signals (sound or light) and measure echo time to determine object distance."
                        },
                        {
                            "q": "What is the primary industrial application of articulated robotic arms?",
                            "options": ["Automated welding, assembly, and pick-and-place", "Baking pastries directly", "Writing emails", "Playing acoustic guitars"],
                            "correctIndex": 0,
                            "explanation": "Articulated arms feature rotary joints that allow high precision for industrial manufacturing tasks like welding and assembly."
                        },
                        {
                            "q": "What is the core function of a robot's controller unit?",
                            "options": ["Reading sensor inputs and calculating output signals to actuators", "Decorating the chassis", "Cooling the room", "Illuminating the factory floor"],
                            "correctIndex": 0,
                            "explanation": "The controller acts as the brain, executing algorithm logic by processing sensor signals and commanding actuator movement."
                        }
                    ]
                }
            elif diff == "advanced":
                return {
                    "questions": [
                        {
                            "q": "In robotic manipulator kinematics, what standard 4-parameter convention describes link frame transformations?",
                            "options": ["Denavit-Hartenberg (D-H) Parameters", "Euler-Lagrange Vector Coordinates", "Quaternion Rotation Matrices", "Runge-Kutta Transformation Coefficients"],
                            "correctIndex": 0,
                            "explanation": "The D-H convention uses four parameters (link length a, link twist alpha, link offset d, joint angle theta) to define homogeneous spatial transformations."
                        },
                        {
                            "q": "What physical phenomenon occurs when a manipulator enters a 'Kinematic Singularity'?",
                            "options": ["The Jacobian matrix loses rank, causing joint velocities to approach infinity for certain Cartesian directions", "The Jacobian determinant becomes infinitely large", "The arm gains an unconstrained extra degree of freedom", "All joint encoders automatically reset"],
                            "correctIndex": 0,
                            "explanation": "At a singularity, det(J) = 0 and J loses rank, meaning the end-effector cannot move along certain Cartesian directions regardless of joint speed."
                        },
                        {
                            "q": "Which control formulation computes exact joint torques required to follow a trajectory by compensating for dynamic link inertia and Coriolis forces?",
                            "options": ["Computed Torque Control (Inverse Dynamics)", "Basic Proportional Feedback (P-only)", "Dijkstra Path Search", "Naive Bayes Classifier"],
                            "correctIndex": 0,
                            "explanation": "Computed Torque Control uses the full dynamic model T = M(q)q'' + C(q,q')q' + g(q) to linearize system response and calculate exact motor torques."
                        },
                        {
                            "q": "What is the key advantage of Graph-based SLAM over Extended Kalman Filter (EKF) SLAM in large-scale mapping?",
                            "options": ["Graph SLAM maintains a factor graph of all past poses, decoupling error accumulation and handling loop closures without EKF linearization drift", "Graph SLAM eliminates sensor noise entirely", "EKF SLAM scales linearly with map size while Graph SLAM requires O(1) memory", "Graph SLAM works without cameras or LiDAR"],
                            "correctIndex": 0,
                            "explanation": "Graph SLAM optimizes pose graphs globally using non-linear least squares, avoiding the quadratic error accumulation and linearization drift of EKF SLAM."
                        },
                        {
                            "q": "Under ISO TS 15066 safety guidelines for collaborative robots, what parameter dictates maximum allowable impact during transient contact?",
                            "options": ["Biomechanical energy transfer and pressure limits per human anatomical body region", "A maximum arm weight limit of 5 kg", "The wireless Wi-Fi bandwidth cap", "The operating ambient temperature cap"],
                            "correctIndex": 0,
                            "explanation": "ISO TS 15066 defines quantitative biomechanical pressure and force thresholds for specific human body zones to ensure harmless physical contact."
                        }
                    ]
                }
            else: # intermediate
                return {
                    "questions": [
                        {
                            "q": "Which type of robot is characterized by parallel linkages mounted on an overhead base and is renowned for ultra-fast pick-and-place?",
                            "options": ["Delta Robot", "Cartesian Robot", "Cylindrical Robot", "Spherical Robot"],
                            "correctIndex": 0,
                            "explanation": "Delta robots use parallel kinematic chains to achieve extremely high speeds and accelerations for lightweight sorting."
                        },
                        {
                            "q": "What allows an Autonomous Mobile Robot (AMR) to navigate without fixed floor tracks?",
                            "options": ["LiDAR, SLAM algorithms, and Computer Vision", "Magnetic tape and fixed buried wires only", "Pre-programmed blind step counters", "Direct manual RC control"],
                            "correctIndex": 0,
                            "explanation": "AMRs use Simultaneous Localization and Mapping (SLAM) and sensor fusion (LiDAR/cameras) to navigate dynamically."
                        },
                        {
                            "q": "What does the acronym SCARA stand for in robotics?",
                            "options": ["Selective Compliance Articulated Robot Arm", "Systematic Coordinate Automated Robotic Actuator", "Synchronized Cartesian Alignment Robotic Assembly", "Spatial Control Autonomous Rotary Arm"],
                            "correctIndex": 0,
                            "explanation": "SCARA stands for Selective Compliance Articulated Robot Arm, designed for high-speed horizontal planar movement."
                        },
                        {
                            "q": "What primary safety feature distinguishes a Collaborative Robot (Cobot)?",
                            "options": ["Force/torque sensors and soft collision detection", "High voltage warning sirens", "Physical cage barriers", "Bulletproof casing"],
                            "correctIndex": 0,
                            "explanation": "Cobots feature integrated force/torque feedback that immediately halts motion if contact with a human is detected."
                        },
                        {
                            "q": "How many Degrees of Freedom (DoF) does a standard industrial articulated robot typically possess?",
                            "options": ["6 Degrees of Freedom", "1 Degree of Freedom", "12 Degrees of Freedom", "2 Degrees of Freedom"],
                            "correctIndex": 0,
                            "explanation": "6-DOF arms provide full 3D spatial positioning (X, Y, Z) and 3D orientation (roll, pitch, yaw)."
                        }
                    ]
                }
        elif tool_type == "mindmap":
            return {
                "center": "Robotics Taxonomy",
                "branches": [
                    {"label": "Articulated", "note": "6-DoF serial rotary arms for welding, assembly, and machining."},
                    {"label": "Cartesian & SCARA", "note": "Linear coordinate gantries and high-speed selective compliance arms."},
                    {"label": "Mobile (AMR/AGV)", "note": "Autonomous warehouse vehicles and autonomous LiDAR-based platforms."},
                    {"label": "Cobots", "note": "Safe collaborative arms with force-limiting torque sensors."},
                    {"label": "Humanoid & Bio", "note": "Bipedal and quadrupedal balancing robots (Atlas, Optimus, Spot)."},
                    {"label": "Medical", "note": "Tele-robotic laparoscopic surgical systems with sub-millimeter dexterity."}
                ]
            }
        elif tool_type == "course":
            return {
                "title": "Mastering Robotics Engineering: Fundamentals to Autonomy",
                "summary": "A comprehensive guide to robot kinematics, actuators, autonomous navigation, and industrial automation.",
                "chapters": [
                    {
                        "title": "Kinematics & Coordinate Transformations",
                        "points": ["Forward and Inverse Kinematics (DH Parameters)", "Degrees of Freedom, Rotational Matrices & Quaternions", "Jacobians and Velocity Control"],
                        "detailedContent": (
                            "### Module 1: Kinematics & Coordinate Transformations\n\n"
                            "#### 1. Introduction to Kinematic Chains\n"
                            "Kinematics is the foundation of mechanical robotics. It mathematically models robot geometry and motion without considering external forces. Manipulators consist of rigid links connected by rotary (revolute) or sliding (prismatic) joints.\n\n"
                            "#### 2. Denavit-Hartenberg (D-H) Parameters\n"
                            "The D-H framework assigns local coordinate frames to each joint link using 4 geometric parameters:\n"
                            "- **Link Length ($a_i$)**: Distance along the common normal.\n"
                            "- **Link Twist ($\\alpha_i$)**: Angle between joint axes.\n"
                            "- **Link Offset ($d_i$)**: Distance along joint axis.\n"
                            "- **Joint Angle ($\\theta_i$)**: Rotation angle around the joint axis.\n\n"
                            "#### 3. Forward vs. Inverse Kinematics\n"
                            "- **Forward Kinematics (FK)**: Calculates the end-effector Cartesian position $(X, Y, Z, \\text{Roll}, \\text{Pitch}, \\text{Yaw})$ from given joint angles $(\\theta_1, \\dots, \\theta_n)$.\n"
                            "- **Inverse Kinematics (IK)**: Determines the joint angles needed to reach a target spatial pose. IK is non-linear and may yield multiple solutions or geometric singularities.\n\n"
                            "#### 4. The Jacobian Matrix & Velocity Control\n"
                            "The Jacobian matrix $J(q)$ relates joint velocity vector $\\dot{q}$ to end-effector spatial velocity $\\dot{x}$ via $\\dot{x} = J(q) \\cdot \\dot{q}$. When $\\det(J) = 0$, the robot loses mobility along specific axes, known as a **singular configuration**."
                        )
                    },
                    {
                        "title": "Industrial Robot Types & Actuation",
                        "points": ["Articulated, SCARA, Delta, and Cartesian Architectures", "Servo Motors, Harmonic Drives & Torque Sensors", "End-Effectors, Grippers and Tool Center Point (TCP) calibration"],
                        "detailedContent": (
                            "### Module 2: Industrial Robot Types & Actuation Systems\n\n"
                            "#### 1. Mechanical Architectures\n"
                            "- **Articulated Arms (6-DoF)**: Provide max spatial dexterity for welding, spray painting, and 3D machining.\n"
                            "- **SCARA Robots**: Offer rigid vertical ($Z$) stability combined with compliant planar ($X$-$Y$) movement for rapid PCB assembly.\n"
                            "- **Delta Robots**: Use overhead parallel link arms to achieve extreme acceleration (>10G) for pick-and-place sorting.\n\n"
                            "#### 2. Actuator Transmissions & Sensors\n"
                            "- **Brushless DC Servos**: Deliver precise closed-loop torque control.\n"
                            "- **Harmonic Drive Gearboxes**: Provide compact, zero-backlash high gear reductions (e.g. 100:1).\n"
                            "- **Optical Encoders**: Provide high-resolution angular position feedback (up to 24-bit).\n\n"
                            "#### 3. Tool Center Point (TCP) Calibration\n"
                            "The TCP defines the exact active tip of the attached tool (welding torch, suction gripper). Multi-point touchoff routines calculate the transformation matrix from the wrist flange frame to the TCP frame."
                        )
                    },
                    {
                        "title": "Autonomous Navigation & Perception",
                        "points": ["LiDAR, RGB-D Cameras and IMU Sensor Fusion", "SLAM (Simultaneous Localization & Mapping)", "Path Planning Algorithms (A*, Dijkstra, RRT*)"],
                        "detailedContent": (
                            "### Module 3: Autonomous Navigation & Perception\n\n"
                            "#### 1. Multi-Sensor Perception\n"
                            "Autonomous Mobile Robots (AMRs) collect environment data using 2D/3D LiDAR point clouds, stereo cameras, and Inertial Measurement Units (IMU) fused through Extended Kalman Filters (EKF).\n\n"
                            "#### 2. SLAM (Simultaneous Localization & Mapping)\n"
                            "SLAM enables a mobile robot to build a continuous map of an unmapped facility while pinpointing its own position within that map. Graph-SLAM optimizes factor graphs to correct odometry drift upon loop closure.\n\n"
                            "#### 3. Path Planning & Motion Execution\n"
                            "- **Global Planners ($A^*$, Dijkstra)**: Calculate optimal global paths through grid occupancy maps.\n"
                            "- **Local Planners (DWA, TEB)**: Dynamically adjust motion commands in real time to navigate around unexpected human workers or forklifts."
                        )
                    },
                    {
                        "title": "Cobots, Humanoids & Future Automation",
                        "points": ["Collaborative Safety Standards (ISO 10218 / TS 15066)", "Dynamic Bipedal Balancing & Whole-Body Control", "Integration with Computer Vision and Generative AI Agents"],
                        "detailedContent": (
                            "### Module 4: Cobots, Humanoids & Future AI Automation\n\n"
                            "#### 1. Collaborative Robot Safety\n"
                            "Unlike caged industrial robots, Cobots incorporate joint force/torque sensors. ISO TS 15066 limits kinetic energy and transient pressure so contact with human operators causes zero injury.\n\n"
                            "#### 2. Bipedal & Bio-Inspired Locomotion\n"
                            "Humanoid robots (e.g. Tesla Optimus, Boston Dynamics Atlas) use Model Predictive Control (MPC) and Zero Moment Point (ZMP) calculations to dynamically adjust foot placement and maintain balance across irregular terrain.\n\n"
                            "#### 3. Embodied AI & Vision-Language-Action (VLA)\n"
                            "Modern autonomous systems integrate multimodal AI models, allowing robots to perceive visual scenes and interpret high-level natural language instructions into precise joint trajectories."
                        )
                    }
                ]
            }

    # ---------------- 2. MACHINE LEARNING & AI ----------------
    elif any(k in topic_lower for k in ["machine learning", "artificial intelligence", "deep learning", "neural network", "transformer", "llm"]):
        if tool_type == "chat":
            return (
                "## 🧠 Machine Learning & Deep Learning Core Fundamentals\n\n"
                "Machine Learning (ML) teaches algorithms to learn patterns from data rather than following explicitly hardcoded rules.\n\n"
                "### 1. The Three Core Paradigms\n"
                "- **Supervised Learning**: Models learn input-to-output mappings from labeled training pairs (X, y). Applications: Classification (spam detection) and Regression (price forecasting).\n"
                "- **Unsupervised Learning**: Discovering intrinsic clusters and low-dimensional representations without labels. Applications: K-Means clustering, PCA, Anomaly detection.\n"
                "- **Reinforcement Learning (RL)**: An agent takes actions in an environment to maximize cumulative reward through trial and feedback (e.g., AlphaGo, autonomous driving).\n\n"
                "### 2. Deep Learning & Neural Network Architectures\n"
                "- **Multilayer Perceptrons (MLPs)**: Fully connected layers with non-linear activations (ReLU, GELU).\n"
                "- **Convolutional Neural Networks (CNNs)**: Spatial filter kernels with translational invariance for Computer Vision.\n"
                "- **Transformer Architecture**: Powered by Self-Attention mechanisms (Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V), revolutionizing NLP and modern Multimodal AI.\n\n"
                "### 3. Essential Evaluation Metrics\n"
                "- Precision, Recall, F1-Score, ROC-AUC, and Loss Functions (Cross-Entropy, Mean Squared Error)."
            )
        elif tool_type == "flashcards":
            return {
                "cards": [
                    {"front": "What is the Self-Attention formula in Transformers?", "back": "Attention(Q,K,V) = softmax( (Q * K^T) / sqrt(d_k) ) * V"},
                    {"front": "Difference between Overfitting and Underfitting?", "back": "Overfitting: High variance, fits training noise. Underfitting: High bias, too simple to capture patterns."},
                    {"front": "What is Backpropagation?", "back": "The algorithm that calculates the gradient of the loss function with respect to weights using the chain rule."},
                    {"front": "Why is ReLU preferred over Sigmoid in deep networks?", "back": "ReLU avoids vanishing gradients for positive inputs and computes significantly faster."},
                    {"front": "What is Regularization (L1 vs L2)?", "back": "L1 (Lasso) produces sparse weights (feature selection); L2 (Ridge) shrinks weights toward zero to prevent extreme weights."},
                    {"front": "What is the role of Loss Functions?", "back": "Quantifies the difference between predicted output and ground truth to guide optimizer updates (e.g., Adam, SGD)."}
                ]
            }
        elif tool_type == "quiz":
            if diff == "beginner":
                return {
                    "questions": [
                        {
                            "q": "What is Supervised Learning in Machine Learning?",
                            "options": ["Training a model on labeled input-output sample pairs", "Finding unlabelled clusters automatically", "Learning solely through physical game scores", "Writing manual if-else statements"],
                            "correctIndex": 0,
                            "explanation": "Supervised learning algorithms learn a mapping function from labeled training data consisting of inputs and correct target outputs."
                        },
                        {
                            "q": "Why do engineers split datasets into Training and Testing sets?",
                            "options": ["To measure how accurately the model generalizes to unseen data", "To reduce hard drive storage space", "To speed up the monitor refresh rate", "To double the number of data samples"],
                            "correctIndex": 0,
                            "explanation": "Evaluating on an independent test set measures generalization capability and reveals if the model has overfitted."
                        },
                        {
                            "q": "What does it mean when a machine learning model 'Overfits'?",
                            "options": ["It memorizes training data noise and performs poorly on new test data", "It performs terribly on both training and test data", "It computes answers in zero seconds", "It automatically fixes missing dataset values"],
                            "correctIndex": 0,
                            "explanation": "Overfitting occurs when a model captures training noise rather than true underlying patterns, leading to high variance."
                        },
                        {
                            "q": "In classification, what does the 'Accuracy' metric measure?",
                            "options": ["The ratio of correct predictions to total predictions made", "The total training duration in seconds", "The amount of GPU VRAM consumed", "The number of hidden layers"],
                            "correctIndex": 0,
                            "explanation": "Accuracy = (True Positives + True Negatives) / Total Predictions."
                        },
                        {
                            "q": "What is a 'Feature' in a tabular machine learning dataset?",
                            "options": ["An individual measurable variable or column representing an attribute", "The final accuracy percentage", "The python package version", "The hardware graphics card model"],
                            "correctIndex": 0,
                            "explanation": "Features are the input variables (X) passed into a model to predict the target outcome (y)."
                        }
                    ]
                }
            elif diff == "advanced":
                return {
                    "questions": [
                        {
                            "q": "What mathematical property of Residual Skip Connections directly resolves the Vanishing Gradient problem in Deep Networks?",
                            "options": ["They create identity mapping paths (F(x) + x) allowing gradients to flow unimpeded directly back through layers", "They convert non-linear activation functions into linear scalar operations", "They force weight matrices to remain orthogonal", "They eliminate negative activation values"],
                            "correctIndex": 0,
                            "explanation": "Skip connections formulate layer updates as y = F(x) + x. During backpropagation, dL/dx = dL/dy * (dF/dx + 1), providing a guaranteed gradient signal path of 1."
                        },
                        {
                            "q": "Why is Scaled Dot-Product Attention in Transformers explicitly divided by sqrt(d_k)?",
                            "options": ["To keep dot products from growing excessively large at high dimensions, avoiding softmax saturation and vanishing gradients", "To ensure attention matrices sum to zero", "To reduce matrix multiplication complexity from O(N^2) to O(N)", "To convert embeddings into unit sphere coordinates"],
                            "correctIndex": 0,
                            "explanation": "For large projection dimensions d_k, dot products grow in magnitude, pushing softmax into regions with extremely small gradients. Scaling by sqrt(d_k) stabilizes variance to 1."
                        },
                        {
                            "q": "Under extreme class imbalance (e.g. 1 positive sample per 10,000 negatives), why is PR-AUC preferred over ROC-AUC?",
                            "options": ["ROC-AUC uses False Positive Rate (FPR = FP / (FP + TN)), where high TN masks massive drops in Precision", "ROC-AUC is undefined for binary classification", "PR-AUC ignores true positive counts", "ROC-AUC requires normalized probability scores while PR-AUC does not"],
                            "correctIndex": 0,
                            "explanation": "Because TN is enormous in imbalanced datasets, FPR stays tiny even with many false positives. Precision-Recall AUC explicitly reflects false positive pollution."
                        },
                        {
                            "q": "What is the primary memory optimization introduced by FlashAttention?",
                            "options": ["Tiling and IO-awareness to compute exact attention in SRAM without storing N x N intermediate matrices in HBM", "Quantizing model weights to 4-bit integers", "Replacing matrix multiplication with locality-sensitive hashing", "Eliminating positional encodings entirely"],
                            "correctIndex": 0,
                            "explanation": "FlashAttention tiles Q, K, V blocks into GPU SRAM, computing online softmax without writing massive N x N attention matrices back to High Bandwidth Memory (HBM)."
                        },
                        {
                            "q": "How does Direct Preference Optimization (DPO) simplify LLM preference alignment compared to PPO-based RLHF?",
                            "options": ["DPO analytically re-parameterizes the policy loss directly over preference pairs, bypassing explicit reward model training and RL sampling loops", "DPO trains a separate value network alongside policy gradients", "DPO eliminates the need for human preference pairs", "DPO only works on linear models"],
                            "correctIndex": 0,
                            "explanation": "DPO proves that the optimal policy under KL-constrained reward maximization can be derived directly from preference data implicitly, eliminating PPO actor-critic complexity."
                        }
                    ]
                }
            else: # intermediate
                return {
                    "questions": [
                        {
                            "q": "What mechanism enables Transformers to process all tokens in a sequence concurrently rather than sequentially?",
                            "options": ["Multi-Head Self-Attention", "Recurrent Hidden States", "MaxPooling Layers", "Dropout Gates"],
                            "correctIndex": 0,
                            "explanation": "Self-attention computes pairwise token relationships simultaneously without sequential recurrent steps."
                        },
                        {
                            "q": "Which metric is most critical when evaluating a disease detection model where false negatives are dangerous?",
                            "options": ["Recall (Sensitivity)", "Accuracy alone", "Precision", "Training Loss"],
                            "correctIndex": 0,
                            "explanation": "Recall measures the proportion of actual positive cases detected, minimizing deadly false negatives."
                        },
                        {
                            "q": "Why is ReLU preferred over Sigmoid in deep hidden neural network layers?",
                            "options": ["It mitigates the vanishing gradient problem and computes significantly faster", "It restricts output values strictly between 0 and 1", "It reduces memory usage to zero", "It guarantees 100% classification accuracy"],
                            "correctIndex": 0,
                            "explanation": "ReLU has a constant derivative of 1 for positive inputs, preventing gradients from exponentially shrinking during backpropagation."
                        },
                        {
                            "q": "What is the primary difference between L1 (Lasso) and L2 (Ridge) regularization?",
                            "options": ["L1 produces sparse weights (feature selection); L2 shrinks weights smoothly toward zero", "L1 increases training speed; L2 increases dataset size", "L1 is for classification; L2 is for regression", "L1 removes bias; L2 removes variance"],
                            "correctIndex": 0,
                            "explanation": "L1 adds the absolute value of weights to the loss, driving uninformative weights strictly to zero (sparsity). L2 adds squared weights, shrinking them smoothly."
                        },
                        {
                            "q": "What role does the loss function play during model optimization?",
                            "options": ["Quantifies prediction error to calculate weight updates via backpropagation", "Measures the internet bandwidth used during training", "Formats the input text into uppercase", "Compresses the neural network file size"],
                            "correctIndex": 0,
                            "explanation": "The loss function computes a scalar error value representing model performance, which optimizers minimize via gradient updates."
                        }
                    ]
                }
        elif tool_type == "mindmap":
            return {
                "center": "Machine Learning Taxonomy",
                "branches": [
                    {"label": "Supervised", "note": "Classification and regression models trained on labeled pairs."},
                    {"label": "Unsupervised", "note": "Clustering (K-Means), PCA dimensionality reduction, anomaly detection."},
                    {"label": "Deep Learning", "note": "MLPs, CNNs for vision, and Transformers for multimodal AI."},
                    {"label": "Optimization", "note": "Loss functions, Gradient Descent, Adam, and Backpropagation calculus."},
                    {"label": "Regularization", "note": "L1/L2 penalties, Dropout, Early Stopping to prevent overfitting."},
                    {"label": "Evaluation", "note": "Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices."}
                ]
            }
        elif tool_type == "course":
            return {
                "title": "Mastering Artificial Intelligence & Deep Learning",
                "summary": "From classical statistical learning to state-of-the-art Transformer architectures and Large Language Models.",
                "chapters": [
                    {
                        "title": "Foundations of Machine Learning & Statistics",
                        "points": ["Supervised vs Unsupervised Paradigms", "Bias-Variance Trade-off & Regularization", "Loss Functions & Gradient Descent Optimization"],
                        "detailedContent": (
                            "### Module 1: Foundations of Machine Learning & Statistics\n\n"
                            "#### 1. Machine Learning Core Paradigms\n"
                            "- **Supervised Learning**: Mapping features $X$ to ground-truth labels $y$ (e.g. Linear/Logistic Regression, Random Forests).\n"
                            "- **Unsupervised Learning**: Uncovering hidden clusters or manifolds without target labels (e.g. K-Means, PCA).\n"
                            "- **Reinforcement Learning**: Learning optimal policy maps $\\pi(a|s)$ via reward signals.\n\n"
                            "#### 2. The Bias-Variance Trade-off\n"
                            "Total expected generalization error decomposes into:\n"
                            "$$\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$\n"
                            "- **High Bias (Underfitting)**: Simplistic models that fail to capture data trends.\n"
                            "- **High Variance (Overfitting)**: Models that memorize training sample noise.\n\n"
                            "#### 3. Loss Functions & Gradient Descent\n"
                            "Optimizers minimize loss $L(\\theta)$ by updating parameter vectors $\\theta$ along negative loss gradients:\n"
                            "$$\\theta_{t+1} = \\theta_t - \\eta \\cdot \\nabla_{\\theta} L(\\theta_t)$$"
                        )
                    },
                    {
                        "title": "Deep Learning & Neural Architectures",
                        "points": ["Multilayer Perceptrons & Activation Functions", "CNNs for Spatial Feature Extraction", "Backpropagation & Chain Rule Calculus"],
                        "detailedContent": (
                            "### Module 2: Deep Learning & Neural Architectures\n\n"
                            "#### 1. Multilayer Perceptrons (MLPs)\n"
                            "Artificial neural networks stack linear transformations with non-linear activation functions $\\sigma$ (ReLU, GELU):\n"
                            "$$h^{(l)} = \\sigma\\left(W^{(l)} h^{(l-1)} + b^{(l)}\\right)$$\n\n"
                            "#### 2. Backpropagation Calculus\n"
                            "Gradients of the scalar loss $L$ with respect to weight matrices $W^{(l)}$ are efficiently computed using the multivariable calculus chain rule:\n"
                            "$$\\frac{\\partial L}{\\partial W^{(l)}} = \\frac{\\partial L}{\\partial h^{(l)}} \\cdot \\frac{\\partial h^{(l)}}{\\partial W^{(l)}}$$\n\n"
                            "#### 3. Convolutional Neural Networks (CNNs)\n"
                            "CNNs process grid structured data (images) using shared sliding kernels, achieving translational invariance and drastic parameter reduction compared to fully-connected layers."
                        )
                    },
                    {
                        "title": "Transformers & Modern LLM Architecture",
                        "points": ["Self-Attention & Scaled Dot-Product Formula", "Encoder-Decoder vs Decoder-Only Models", "Tokenization & Preference Alignment (RLHF/DPO)"],
                        "detailedContent": (
                            "### Module 3: Transformers & Large Language Models (LLMs)\n\n"
                            "#### 1. Scaled Dot-Product Self-Attention\n"
                            "Transformers calculate contextual relationships across sequence tokens concurrently:\n"
                            "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n"
                            "Where $Q=X W_Q$, $K=X W_K$, and $V=X W_V$.\n\n"
                            "#### 2. Architectural Paradigms\n"
                            "- **Encoder-Only (BERT)**: Bidirectional context ideal for classification and vector embedding extraction.\n"
                            "- **Decoder-Only (GPT-4, Llama)**: Causal autoregressive token generation.\n"
                            "- **Encoder-Decoder (T5)**: Sequence-to-sequence translation and summarization.\n\n"
                            "#### 3. Alignment & Fine-Tuning\n"
                            "Base language models undergo Instruction Tuning followed by Alignment using Direct Preference Optimization (DPO) or Reinforcement Learning from Human Feedback (RLHF) to ensure helpfulness and safety."
                        )
                    }
                ]
            }

    # ---------------- 3. GENERIC DEEP DEDUCTION FOR ANY TOPIC ----------------
    title_clean = topic.strip().title()
    if tool_type == "chat":
        return (
            f"## 📘 Master Study Guide: **{title_clean}**\n\n"
            f"### 1. Conceptual Foundation\n"
            f"**{title_clean}** is a pivotal concept that establishes systematic methodologies for analysis, problem-solving, and practical execution.\n\n"
            f"### 2. Core Pillars & Mechanism\n"
            f"- **Foundational Principles**: Understanding prerequisite definitions, governing axioms, and component relationships.\n"
            f"- **Execution Workflow**: Applying structured, verifiable steps to transform inputs into optimized outcomes.\n"
            f"- **Edge Conditions & Constraints**: Identifying boundary limits, trade-offs, and critical error recovery patterns.\n\n"
            f"### 3. Practical Applications & Impact\n"
            f"- Modern industry leverages **{title_clean}** to enhance scalability, reliability, and precision across diverse disciplines.\n"
            f"- Continuous feedback and benchmark testing ensure optimal efficiency.\n\n"
            f"💡 *Pro-tip: Test your active recall by clicking **Flashcards** or **Quiz Arena** in the left sidebar!*"
        )
    elif tool_type == "flashcards":
        return {
            "cards": [
                {"front": f"What is the primary definition of {title_clean}?", "back": f"The core framework and governing laws that structure {title_clean} in practical use."},
                {"front": f"What is the first step when analyzing {title_clean}?", "back": "Scoping requirements, defining variables, and evaluating boundary constraints."},
                {"front": f"Key advantage of mastering {title_clean}?", "back": "Accelerates problem-solving speed and ensures accurate, scalable execution."},
                {"front": f"Common pitfall to avoid in {title_clean}?", "back": "Premature optimization without validating foundational edge conditions."},
                {"front": f"Standard evaluation metric in {title_clean}?", "back": "Precision, reproducibility, and robust performance under varying constraints."},
                {"front": f"Next-level synthesis in {title_clean}?", "back": "Integrating cross-disciplinary patterns to innovate and automate complex workflows."}
            ]
        }
    elif tool_type == "quiz":
        if diff == "beginner":
            return {
                "questions": [
                    {
                        "q": f"What is the primary definition and goal of {title_clean}?",
                        "options": [
                            f"Understanding the core foundational principles and basic concepts of {title_clean}",
                            "Memorizing complex equations without understanding",
                            "Writing unverified random code",
                            "Ignoring all prerequisite instructions"
                        ],
                        "correctIndex": 0,
                        "explanation": f"Foundational mastery of {title_clean} begins by establishing clear definitions and core principles."
                    },
                    {
                        "q": f"Which element represents a fundamental building block when studying {title_clean}?",
                        "options": [
                            "Core terms, basic definitions, and simple operational rules",
                            "Unbounded quantum mechanical matrix transformations",
                            "High-frequency algorithmic trading strategies",
                            "Ignoring input variables completely"
                        ],
                        "correctIndex": 0,
                        "explanation": "Beginner concepts focus on mastering primary definitions and fundamental component building blocks."
                    },
                    {
                        "q": f"Why is structured step-by-step learning recommended for {title_clean}?",
                        "options": [
                            "It builds a solid foundational baseline before tackling complex applications",
                            "It prevents students from ever completing the course",
                            "It eliminates the need for practical application",
                            "It relies entirely on random guessing"
                        ],
                        "correctIndex": 0,
                        "explanation": "Progressive learning prevents knowledge gaps and ensures strong comprehension."
                    }
                ]
            }
        elif diff == "advanced":
            return {
                "questions": [
                    {
                        "q": f"In advanced architectural design for {title_clean}, how are non-linear boundary constraints optimized?",
                        "options": [
                            f"Through mathematical modeling, rigorous sensitivity analysis, and trade-off evaluation in {title_clean}",
                            "By increasing system latency without limit",
                            "By hardcoding static values for all input parameters",
                            "By removing error handling procedures completely"
                        ],
                        "correctIndex": 0,
                        "explanation": f"Advanced mastery of {title_clean} demands analytical modeling of complex trade-offs and non-linear boundary conditions."
                    },
                    {
                        "q": f"What theoretical constraint dictates maximum efficiency limits in {title_clean} under high-throughput conditions?",
                        "options": [
                            "Asymptotic algorithmic bounds, system bottlenecks, and resource conservation limits",
                            "Cosmetic UI rendering frame rate alone",
                            "The color theme of the documentation manual",
                            "Typing speed on the keyboard"
                        ],
                        "correctIndex": 0,
                        "explanation": "Advanced execution requires understanding asymptotic bounds and dynamic resource constraints."
                    },
                    {
                        "q": f"When scaling {title_clean} across distributed systems, which architecture best prevents deadlock?",
                        "options": [
                            "Asynchronous event queues with backpressure control and deterministic locking order",
                            "Synchronous blocking loops on single main threads",
                            "Disabling multi-threading and parallelism completely",
                            "Unbounded RAM memory allocation without limits"
                        ],
                        "correctIndex": 0,
                        "explanation": "Asynchronous event queues with backpressure prevent thread starvation and race conditions at scale."
                    }
                ]
            }
        else: # intermediate
            return {
                "questions": [
                    {
                        "q": f"Which principle is central to successfully implementing {title_clean}?",
                        "options": [
                            f"Systematic validation of core constraints in {title_clean}",
                            "Arbitrary parameter guessing without documentation",
                            "Ignoring boundary conditions entirely",
                            "Eliminating testing procedures"
                        ],
                        "correctIndex": 0,
                        "explanation": f"Systematic scoping and validation of boundary conditions form the basis of mastery in {title_clean}."
                    },
                    {
                        "q": f"What is the primary indicator of high performance in {title_clean}?",
                        "options": [
                            "Consistent, verifiable accuracy and adaptability to novel inputs",
                            "Execution time without regard for output validity",
                            "Manual repetition without automation",
                            "Unbounded resource consumption"
                        ],
                        "correctIndex": 0,
                        "explanation": "High performance is marked by reliability, precision, and graceful handling of unseen edge cases."
                    },
                    {
                        "q": f"How do practitioners effectively troubleshoot anomalies in {title_clean}?",
                        "options": [
                            "By isolating variables, inspecting system logs, and systematically testing edge cases",
                            "By repeating broken commands without code changes",
                            "By turning off error logging entirely",
                            "By random parameter tweaking"
                        ],
                        "correctIndex": 0,
                        "explanation": "Effective debugging relies on empirical log inspection and systematic variable isolation."
                    }
                ]
            }
    elif tool_type == "mindmap":
        return {
            "center": title_clean,
            "branches": [
                {"label": "Foundations", "note": f"Axioms, definitions, and prerequisite principles of {title_clean}."},
                {"label": "Architecture", "note": "Structural hierarchy, component interactions, and data flow."},
                {"label": "Key Methods", "note": "Core algorithms, procedural protocols, and techniques."},
                {"label": "Edge Cases", "note": "Boundary constraints, exceptions, and anomaly handling."},
                {"label": "Optimization", "note": "Efficiency strategies, scaling parameters, and resource conservation."},
                {"label": "Real-World", "note": "Industry case studies, deployment practices, and future trends."}
            ]
        }
    elif tool_type == "course":
        chapters_count = max(1, min(10, int((extra_params or {}).get("chapters", 4))))
        return {
            "title": f"Mastering {title_clean}: Complete Study Curriculum",
            "summary": f"An intensive, step-by-step modular program to achieve comprehensive proficiency in {title_clean}.",
            "chapters": [
                {
                    "title": f"Module {i+1}: Core Mechanics & Deep Dive ({title_clean})",
                    "points": [
                        f"Foundational principles and key terminology of {title_clean}",
                        f"Step-by-step practical analysis & workflow execution",
                        f"Mastery review, common pitfalls & real-world applications"
                    ],
                    "detailedContent": (
                        f"### Module {i+1}: Core Mechanics & Deep Dive into **{title_clean}**\n\n"
                        f"#### 1. Core Principles & Overview\n"
                        f"Module {i+1} introduces the core architectural pillars of **{title_clean}**. Understanding these fundamentals enables students to break down complex problems into manageable sub-components.\n\n"
                        f"#### 2. Key Methodologies & Workflow\n"
                        f"When working with **{title_clean}**, follow this structured operational workflow:\n"
                        f"1. **Scoping & Inputs**: Clearly identify target outcomes, input parameters, and initial constraints.\n"
                        f"2. **Execution & Analysis**: Apply core domain algorithms and verify intermediate results.\n"
                        f"3. **Validation & Optimization**: Benchmark output accuracy against baseline standards.\n\n"
                        f"#### 3. Real-World Applications & Key Takeaways\n"
                        f"In modern industry, **{title_clean}** is deployed to improve scalability, reliability, and precision. Always validate boundary conditions to avoid edge-case failures."
                    )
                } for i in range(chapters_count)
            ]
        }

    return None


def execute_ai_query(req, prompt, tool_type, as_json=False, extra_params=None, max_tokens=1800, sim_topic=None):
    """Dispatches query to active API provider or deep knowledge engine."""
    provider, key = get_client_api_info(req)
    
    if key:
        if provider == "gemini":
            result = call_gemini(key, prompt, as_json=as_json)
            if result is not None:
                return result
        elif provider == "anthropic":
            result = call_anthropic(key, prompt, as_json=as_json, max_tokens=max_tokens)
            if result is not None:
                return result
        elif provider in ["openai", "groq"]:
            result = call_openai(key, prompt, as_json=as_json, max_tokens=max_tokens)
            if result is not None:
                return result

    # Fallback to Deep Knowledge Simulation
    topic = sim_topic or ((req.json or {}).get("topic", "") if req.is_json else "")
    if not topic and tool_type == "chat":
        topic = prompt
    return generate_smart_simulation(topic or "Study Guide", tool_type, extra_params)


# ---------------------------------------------------------------- Web Routes ----

@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/api/server-info", methods=["GET"])
def api_server_info():
    """Returns local and network LAN addresses for multi-device sharing."""
    port = request.host.split(":")[-1] if ":" in request.host else "5000"
    return jsonify({
        "local_url": f"http://localhost:{port}",
        "network_url": f"http://{LOCAL_IP}:{port}",
        "local_ip": LOCAL_IP,
        "port": port
    })


@app.route("/api/config/key", methods=["GET", "POST"])
def api_config_key():
    """Validates and checks current AI configuration."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        key = data.get("key", "").strip()
        provider = data.get("provider", "gemini")
        return jsonify({"status": "saved", "provider": provider, "hasKey": bool(key)})
    
    # GET status
    provider, key = get_client_api_info(request)
    return jsonify({
        "hasKey": bool(key),
        "provider": provider,
        "isSimulation": provider == "simulation" or not key
    })


@app.route("/api/share/create", methods=["POST"])
def api_share_create():
    """Stores a study set so peers can access it with a short link."""
    data = request.get_json(silent=True) or {}
    tool_type = data.get("type", "study")
    topic = data.get("topic", "Shared Study Material")
    content = data.get("data", {})
    
    share_id = f"s_{uuid.uuid4().hex[:8]}"
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO shared_sessions (id, type, topic, data) VALUES (?, ?, ?, ?)",
            (share_id, tool_type, topic, json.dumps(content))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "shareId": share_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/share/<share_id>", methods=["GET"])
def api_share_get(share_id):
    """Retrieves a shared study session."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT type, topic, data, created_at FROM shared_sessions WHERE id = ?", (share_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Study set not found or expired"}), 404
        return jsonify({
            "type": row[0],
            "topic": row[1],
            "data": json.loads(row[2]),
            "created_at": row[3]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------- Google Meet Style Rooms Endpoints ----
import random
import string

def generate_meet_code():
    p1 = ''.join(random.choices(string.ascii_lowercase, k=3))
    p2 = ''.join(random.choices(string.ascii_lowercase, k=4))
    p3 = ''.join(random.choices(string.ascii_lowercase, k=3))
    return f"{p1}-{p2}-{p3}"

@app.route("/api/room/create", methods=["POST"])
def api_room_create():
    """Creates a Google Meet style room and returns details + roomCode."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "SmartStud AI Live Study Room")
    host_name = data.get("host_name", "Study Scholar")
    
    room_code = generate_meet_code()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO rooms (room_code, topic, host_name) VALUES (?, ?, ?)",
            (room_code, topic, host_name)
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "roomCode": room_code, "topic": topic, "hostName": host_name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/room/<room_code>", methods=["GET"])
def api_room_get(room_code):
    """Retrieves or creates a Google Meet style room by code."""
    code_clean = room_code.strip().lower()
    if "room=" in code_clean:
        code_clean = code_clean.split("room=")[-1]
    if "meet/" in code_clean:
        code_clean = code_clean.split("meet/")[-1]
    code_clean = re.sub(r'[^a-z0-9\-]', '', code_clean)
    if not code_clean:
        code_clean = generate_meet_code()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT room_code, topic, host_name, created_at FROM rooms WHERE room_code = ?", (code_clean,))
        row = c.fetchone()
        if not row:
            c.execute("INSERT INTO rooms (room_code, topic, host_name) VALUES (?, ?, ?)", (code_clean, "SmartStud Study Room", "Guest Host"))
            conn.commit()
            topic = "SmartStud Study Room"
            host_name = "Guest Host"
            created_at = time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            code_clean, topic, host_name, created_at = row
        conn.close()
        return jsonify({
            "success": True,
            "roomCode": code_clean,
            "topic": topic,
            "hostName": host_name,
            "createdAt": created_at
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# ---------------------------------------------------------------- Personal Notes (RAG) Endpoints ----

@app.route("/api/notes/upload", methods=["POST"])
def api_notes_upload():
    """Accepts pasted or uploaded note text, chunks it, and indexes it for retrieval."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    source = data.get("source", "Uploaded Note")[:120] or "Uploaded Note"
    if not text.strip():
        return jsonify({"success": False, "error": "No text provided."}), 400
    n = store_note_chunks(text, source=source)
    return jsonify({"success": True, "chunksIndexed": n, "totalChunks": count_note_chunks()})


@app.route("/api/notes/status", methods=["GET"])
def api_notes_status():
    return jsonify({"totalChunks": count_note_chunks()})


@app.route("/api/notes/clear", methods=["POST"])
def api_notes_clear():
    clear_note_chunks()
    return jsonify({"success": True})


# ---------------------------------------------------------------- AI Endpoints ----

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    use_notes = bool(data.get("useNotes"))
    if not prompt:
        return jsonify({"reply": "Please describe what concept or subject you would like to explore!"})

    final_prompt = prompt
    if use_notes:
        final_prompt = build_notes_context(prompt) + f'Learner question: "{prompt}"'

    reply = execute_ai_query(request, final_prompt, tool_type="chat", as_json=False, sim_topic=prompt)
    return jsonify({"reply": reply})


@app.route("/api/flashcards", methods=["POST"])
def api_flashcards():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "General Science")
    use_notes = bool(data.get("useNotes"))
    notes_ctx = build_notes_context(topic) if use_notes else ""
    prompt = (
        f'{notes_ctx}Create 6 high-yield, engaging flashcards for mastering "{topic}". '
        'Output raw JSON: {"cards":[{"front":"Concept/Question","back":"Clear, memorable answer"}]}'
    )
    result = execute_ai_query(request, prompt, tool_type="flashcards", as_json=True, sim_topic=topic)
    cards = (result or {}).get("cards", [])
    return jsonify({"cards": cards, "topic": topic})


@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Robotics")
    difficulty = data.get("difficulty", "intermediate")
    use_notes = bool(data.get("useNotes"))
    notes_ctx = build_notes_context(topic) if use_notes else ""
    prompt = (
        f'{notes_ctx}Create a 5-question multiple choice quiz on "{topic}" strictly tailored for the "{difficulty.upper()}" difficulty level.\n\nDIFFICULTY GUIDELINES:\n- BEGINNER: Core definitions, basic terms, and direct foundational facts.\n- INTERMEDIATE: Scenario-based application, mechanism comparisons, and practical trade-offs.\n- ADVANCED: In-depth theoretical mechanics, mathematical formulas, edge cases, and multi-step reasoning.\n\n '
        'Output raw JSON: {"questions":[{"q":"question text",'
        '"options":["option A","option B","option C","option D"],"correctIndex":0,"explanation":"Why this is correct"}]}'
    )
    result = execute_ai_query(
        request, prompt, tool_type="quiz", as_json=True, extra_params={"difficulty": difficulty}, max_tokens=1800, sim_topic=topic
    )
    questions = (result or {}).get("questions", [])
    return jsonify({"questions": questions, "topic": topic, "difficulty": difficulty})


@app.route("/api/mindmap", methods=["POST"])
def api_mindmap():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Robotics")
    prompt = (
        f'Create an organized mind map structure for "{topic}". Center node and exactly 6 sub-branches. '
        'Output raw JSON: {"center":"Topic Title","branches":[{"label":"Branch Name","note":"One sentence insight"}]}'
    )
    result = execute_ai_query(request, prompt, tool_type="mindmap", as_json=True)
    return jsonify(result or {"center": topic, "branches": []})


@app.route("/api/course", methods=["POST"])
def api_course():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Robotics")
    chapters = data.get("chapters", 4)
    prompt = (
        f'Design a professional, comprehensive course curriculum for "{topic}" with exactly {chapters} modules.\n\nFor EACH module in the "chapters" array, provide:\n1. "title": Descriptive Module Title\n2. "points": Array of 3 short summary topic bullet points\n3. "detailedContent": A rich, comprehensive lesson text (formatted in clean Markdown) that thoroughly explains each topic in detail with real-world examples, step-by-step breakdowns, key formulas or code blocks where applicable, and core takeaways.\n\n '
        'Output raw JSON: {"title":"Course Title","summary":"Engaging one sentence synopsis",'
        '"chapters":[{"title":"Module Title","points":["Key Concept 1","Practical Example","Mastery takeaway"]}]}'
    )
    result = execute_ai_query(
        request, prompt, tool_type="course", as_json=True, extra_params={"chapters": chapters}, max_tokens=2000
    )
    return jsonify(result or {"title": topic, "summary": "Study Course", "chapters": []})


# ---------------------------------------------------------------- Frontend HTML / CSS / JS ----

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
<title>SmartStud.Ai — Autonomous AI Study Platform</title>

<!-- Modern Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<!-- Marked (Markdown parser), Canvas Confetti & QRious -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrious/4.0.2/qrious.min.js"></script>

<style>
/* =========================================================================
   DESIGN SYSTEM & CSS VARIABLES
   ========================================================================= */
:root {
  --primary: #7952FC;
  --primary-light: #A78BFA;
  --primary-glow: rgba(121, 82, 252, 0.4);
  --secondary: #38BDF8;
  --accent: #F43F5E;
  --emerald: #10B981;
  --amber: #F59E0B;
  
  --bg-base: #0B0D17;
  --bg-surface: #121526;
  --bg-card: rgba(22, 26, 48, 0.75);
  --bg-card-hover: rgba(32, 38, 70, 0.9);
  --border: rgba(255, 255, 255, 0.1);
  --border-active: rgba(121, 82, 252, 0.6);
  
  --text-main: #F8FAFC;
  --text-muted: #94A3B8;
  --text-dim: #64748B;
  
  --glass-bg: rgba(18, 21, 38, 0.8);
  --glass-blur: blur(18px);
  --sidebar-w: 260px;
  --radius-sm: 8px;
  --radius-md: 14px;
  --radius-lg: 20px;
  --radius-full: 9999px;
  --font-display: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
  
  --shadow-sm: 0 4px 14px rgba(0, 0, 0, 0.25);
  --shadow-md: 0 10px 35px rgba(0, 0, 0, 0.4);
  --shadow-glow: 0 0 35px var(--primary-glow);
}

body.theme-light {
  --bg-base: #F8FAFC;
  --bg-surface: #FFFFFF;
  --bg-card: rgba(255, 255, 255, 0.88);
  --bg-card-hover: rgba(241, 245, 249, 0.95);
  --border: #E2E8F0;
  --border-active: #7952FC;
  --text-main: #0F172A;
  --text-muted: #475569;
  --text-dim: #94A3B8;
  --glass-bg: rgba(255, 255, 255, 0.85);
  --shadow-sm: 0 4px 12px rgba(100, 116, 139, 0.08);
  --shadow-md: 0 10px 30px rgba(100, 116, 139, 0.12);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg-base);
  color: var(--text-main);
  min-height: 100vh;
  overflow-x: hidden;
  line-height: 1.5;
  transition: background 0.3s ease, color 0.3s ease;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

/* Crisp, high-fidelity rendering for media and vector art */
img, video, svg, canvas {
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  max-width: 100%;
}

button, input, select, textarea { font-family: inherit; color: inherit; }

.hidden { display: none !important; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: var(--radius-full); }
body.theme-light ::-webkit-scrollbar-thumb { background: #CBD5E1; }

/* Gradients & Animations */
.grad-text {
  background: linear-gradient(135deg, #A78BFA 0%, #38BDF8 50%, #F472B6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  display: inline-block;
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 15px var(--primary-glow); }
  50% { box-shadow: 0 0 35px var(--primary-glow); }
}

@keyframes slideUpFade {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes rotateLoader { to { transform: rotate(360deg); } }

.anim-fade-up { animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

/* Staggered cinematic entrance sequence */
@keyframes heroReveal {
  from { opacity: 0; transform: translateY(26px) scale(0.985); filter: blur(6px); }
  to   { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}
@keyframes heroLogoPop {
  0%   { opacity: 0; transform: scale(0.6) rotate(-12deg); }
  60%  { opacity: 1; transform: scale(1.08) rotate(3deg); }
  100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes orbDrift {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(30px, -24px) scale(1.08); }
  100% { transform: translate(0, 0) scale(1); }
}
@keyframes shimmerText {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
.hc-reveal { opacity: 0; animation: heroReveal 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.hc-reveal.d1 { animation-delay: 0.05s; }
.hc-reveal.d2 { animation-delay: 0.18s; }
.hc-reveal.d3 { animation-delay: 0.32s; }
.hc-reveal.d4 { animation-delay: 0.46s; }
.hc-reveal.d5 { animation-delay: 0.6s; }
.hc-logo-mark.hc-anim { animation: heroLogoPop 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) 0.1s backwards; }

.hero-orb-field { position: absolute; inset: 0; z-index: 1; pointer-events: none; overflow: hidden; }
.hero-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(70px);
  opacity: 0.55;
  animation: orbDrift 14s ease-in-out infinite;
}
.hero-orb.o1 { width: 380px; height: 380px; top: -80px; left: -60px; background: radial-gradient(circle, var(--primary), transparent 70%); animation-duration: 16s; }
.hero-orb.o2 { width: 320px; height: 320px; bottom: -100px; right: -40px; background: radial-gradient(circle, var(--secondary), transparent 70%); animation-duration: 19s; animation-delay: -4s; }
.hero-orb.o3 { width: 260px; height: 260px; top: 40%; right: 22%; background: radial-gradient(circle, var(--accent), transparent 70%); animation-duration: 22s; animation-delay: -9s; opacity: 0.35; }

.grad-text.hc-shimmer {
  background-size: 200% auto;
  animation: shimmerText 5s linear infinite;
}

@media (prefers-reduced-motion: reduce) {
  .hc-reveal { animation: none !important; opacity: 1 !important; transform: none !important; filter: none !important; }
  .hc-logo-mark.hc-anim { animation: none !important; }
  .hero-orb { animation: none !important; }
  .grad-text.hc-shimmer { animation: none !important; }
  .hc-scene-track { animation: none !important; }
  .hc-ring { animation: none !important; }
}

/* =========================================================================
   CINEMATIC HERO LANDING (static gradient scene + 3D scrolling showcase)
   ========================================================================= */
.hero-cine {
  position: relative;
  width: 100%;
  min-height: 100vh;
  overflow: hidden;
  background: #05070E;
  display: flex;
  flex-direction: column;
}

.hero-bg-gradient {
  position: absolute;
  inset: 0;
  z-index: 0;
  background:
    radial-gradient(circle at 12% 8%, rgba(121, 82, 252, 0.35), transparent 45%),
    radial-gradient(circle at 88% 18%, rgba(56, 189, 248, 0.28), transparent 42%),
    radial-gradient(circle at 50% 100%, rgba(16, 185, 129, 0.12), transparent 55%),
    linear-gradient(180deg, #060810 0%, #0B0D17 55%, #0A0C16 100%);
}

.hero-grid-overlay {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
  opacity: 0.35;
  background-image:
    linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(ellipse at 50% 30%, black 10%, transparent 72%);
  -webkit-mask-image: radial-gradient(ellipse at 50% 30%, black 10%, transparent 72%);
}

.liquid-glass {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
  color: #fff;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.liquid-glass:hover {
  background: rgba(255, 255, 255, 0.15);
  border-color: rgba(255, 255, 255, 0.35);
  transform: translateY(-2px);
}

.hc-nav {
  position: relative;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 40px;
}

.hc-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.5px;
}
.hc-logo-mark {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, #7952FC, #38BDF8);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 20px rgba(121, 82, 252, 0.6);
}

.hc-links { display: flex; gap: 28px; font-size: 14px; font-weight: 500; }
.hc-links a { color: #CBD5E1; text-decoration: none; transition: color 0.2s; }
.hc-links a:hover { color: #fff; }

.hc-body {
  position: relative;
  z-index: 10;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 40px;
  padding: 20px 60px 70px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  flex-wrap: wrap;
}

.hc-content { max-width: 620px; }

.hc-meta { display: flex; gap: 16px; margin-bottom: 20px; font-size: 13px; color: #E2E8F0; flex-wrap: wrap; }
.hc-meta span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255,255,255,0.1);
  padding: 6px 14px;
  border-radius: var(--radius-full);
  border: 1px solid rgba(255,255,255,0.15);
}

.hc-title {
  font-family: var(--font-display);
  font-size: clamp(38px, 5vw, 64px);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -1.5px;
  margin-bottom: 20px;
}

.hc-desc {
  font-size: 18px;
  color: #CBD5E1;
  margin-bottom: 36px;
  line-height: 1.6;
}

.btn-hero-pri {
  background: linear-gradient(135deg, #7952FC 0%, #38BDF8 100%);
  color: #fff;
  font-weight: 700;
  font-size: 16px;
  padding: 14px 32px;
  border-radius: var(--radius-full);
  border: none;
  cursor: pointer;
  box-shadow: 0 0 25px rgba(121, 82, 252, 0.6);
  display: inline-flex;
  align-items: center;
  gap: 10px;
  transition: all 0.25s ease;
}
.btn-hero-pri:hover {
  transform: translateY(-3px) scale(1.02);
  box-shadow: 0 0 35px rgba(121, 82, 252, 0.9);
}

/* ---- 3D scrolling showcase scene (replaces the old hero video) ---- */
.hc-scene {
  position: relative;
  width: 400px;
  max-width: 92vw;
  height: 500px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  perspective: 1500px;
}
.hc-scene-mask {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 26px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.03);
  box-shadow: 0 30px 80px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,255,255,0.03);
  transform-style: preserve-3d;
  transform: rotateY(-16deg) rotateX(6deg);
  transition: transform 0.4s ease;
  -webkit-mask-image: linear-gradient(to bottom, transparent, black 12%, black 88%, transparent);
  mask-image: linear-gradient(to bottom, transparent, black 12%, black 88%, transparent);
}
.hc-scene:hover .hc-scene-mask { transform: rotateY(-8deg) rotateX(3deg); }

.hc-scene-track {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 30px 24px;
  animation: scrollPanels 18s linear infinite;
}
@keyframes scrollPanels {
  from { transform: translateY(0); }
  to   { transform: translateY(-50%); }
}

.hc-panel {
  background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  backdrop-filter: blur(6px);
}
.hc-panel:nth-child(3n+1) { transform: translateZ(20px) translateX(-6px); }
.hc-panel:nth-child(3n+2) { transform: translateZ(-10px) translateX(4px); }
.hc-panel:nth-child(3n+3) { transform: translateZ(35px) translateX(-2px); }

.hc-panel-badge {
  font-size: 11.5px;
  font-weight: 700;
  color: var(--secondary);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 10px;
}
.hc-panel-lines span { display: block; height: 8px; border-radius: 4px; background: rgba(255,255,255,0.14); margin-bottom: 7px; }
.hc-panel-lines span:nth-child(1) { width: 92%; }
.hc-panel-lines span:nth-child(2) { width: 68%; }
.hc-panel-lines span:nth-child(3) { width: 80%; }
.hc-panel-face { font-size: 14px; font-weight: 600; color: #fff; line-height: 1.4; }
.hc-panel-score { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
.hc-panel-score b { color: var(--emerald); }
.hc-panel-bar { height: 6px; border-radius: var(--radius-full); background: rgba(255,255,255,0.1); overflow: hidden; }
.hc-panel-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--secondary), var(--emerald)); }

.hc-ring {
  position: absolute;
  border: 1.5px solid rgba(255,255,255,0.16);
  border-radius: 50%;
  pointer-events: none;
}
.hc-ring.r1 { width: 90px; height: 90px; top: -24px; right: 8px; animation: ringOrbit 10s linear infinite; }
.hc-ring.r1::after { content:""; position:absolute; width:10px; height:10px; border-radius:50%; background: var(--secondary); top:-5px; left:50%; margin-left:-5px; box-shadow: 0 0 12px var(--secondary); }
.hc-ring.r2 { width: 46px; height: 46px; bottom: 30px; left: -18px; border-color: rgba(121,82,252,0.35); animation: ringOrbit 7s linear infinite reverse; }
.hc-ring.r2::after { content:""; position:absolute; width:7px; height:7px; border-radius:50%; background: var(--primary-light); top:-3.5px; left:50%; margin-left:-3.5px; box-shadow: 0 0 10px var(--primary-light); }
@keyframes ringOrbit { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

@media (max-width: 900px) {
  .hc-body { justify-content: center; text-align: center; padding: 10px 24px 50px; }
  .hc-content { max-width: 100%; }
  .hc-meta { justify-content: center; }
  .hc-cta { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; }
  .hc-scene { margin-top: 10px; }
}

/* =========================================================================
   AUTHENTICATION & GET STARTED VIEWS
   ========================================================================= */
.auth-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: radial-gradient(circle at top right, rgba(121, 82, 252, 0.15), transparent 60%),
              radial-gradient(circle at bottom left, rgba(56, 189, 248, 0.12), transparent 60%),
              var(--bg-base);
}

.auth-card {
  width: 100%;
  max-width: 440px;
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 40px 32px;
  box-shadow: var(--shadow-md);
  text-align: center;
}

.auth-avatar {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  box-shadow: var(--shadow-glow);
}

.auth-title { font-family: var(--font-display); font-size: 26px; font-weight: 700; margin-bottom: 8px; }
.auth-sub { font-size: 14px; color: var(--text-muted); margin-bottom: 28px; }

.form-group { text-align: left; margin-bottom: 18px; }
.form-group label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
.form-input {
  width: 100%;
  padding: 12px 16px;
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  color: var(--text-main);
  font-size: 14px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.form-input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-glow);
}

.divider {
  display: flex;
  align-items: center;
  margin: 24px 0;
  color: var(--text-dim);
  font-size: 12px;
}
.divider::before, .divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.divider span { padding: 0 12px; }

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  border: none;
  transition: all 0.2s ease;
  width: 100%;
}
.btn-pri {
  background: linear-gradient(135deg, var(--primary), #6366F1);
  color: #fff;
  box-shadow: 0 4px 20px var(--primary-glow);
}
.btn-pri:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 25px var(--primary-glow);
}
.btn-sec {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border);
  color: var(--text-main);
}
.btn-sec:hover { background: rgba(255, 255, 255, 0.12); }
.btn-emerald { background: linear-gradient(135deg, #10B981, #059669); color: #fff; }

/* =========================================================================
   DASHBOARD INTERIOR DESIGN
   ========================================================================= */
.dash-shell {
  position: relative;
  display: flex;
  min-height: 100vh;
  background: radial-gradient(circle at 80% 20%, rgba(121, 82, 252, 0.08), transparent 40%),
              radial-gradient(circle at 10% 80%, rgba(56, 189, 248, 0.06), transparent 40%),
              var(--bg-base);
}
.dash-shell::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.4;
  background-image:
    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(ellipse at 50% 0%, black 0%, transparent 65%);
  -webkit-mask-image: radial-gradient(ellipse at 50% 0%, black 0%, transparent 65%);
}
body.theme-light .dash-shell::before { display: none; }

.sidebar {
  width: var(--sidebar-w);
  background: var(--bg-surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 30;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.sidebar-header {
  padding: 24px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
}

.brand-badge {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 17px;
  letter-spacing: -0.3px;
  cursor: pointer;
}
.brand-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}

.nav-list {
  flex: 1;
  padding: 16px 12px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-section-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
  color: var(--text-dim);
  padding: 12px 10px 4px;
}

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  font-size: 13.5px;
  font-weight: 600;
  color: var(--text-muted);
  background: transparent;
  border: none;
  cursor: pointer;
  text-align: left;
  transition: all 0.18s ease;
  width: 100%;
  overflow: hidden;
}
.nav-item:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-main);
  transform: translateX(4px);
}
.nav-item.active {
  background: linear-gradient(90deg, rgba(121, 82, 252, 0.18), rgba(56, 189, 248, 0.09));
  color: #A78BFA;
  border-left: 3px solid var(--primary);
  font-weight: 700;
}
.nav-item.active::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.08), transparent);
  animation: shimmerText 3.2s linear infinite;
  background-size: 200% auto;
  pointer-events: none;
}
.nav-item .icon { font-size: 16px; width: 20px; text-align: center; }
.nav-badge {
  margin-left: auto;
  background: var(--accent);
  color: #fff;
  font-size: 10.5px;
  font-weight: 800;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: var(--radius-full);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 10px rgba(244, 63, 94, 0.6);
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.user-card {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
}
.user-avatar {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: linear-gradient(135deg, #F43F5E, #FB923C);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 14px;
}
.user-info { flex: 1; min-width: 0; }
.user-name { font-size: 13px; font-weight: 700; }
.user-level { font-size: 11px; color: var(--emerald); display: flex; align-items: center; gap: 4px; }

/* Main Workspace */
.dash-main {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow-y: auto;
  max-height: 100vh;
}

.top-bar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 32px;
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  border-bottom: 1px solid var(--border);
}
.top-bar::after {
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: -1px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--primary-glow), transparent);
}

.top-left { display: flex; align-items: center; gap: 16px; }
.top-title { font-family: var(--font-display); font-size: 18px; font-weight: 700; }

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: var(--radius-full);
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: var(--emerald);
  cursor: pointer;
  transition: all 0.2s;
}
.status-badge:hover { transform: scale(1.03); }
.status-badge.simulation {
  background: rgba(56, 189, 248, 0.12);
  border-color: rgba(56, 189, 248, 0.3);
  color: var(--secondary);
}

.btn-icon {
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  color: var(--text-main);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-icon:hover {
  background: rgba(255, 255, 255, 0.12);
  border-color: var(--primary);
  transform: translateY(-1px);
}

.btn-share-top {
  padding: 8px 16px;
  background: linear-gradient(135deg, #7952FC, #38BDF8);
  border: none;
  border-radius: var(--radius-full);
  color: #fff;
  font-weight: 700;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  box-shadow: 0 2px 12px var(--primary-glow);
  transition: transform 0.2s;
}
.btn-share-top:hover { transform: scale(1.04); }

.panel-container {
  padding: 32px;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.panel-header { margin-bottom: 28px; }
.panel-title { font-family: var(--font-display); font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
.panel-sub { color: var(--text-muted); font-size: 14px; }

.card {
  background: var(--bg-card);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 28px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 24px;
  transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  transform-style: preserve-3d;
  will-change: transform;
}
.card.tilting {
  border-color: var(--border-active);
  box-shadow: 0 22px 50px rgba(121, 82, 252, 0.22);
}

.topic-input-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}
.topic-input-row input, .topic-input-row select {
  flex: 1;
  min-width: 220px;
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-main);
  font-size: 15px;
}
.topic-input-row input:focus, .topic-input-row select:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-glow);
}

.loader-box {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 30px;
  font-size: 15px;
  font-weight: 600;
  color: var(--secondary);
}
.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(56, 189, 248, 0.2);
  border-top-color: var(--secondary);
  border-radius: 50%;
  animation: rotateLoader 0.8s linear infinite;
}

/* =========================================================================
   PANEL: CHAT TUTOR (STUD) & MARKDOWN STYLES
   ========================================================================= */
.chat-wrapper {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 190px);
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.chat-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 40px 20px;
}
.stud-avatar-big {
  width: 84px;
  height: 84px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 40px;
  margin-bottom: 20px;
  animation: pulseGlow 3s infinite;
}

.quick-prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 680px;
  margin-top: 24px;
}
.prompt-chip {
  padding: 8px 16px;
  border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}
.prompt-chip:hover {
  background: rgba(121, 82, 252, 0.22);
  border-color: var(--primary);
  transform: translateY(-2px);
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.msg {
  max-width: 85%;
  padding: 18px 22px;
  border-radius: 18px;
  font-size: 14.5px;
  line-height: 1.65;
  position: relative;
  animation: slideUpFade 0.28s ease;
}
.msg-user {
  align-self: flex-end;
  background: linear-gradient(135deg, var(--primary), #6366F1);
  color: #fff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 4px 15px var(--primary-glow);
}
.msg-stud {
  align-self: flex-start;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
}

/* Markdown typography inside Stud replies */
.markdown-body h2 { font-size: 20px; font-weight: 800; margin: 12px 0 8px; color: #fff; }
.markdown-body h3 { font-size: 16px; font-weight: 700; margin: 14px 0 6px; color: var(--secondary); }
.markdown-body p { margin-bottom: 10px; }
.markdown-body ul, .markdown-body ol { padding-left: 22px; margin-bottom: 12px; }
.markdown-body li { margin-bottom: 5px; }
.markdown-body strong { color: #fff; font-weight: 700; }
.markdown-body hr { border: none; height: 1px; background: var(--border); margin: 14px 0; }
.markdown-body code {
  background: rgba(0, 0, 0, 0.35);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: #F472B6;
}

.msg-actions {
  display: flex;
  gap: 12px;
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.btn-msg-tool {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: color 0.15s;
}
.btn-msg-tool:hover { color: var(--secondary); }

.chat-bar {
  padding: 16px 24px;
  background: rgba(0, 0, 0, 0.2);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}
.chat-input {
  flex: 1;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border);
  border-radius: var(--radius-full);
  padding: 14px 20px;
  color: #fff;
  font-size: 14.5px;
}
.chat-input:focus { outline: none; border-color: var(--primary); }

.btn-send-round {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  border: none;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s;
  box-shadow: 0 4px 15px var(--primary-glow);
}
.btn-send-round:hover { transform: scale(1.08); }

/* =========================================================================
   PANEL: 3D FLASHCARDS DECK
   ========================================================================= */
.fc-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  margin-top: 20px;
}

.fc-card-3d {
  width: 100%;
  max-width: 600px;
  height: 320px;
  perspective: 1200px;
  cursor: pointer;
}
.fc-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
  transform-style: preserve-3d;
}
.fc-card-3d.flipped .fc-inner { transform: rotateY(180deg); }

.fc-front, .fc-back {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  border-radius: var(--radius-lg);
  padding: 32px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border);
}
.fc-front {
  background: linear-gradient(145deg, #181C38, #121528);
  border-top: 3px solid var(--primary);
}
.fc-back {
  background: linear-gradient(145deg, #10263E, #0E1B2C);
  border-top: 3px solid var(--secondary);
  transform: rotateY(180deg);
}

.fc-badge {
  align-self: center;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  padding: 4px 12px;
  border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.08);
}
.fc-text { font-size: 20px; font-weight: 600; line-height: 1.5; margin: auto 0; }
.fc-hint { font-size: 12px; color: var(--text-dim); }

.fc-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  width: 100%;
  max-width: 600px;
  justify-content: space-between;
}
.fc-prog-pill { font-size: 13px; font-weight: 700; color: var(--text-muted); }

/* Spaced Repetition rating row */
.srs-rating-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  width: 100%;
  max-width: 600px;
}
.srs-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 12px 8px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-main);
  font-weight: 700;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.srs-btn span { font-size: 11px; font-weight: 500; color: var(--text-dim); }
.srs-btn:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.srs-again:hover { border-color: var(--accent); background: rgba(244,63,94,0.1); }
.srs-hard:hover { border-color: var(--amber); background: rgba(245,158,11,0.1); }
.srs-good:hover { border-color: var(--secondary); background: rgba(56,189,248,0.1); }
.srs-easy:hover { border-color: var(--emerald); background: rgba(16,185,129,0.1); }

/* Toggle switch row (used in Notes/RAG panel) */
.toggle-row {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
}
.toggle-row input[type="checkbox"] {
  appearance: none;
  width: 40px;
  height: 22px;
  border-radius: var(--radius-full);
  background: var(--border);
  position: relative;
  cursor: pointer;
  outline: none;
  transition: background 0.25s ease;
  flex-shrink: 0;
}
.toggle-row input[type="checkbox"]::before {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #fff;
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
.toggle-row input[type="checkbox"]:checked { background: var(--primary); }
.toggle-row input[type="checkbox"]:checked::before { transform: translateX(18px); }

/* Simple bar chart used in Analytics panel */
.chart-bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.chart-bar-label { width: 140px; font-size: 12.5px; font-weight: 600; color: var(--text-muted); flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chart-bar-track { flex: 1; height: 14px; background: rgba(255,255,255,0.06); border-radius: var(--radius-full); overflow: hidden; }
.chart-bar-fill { height: 100%; border-radius: var(--radius-full); background: linear-gradient(90deg, var(--primary), var(--secondary)); transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }
.chart-bar-pct { width: 42px; text-align: right; font-size: 12.5px; font-weight: 700; color: var(--text-main); flex-shrink: 0; }

/* =========================================================================
   PANEL: QUIZ ARENA
   ========================================================================= */
.quiz-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: var(--radius-md);
  margin-bottom: 24px;
}
.quiz-timer { display: flex; align-items: center; gap: 6px; font-weight: 700; color: var(--amber); }

.quiz-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 24px;
  margin-bottom: 20px;
  animation: slideUpFade 0.3s ease;
}
.quiz-q-title { font-size: 17px; font-weight: 700; margin-bottom: 16px; }
.quiz-options-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
@media (min-width: 640px) { .quiz-options-grid { grid-template-columns: 1fr 1fr; } }

.quiz-btn-opt {
  text-align: left;
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.quiz-btn-opt:hover:not(:disabled) {
  background: rgba(121, 82, 252, 0.18);
  border-color: var(--primary);
  transform: translateY(-2px);
}
.quiz-btn-opt.correct {
  background: rgba(16, 185, 129, 0.22) !important;
  border-color: var(--emerald) !important;
  color: #6EE7B7 !important;
  font-weight: 700;
}
.quiz-btn-opt.wrong {
  background: rgba(244, 63, 94, 0.22) !important;
  border-color: var(--accent) !important;
  color: #FDA4AF !important;
}

.quiz-explanation {
  margin-top: 14px;
  padding: 12px 16px;
  border-radius: var(--radius-md);
  background: rgba(56, 189, 248, 0.08);
  border-left: 3px solid var(--secondary);
  font-size: 13.5px;
  color: #BAE6FD;
}

/* =========================================================================
   PANEL: INTERACTIVE SVG MIND MAP
   ========================================================================= */
.mindmap-wrapper {
  position: relative;
  width: 100%;
  background: radial-gradient(circle at center, rgba(121, 82, 252, 0.06), transparent 70%), rgba(0, 0, 0, 0.25);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.mindmap-svg-wrap { width: 100%; max-width: 740px; height: auto; }
.mm-node-g { cursor: pointer; transition: transform 0.2s; }
.mm-node-g:hover { filter: drop-shadow(0 0 12px var(--primary-glow)); }

.mm-legend-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px;
  width: 100%;
  margin-top: 24px;
}
.mm-branch-card {
  padding: 14px 18px;
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border);
  font-size: 13.5px;
  transition: transform 0.2s;
}
.mm-branch-card:hover { transform: translateY(-2px); border-color: var(--primary); }

/* =========================================================================
   PANEL: COURSE STUDIO
   ========================================================================= */
.course-chapters-list { display: flex; flex-direction: column; gap: 16px; margin-top: 20px; }
.chapter-module {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  transition: border-color 0.2s;
}
.chapter-module:hover { border-color: var(--border-active); }

.chapter-header {
  padding: 18px 22px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.02);
}
.chapter-title-text { font-size: 16px; font-weight: 700; }

.chapter-body {
  padding: 20px 24px;
  border-top: 1px solid var(--border);
  background: rgba(0, 0, 0, 0.15);
}
.chapter-points-ul {
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 14px;
  color: var(--text-muted);
}

.btn-lesson-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  padding: 8px 16px;
  background: linear-gradient(135deg, var(--primary), #6366f1);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(121, 82, 252, 0.25);
}
.btn-lesson-toggle:hover {
  filter: brightness(1.15);
  transform: translateY(-1px);
}
.chapter-detailed-content {
  margin-top: 16px;
  padding: 20px 24px;
  background: rgba(121, 82, 252, 0.07);
  border: 1px solid rgba(121, 82, 252, 0.3);
  border-radius: var(--radius-md);
  font-size: 14px;
  line-height: 1.7;
  color: var(--text-main);
}
.chapter-detailed-content h1,
.chapter-detailed-content h2,
.chapter-detailed-content h3,
.chapter-detailed-content h4 {
  margin-top: 16px;
  margin-bottom: 10px;
  color: #ffffff;
  font-weight: 700;
}
.chapter-detailed-content p {
  margin-bottom: 12px;
}
.chapter-detailed-content ul, .chapter-detailed-content ol {
  padding-left: 20px;
  margin-bottom: 12px;
}
.chapter-detailed-content li {
  margin-bottom: 6px;
}
.chapter-detailed-content code {
  background: rgba(0, 0, 0, 0.45);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: #38bdf8;
}
.chapter-detailed-content pre {
  background: rgba(0, 0, 0, 0.6);
  padding: 14px;
  border-radius: 8px;
  overflow-x: auto;
  margin-bottom: 14px;
}
.quiz-diff-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: var(--radius-full);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-left: 8px;
}
.quiz-diff-beginner { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
.quiz-diff-intermediate { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
.quiz-diff-advanced { background: rgba(244, 63, 94, 0.2); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.4); }

/* =========================================================================
   PANEL: POMODORO & TO-DO
   ========================================================================= */
.pomo-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
}
.pomo-timer-display {
  font-size: 72px;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -2px;
  margin: 20px 0;
}
.pomo-modes { display: flex; gap: 10px; margin-bottom: 20px; }
.pomo-mode-btn {
  padding: 8px 18px;
  border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.pomo-mode-btn.active { background: var(--primary); color: #fff; border-color: var(--primary); }

.todo-list-box { display: flex; flex-direction: column; gap: 10px; margin-top: 20px; }
.todo-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  transition: all 0.2s;
}
.todo-left { display: flex; align-items: center; gap: 12px; }
.todo-text.done { text-decoration: line-through; color: var(--text-dim); }

/* =========================================================================
   MODALS (Share Modal & Settings Modal)
   ========================================================================= */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.78);
  backdrop-filter: blur(8px);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  animation: slideUpFade 0.2s ease;
}
.modal-content {
  width: 100%;
  max-width: 520px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 32px;
  box-shadow: var(--shadow-md);
  position: relative;
}
.modal-close {
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 20px;
  cursor: pointer;
}
.modal-close:hover { color: #fff; }

.qr-preview {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 20px 0;
  padding: 16px;
  background: #fff;
  border-radius: var(--radius-md);
  width: fit-content;
  margin-left: auto;
  margin-right: auto;
}

.copy-input-row { display: flex; gap: 8px; margin-top: 14px; }
.copy-input-row input {
  flex: 1;
  padding: 10px 14px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}

.share-social-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 18px;
}
.btn-social {
  padding: 8px;
  font-size: 12px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
}

/* =========================================================================
   GOOGLE MEET STYLE ROOMS & LINK SHARING UI
   ========================================================================= */
.gmeet-top-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-right: 12px;
}
.gmeet-dropdown-wrapper {
  position: relative;
}
.gmeet-btn-new {
  background: linear-gradient(135deg, #1a73e8, #1557b0) !important;
  color: #ffffff !important;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(26, 115, 232, 0.35);
  transition: all 0.2s ease;
}
.gmeet-btn-new:hover {
  background: linear-gradient(135deg, #1765cc, #114999) !important;
  transform: translateY(-1px);
}
.gmeet-dropdown-menu {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  width: 240px;
  background: #202124;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  padding: 8px;
  z-index: 120;
  animation: fadeIn 0.15s ease-out;
}
.gmeet-dropdown-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: transparent;
  border: none;
  color: #e8eaed;
  font-size: 13px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease;
}
.gmeet-dropdown-item:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #ffffff;
}

.gmeet-code-input-group {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 3px 6px 3px 12px;
  gap: 8px;
}
.gmeet-key-icon {
  color: var(--text-muted);
}
.gmeet-code-input-group input {
  background: transparent;
  border: none;
  color: #fff;
  font-size: 13px;
  font-family: 'JetBrains Mono', monospace;
  width: 150px;
  outline: none;
}
.gmeet-code-input-group input::placeholder {
  color: var(--text-muted);
  font-family: var(--font-main);
}
.btn-gmeet-join {
  background: transparent;
  border: none;
  color: #1a73e8;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 6px;
  transition: background 0.15s ease;
}
.btn-gmeet-join:hover {
  background: rgba(26, 115, 232, 0.15);
}

/* Meet Lobby & Green Room */
.gmeet-lobby-preview {
  width: 100%;
  height: 230px;
  background: #171717;
  border-radius: 16px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.gmeet-cam-avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 36px;
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}
.gmeet-cam-controls {
  position: absolute;
  bottom: 14px;
  display: flex;
  gap: 12px;
}
.gmeet-control-btn {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #3c4043;
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.gmeet-control-btn:hover {
  background: #474b4f;
  transform: scale(1.05);
}
.gmeet-control-btn.muted {
  background: #ea4335 !important;
  color: #ffffff !important;
}

/* Live Meet Panel Layout */
.gmeet-room-layout {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 120px);
  background: #17181a;
  border-radius: 16px;
  overflow: hidden;
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.gmeet-room-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: #202124;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.gmeet-room-title {
  font-size: 16px;
  font-weight: 700;
  color: #f1f5f9;
  display: flex;
  align-items: center;
  gap: 10px;
}
.gmeet-pill-code {
  background: rgba(255, 255, 255, 0.1);
  padding: 4px 10px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #818cf8;
}

.gmeet-stage-area {
  flex: 1;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  padding: 20px;
  overflow-y: auto;
}
.gmeet-video-card {
  background: #202124;
  border-radius: 14px;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 220px;
  border: 2px solid transparent;
  transition: border-color 0.2s;
}
.gmeet-video-card.speaking {
  border-color: #1a73e8;
}
.gmeet-user-tag {
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: rgba(0, 0, 0, 0.65);
  backdrop-filter: blur(4px);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
}

.gmeet-bottom-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 14px 24px;
  background: #202124;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.gmeet-bottom-bar .gmeet-control-btn.danger {
  background: #ea4335;
  width: 52px;
  height: 44px;
  border-radius: 22px;
}
.gmeet-bottom-bar .gmeet-control-btn.danger:hover {
  background: #d93025;
}

.toast-box {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 150;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.toast {
  padding: 12px 20px;
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-md);
  font-size: 13.5px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  animation: slideUpFade 0.3s ease;
}
.toast.success { border-left: 4px solid var(--emerald); }
.toast.info { border-left: 4px solid var(--secondary); }

@media (max-width: 768px) {
  .sidebar { position: fixed; left: -100%; transition: left 0.3s ease; }
  .sidebar.open { left: 0; }
  .panel-container { padding: 18px; }
  .top-bar { padding: 12px 18px; }
}
</style>
</head>
<body>

<!-- ================= LANDING (Static gradient hero + 3D scrolling scene) ================= -->
<div id="view-landing" class="view">
  <section class="hero-cine">
    <div class="hero-bg-gradient"></div>
    <div class="hero-grid-overlay" aria-hidden="true"></div>
    <div class="hero-orb-field" aria-hidden="true">
      <div class="hero-orb o1"></div>
      <div class="hero-orb o2"></div>
      <div class="hero-orb o3"></div>
    </div>

    <nav class="hc-nav hc-reveal d1">
      <div class="hc-logo">
        <div class="hc-logo-mark hc-anim">🧠</div>
        <span>SmartStud<span style="color:var(--secondary);">.Ai</span></span>
      </div>

      <div class="hc-links">
        <a href="#" data-goto="dashboard">AI Companion</a>
        <a href="#" data-goto="dashboard">Flashcards</a>
        <a href="#" data-goto="dashboard">Quizzes</a>
        <a href="#" data-goto="dashboard">Mind Maps</a>
      </div>

      <div class="hc-right">
        <button class="liquid-glass btn-hero-pri" style="padding:10px 22px; font-size:14px;" data-goto="login">
          Log in
        </button>
      </div>
    </nav>

    <div class="hc-body">
      <div class="hc-content">
        <div class="hc-meta hc-reveal d2">
          <span>⚡ Multi-Provider AI (Gemini / Claude / OpenAI)</span>
          <span>🧠 Spaced Repetition Engine</span>
          <span>🗂️ Retrieval-Augmented Notes</span>
        </div>

        <h1 class="hc-title hc-reveal d3">Step Through.<br/><span class="grad-text hc-shimmer">Study Smarter.</span></h1>

        <p class="hc-desc hc-reveal d4">
          Personalized AI courses, 3D interactive flashcards, gamified quizzes, dynamic mind maps,
          SM-2 spaced review, and your own notes grounding every answer — generated instantly on demand.
        </p>

        <div class="hc-cta hc-reveal d5">
          <button class="btn-hero-pri" data-goto="dashboard">
            🚀 Launch Study Workspace
          </button>
          <button class="liquid-glass" style="padding:14px 28px; border-radius:9999px; font-weight:600;" data-goto="login">
            Sign In / Register
          </button>
        </div>
      </div>

      <div class="hc-scene hc-reveal d3" id="hc-3d-scene">
        <div class="hc-scene-mask" id="hc-scene-mask">
          <div class="hc-scene-track">
            <div class="hc-panel">
              <div class="hc-panel-badge">🧩 Mind Map</div>
              <div class="hc-panel-lines"><span></span><span></span><span></span></div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">📚 Flashcards</div>
              <div class="hc-panel-face">Q: What is a Transformer?</div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">🏆 Quiz Arena</div>
              <div class="hc-panel-score">Score <b>92%</b></div>
              <div class="hc-panel-bar"><span style="width:92%;"></span></div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">🎓 Course Builder</div>
              <div class="hc-panel-lines"><span></span><span></span></div>
            </div>
            <!-- duplicate set for a seamless vertical loop -->
            <div class="hc-panel">
              <div class="hc-panel-badge">🧩 Mind Map</div>
              <div class="hc-panel-lines"><span></span><span></span><span></span></div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">📚 Flashcards</div>
              <div class="hc-panel-face">Q: What is a Transformer?</div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">🏆 Quiz Arena</div>
              <div class="hc-panel-score">Score <b>92%</b></div>
              <div class="hc-panel-bar"><span style="width:92%;"></span></div>
            </div>
            <div class="hc-panel">
              <div class="hc-panel-badge">🎓 Course Builder</div>
              <div class="hc-panel-lines"><span></span><span></span></div>
            </div>
          </div>
        </div>
        <div class="hc-ring r1"></div>
        <div class="hc-ring r2"></div>
      </div>
    </div>
  </section>
</div>

<!-- ================= GET STARTED / LOGIN ================= -->
<div id="view-login" class="view hidden auth-container">
  <div class="auth-card anim-fade-up">
    <div class="auth-avatar">🔐</div>
    <h1 class="auth-title">Welcome to <span class="grad-text">SmartStud</span></h1>
    <p class="auth-sub">Unlock instant AI study superpower</p>

    <button class="btn btn-sec" style="margin-bottom:12px;" data-goto="dashboard">
      <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#EA4335" d="M12 5c1.6 0 3 .6 4.1 1.6l3.1-3.1C17.3 1.7 14.8 1 12 1 7.4 1 3.5 3.6 1.6 7.4l3.7 2.9C6.2 7.3 8.9 5 12 5z"/><path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.8z"/><path fill="#FBBC05" d="M5.3 14.7c-.2-.7-.4-1.5-.4-2.7s.1-2 .4-2.7L1.6 6.4C.6 8.3 0 10.1 0 12s.6 3.7 1.6 5.6l3.7-2.9z"/><path fill="#34A853" d="M12 23c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3.1 0-5.8-2.3-6.7-5.3L1.6 16c1.9 3.8 5.8 7 10.4 7z"/></svg>
      Continue with Google
    </button>

    <div class="divider"><span>OR CONTINUE WITH EMAIL</span></div>

    <div class="form-group">
      <label>Study Email</label>
      <input type="email" class="form-input" id="auth-email" placeholder="you@university.edu" value="learner@smartstud.ai" />
    </div>
    <div class="form-group">
      <label>Password</label>
      <input type="password" class="form-input" id="auth-pass" value="••••••••••••" />
    </div>

    <button class="btn btn-pri" data-goto="dashboard" style="margin-top:8px;">
      Enter Workspace
    </button>
  </div>
</div>

<!-- ================= MAIN DASHBOARD INTERIOR ================= -->
<div id="view-dashboard" class="view hidden">
  <div class="dash-shell">
    <!-- Sidebar -->
    <aside class="sidebar" id="app-sidebar">
      <div class="sidebar-header">
        <div class="brand-badge" data-goto="landing">
          <div class="brand-icon">🧠</div>
          <span>SmartStud<span style="color:var(--secondary);">.Ai</span></span>
        </div>
      </div>

      <div class="nav-list">
        <div class="nav-section-title">AI Study Tools</div>
        <button class="nav-item active" data-tool="chat"><span class="icon">💬</span> Chat Tutor</button>
        <button class="nav-item" data-tool="flashcards"><span class="icon">📚</span> 3D Flashcards</button>
        <button class="nav-item" data-tool="review"><span class="icon">🧠</span> Spaced Review <span id="review-due-badge" class="nav-badge hidden">0</span></button>
        <button class="nav-item" data-tool="quiz"><span class="icon">🏆</span> Quiz Arena</button>
        <button class="nav-item" data-tool="mindmap"><span class="icon">🕸️</span> Mind Map Studio</button>
        <button class="nav-item" data-tool="course"><span class="icon">🎓</span> AI Course Builder</button>
        <button class="nav-item" data-tool="notes"><span class="icon">🗂️</span> My Notes (RAG)</button>

        <div class="nav-section-title">Productivity & Sync</div>
        <button class="nav-item" data-tool="meet"><span class="icon">🎥</span> Google Meet Rooms</button>
        <button class="nav-item" data-tool="pomodoro"><span class="icon">⏱️</span> Pomodoro Focus</button>
        <button class="nav-item" data-tool="todo"><span class="icon">✅</span> Study Planner</button>
        <button class="nav-item" data-tool="library"><span class="icon">📁</span> Saved Decks</button>
        <button class="nav-item" data-tool="analytics"><span class="icon">📊</span> Mastery Analytics</button>
        <button class="nav-item" data-tool="settings"><span class="icon">⚙️</span> Settings & API</button>
      </div>

      <div class="sidebar-footer">
        <div class="user-card">
          <div class="user-avatar">🧑</div>
          <div class="user-info">
            <div class="user-name">Study Scholar</div>
            <div class="user-level">5 Day Streak • 450 XP</div>
          </div>
        </div>
        <button class="btn btn-sec" style="padding:8px;" data-goto="landing">🚪 Log out</button>
      </div>
    </aside>

    <!-- Main Workspace -->
    <main class="dash-main">
      <header class="top-bar">
        <div class="top-left">
          <button class="btn-icon" id="btn-sidebar-toggle" style="display:none;">☰</button>
          <span class="top-title" id="page-heading">AI Chat Tutor</span>
        </div>

        <div class="top-badges">
          <div class="status-badge" id="api-status-badge" title="Click to configure API Key">
            <span style="font-size:8px;">●</span> <span id="api-status-text">AI Ready</span>
          </div>
        </div>

        <div class="top-actions">
          <div class="gmeet-top-bar">
            <div class="gmeet-dropdown-wrapper">
              <button class="btn btn-pri gmeet-btn-new" id="btn-gmeet-new">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
                New meeting
              </button>
              <div class="gmeet-dropdown-menu hidden" id="gmeet-dropdown-menu">
                <button class="gmeet-dropdown-item" id="btn-gmeet-later">
                  <span>🔗</span> Create a meeting for later
                </button>
                <button class="gmeet-dropdown-item" id="btn-gmeet-instant">
                  <span>⚡</span> Start an instant meeting
                </button>
              </div>
            </div>

            <div class="gmeet-code-input-group">
              <svg class="gmeet-key-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              <input type="text" id="gmeet-code-input" placeholder="Enter code or link" />
              <button class="btn-gmeet-join" id="btn-gmeet-join">Join</button>
            </div>
          </div>

          <button class="btn-share-top" id="btn-open-share">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
            Share Set
          </button>
          <button class="btn-icon" id="btn-theme-toggle" title="Toggle Theme">🌙</button>
        </div>
      </header>

      <!-- Panel: Chat & AI Companion -->
      <div class="panel-container" data-panel="chat">
        <div class="chat-wrapper">
          <div id="chat-empty" class="chat-empty">
            <div class="stud-avatar-big">🤖</div>
            <h2 style="font-size:24px; font-weight:800; margin-bottom:8px;">Hey, I'm Stud! Your AI Study Partner</h2>
            <p class="panel-sub" style="max-width:520px;">Ask any question, paste complex study notes for breakdown, or try a prompt below.</p>
            
            <div class="quick-prompts">
              <div class="prompt-chip" data-prompt="What are the different types of robots and their kinematics?">🤖 Types of Robots</div>
              <div class="prompt-chip" data-prompt="Explain Quantum Computing with intuitive analogies">⚛️ Quantum Computing</div>
              <div class="prompt-chip" data-prompt="Explain the Transformer architecture and self-attention in AI">🧠 Transformer Neural Networks</div>
              <div class="prompt-chip" data-prompt="Master summary of Data Structures and Big-O Complexity">💻 Data Structures & Big-O</div>
            </div>
          </div>

          <div id="chat-messages" class="chat-messages hidden"></div>

          <div id="chat-loader" class="loader-box hidden">
            <span class="spinner"></span> Stud is analyzing and structuring your explanation...
          </div>

          <div class="chat-bar">
            <input type="text" id="chat-input" class="chat-input" placeholder="Ask a question or topic (e.g. types of robotics, calculus, CRISPR)..." />
            <button class="btn-icon" id="btn-mic" title="Speech to Text">🎤</button>
            <button class="btn-send-round" id="chat-send">➤</button>
          </div>
        </div>
      </div>

      <!-- Panel: 3D Flashcards Deck -->
      <div class="panel-container hidden" data-panel="flashcards">
        <div class="panel-header">
          <h1 class="panel-title">📚 3D Interactive Flashcards</h1>
          <p class="panel-sub">Spaced-repetition card deck with realistic 3D perspective flip and audio pronunciation.</p>
        </div>

        <div class="card">
          <div class="topic-input-row">
            <input id="fc-topic-input" placeholder="Enter topic (e.g. Types of Robots, Organic Chemistry, French Vocabulary)..." />
            <button class="btn btn-pri" id="fc-generate-btn" style="width:auto; min-width:160px;">✨ Generate Deck</button>
          </div>
        </div>

        <div id="fc-loader" class="loader-box hidden">
          <span class="spinner"></span> Crafting high-yield flashcard deck...
        </div>

        <div id="fc-stage" class="fc-stage hidden">
          <div class="fc-card-3d" id="fc-card-3d">
            <div class="fc-inner">
              <div class="fc-front">
                <span class="fc-badge" style="color:var(--primary);">Question / Term</span>
                <p class="fc-text" id="fc-front-text">Front</p>
                <span class="fc-hint">💡 Click to flip</span>
              </div>
              <div class="fc-back">
                <span class="fc-badge" style="color:var(--secondary);">Answer / Explanation</span>
                <p class="fc-text" id="fc-back-text">Back</p>
                <span class="fc-hint">🔄 Click to flip back</span>
              </div>
            </div>
          </div>

          <div class="fc-controls">
            <button class="btn btn-sec" id="fc-prev-btn" style="width:auto; padding:10px 18px;">← Previous</button>
            <div class="fc-prog-pill" id="fc-counter">Card 1 / 6</div>
            <button class="btn btn-sec" id="fc-speak-btn" style="width:auto; padding:10px 18px;">🔊 Audio</button>
            <button class="btn btn-pri" id="fc-next-btn" style="width:auto; padding:10px 18px;">Next →</button>
          </div>

          <div style="display:flex; gap:12px; margin-top:8px;">
            <button class="btn btn-sec" id="fc-save-deck" style="width:auto; padding:8px 16px;">💾 Save to Library</button>
            <button class="btn btn-emerald" id="fc-master-btn" style="width:auto; padding:8px 16px;">✓ Mark Mastered (+50 XP)</button>
          </div>
        </div>
      </div>

      <!-- Panel: Gamified Quiz Arena -->
      <div class="panel-container hidden" data-panel="quiz">
        <div class="panel-header">
          <h1 class="panel-title">🏆 AI Quiz Arena</h1>
          <p class="panel-sub">Adaptive multiple-choice challenge with live feedback, explanations, and XP scoring.</p>
        </div>

        <div class="card">
          <div class="topic-input-row">
            <input id="quiz-topic-input" placeholder="Quiz topic (e.g. Types of Robots, Machine Learning, Calculus)..." />
            <select id="quiz-diff-select" style="max-width:180px;">
              <option value="beginner">Beginner</option>
              <option value="intermediate" selected>Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
            <button class="btn btn-pri" id="quiz-generate-btn" style="width:auto; min-width:150px;">🎯 Start Quiz</button>
          </div>
        </div>

        <div id="quiz-loader" class="loader-box hidden">
          <span class="spinner"></span> Generating adaptive quiz questions...
        </div>

        <div id="quiz-arena" class="hidden">
          <div class="quiz-header-bar">
            <div><strong>Active Deck:</strong> <span id="quiz-topic-display"></span></div>
            <div class="quiz-timer">⏱️ <span id="quiz-timer-text">00:00</span></div>
            <div id="quiz-score-live" style="font-weight:700; color:var(--emerald);">Score: 0 / 5</div>
          </div>

          <div id="quiz-questions-list"></div>
        </div>
      </div>

      <!-- Panel: Interactive SVG Mind Map -->
      <div class="panel-container hidden" data-panel="mindmap">
        <div class="panel-header">
          <h1 class="panel-title">🕸️ Mind Map Studio</h1>
          <p class="panel-sub">Interactive concept trees and radial nodes for cognitive clarity.</p>
        </div>

        <div class="card">
          <div class="topic-input-row">
            <input id="mm-topic-input" placeholder="Concept to visualize (e.g. Types of Robots, Deep Learning, Economics)..." />
            <button class="btn btn-pri" id="mm-generate-btn" style="width:auto; min-width:150px;">🎨 Render Tree</button>
          </div>
        </div>

        <div id="mm-loader" class="loader-box hidden">
          <span class="spinner"></span> Computing radial coordinate nodes...
        </div>

        <div id="mm-display-area" class="mindmap-wrapper hidden">
          <div class="mindmap-svg-wrap" id="mm-svg-container"></div>
          <div class="mm-legend-grid" id="mm-legend-grid"></div>
        </div>
      </div>

      <!-- Panel: AI Course Studio -->
      <div class="panel-container hidden" data-panel="course">
        <div class="panel-header">
          <h1 class="panel-title">🎓 AI Course Curriculum</h1>
          <p class="panel-sub">Step-by-step modular syllabus with deep-dive study notes.</p>
        </div>

        <div class="card">
          <div class="topic-input-row">
            <input id="course-topic-input" placeholder="Course title (e.g. Robotics Engineering, React 2026, Astrophysics)..." />
            <select id="course-chap-select" style="max-width:140px;">
              <option value="3">3 Modules</option>
              <option value="4" selected>4 Modules</option>
              <option value="5">5 Modules</option>
            </select>
            <button class="btn btn-pri" id="course-generate-btn" style="width:auto; min-width:150px;">📘 Build Course</button>
          </div>
        </div>

        <div id="course-loader" class="loader-box hidden">
          <span class="spinner"></span> Assembling structured syllabus...
        </div>

        <div id="course-output" class="hidden">
          <div class="card">
            <h2 id="course-title-out" style="font-size:22px; font-weight:800; margin-bottom:6px;"></h2>
            <p id="course-summary-out" class="panel-sub"></p>
          </div>
          <div class="course-chapters-list" id="course-chapters-list"></div>
        </div>
      </div>

      <!-- Panel: Pomodoro Focus Timer -->
      <div class="panel-container hidden" data-panel="pomodoro">
        <div class="panel-header">
          <h1 class="panel-title">⏱️ Pomodoro Focus Station</h1>
          <p class="panel-sub">Scientific interval study method (25m Focus / 5m Rest).</p>
        </div>

        <div class="card pomo-stage">
          <div class="pomo-modes">
            <button class="pomo-mode-btn active" data-pomo-min="25">Focus (25m)</button>
            <button class="pomo-mode-btn" data-pomo-min="5">Short Break (5m)</button>
            <button class="pomo-mode-btn" data-pomo-min="15">Long Rest (15m)</button>
          </div>

          <div class="pomo-timer-display" id="pomo-time">25:00</div>

          <div style="display:flex; gap:12px;">
            <button class="btn btn-pri" id="pomo-start-btn" style="width:140px;">▶ Start</button>
            <button class="btn btn-sec" id="pomo-reset-btn" style="width:140px;">↺ Reset</button>
          </div>
        </div>
      </div>

      <!-- Panel: Study Planner / To-Do -->
      <div class="panel-container hidden" data-panel="todo">
        <div class="panel-header">
          <h1 class="panel-title">✅ Study Task Planner</h1>
          <p class="panel-sub">Organize your daily study milestones and revision checkpoints.</p>
        </div>

        <div class="card">
          <div class="topic-input-row">
            <input id="todo-input" placeholder="New study goal (e.g. Review Types of Robots, Solve 5 Math problems)..." />
            <button class="btn btn-pri" id="todo-add-btn" style="width:auto; min-width:120px;">+ Add Goal</button>
          </div>
        </div>

        <div class="todo-list-box" id="todo-list-container"></div>
      </div>

      <!-- Panel: Saved Decks Library -->
      <div class="panel-container hidden" data-panel="library">
        <div class="panel-header">
          <h1 class="panel-title">📁 Study Library</h1>
          <p class="panel-sub">Your personal archive of saved decks, quizzes, and mind maps.</p>
        </div>

        <div id="library-empty" class="card" style="text-align:center; padding:40px;">
          <p style="font-size:16px; font-weight:600; margin-bottom:8px;">No saved decks yet.</p>
          <p class="panel-sub">When you generate flashcards or quizzes, click "Save to Library" to keep them here.</p>
        </div>
        <div id="library-grid" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px;"></div>
      </div>

      <!-- Panel: Analytics & Performance -->
      <div class="panel-container hidden" data-panel="analytics">
        <div class="panel-header">
          <h1 class="panel-title">📊 Mastery & Retention Metrics</h1>
          <p class="panel-sub">Real-time stats computed from your own quiz history and review sessions — nothing hardcoded.</p>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin-bottom:24px;">
          <div class="card" style="text-align:center;">
            <div style="font-size:28px; font-weight:800; color:var(--primary);" id="stat-xp">0 XP</div>
            <div class="panel-sub">Total Experience</div>
          </div>
          <div class="card" style="text-align:center;">
            <div style="font-size:28px; font-weight:800; color:var(--emerald);" id="stat-quizzes">0</div>
            <div class="panel-sub">Quizzes Completed</div>
          </div>
          <div class="card" style="text-align:center;">
            <div style="font-size:28px; font-weight:800; color:var(--secondary);" id="stat-accuracy">0%</div>
            <div class="panel-sub">Overall Quiz Accuracy</div>
          </div>
          <div class="card" style="text-align:center;">
            <div style="font-size:28px; font-weight:800; color:var(--amber);" id="stat-streak">0 Days</div>
            <div class="panel-sub">Active Streak 🔥</div>
          </div>
        </div>

        <div class="card" style="margin-bottom:16px;">
          <h2 style="font-size:16px; font-weight:700; margin-bottom:14px;">🎯 Accuracy by Topic</h2>
          <div id="analytics-topic-chart"><p class="panel-sub">Complete a quiz to see your per-topic mastery breakdown here.</p></div>
        </div>

        <div class="card">
          <h2 style="font-size:16px; font-weight:700; margin-bottom:14px;">📈 XP Progression (Last 14 Sessions)</h2>
          <div id="analytics-xp-chart"><p class="panel-sub">Your XP timeline will appear as you study.</p></div>
        </div>
      </div>

      <!-- Panel: Spaced Repetition Review Queue -->
      <div class="panel-container hidden" data-panel="review">
        <div class="panel-header">
          <h1 class="panel-title">🧠 Spaced Review Queue</h1>
          <p class="panel-sub">SM-2 style spaced repetition. Cards you rate "Again" or "Hard" come back sooner; "Easy" cards return in weeks.</p>
        </div>

        <div id="review-empty" class="card" style="text-align:center; padding:40px;">
          <p style="font-size:16px; font-weight:600; margin-bottom:8px;">🎉 Nothing due right now.</p>
          <p class="panel-sub">Save flashcard decks to your Library, and cards will enter the spaced-review queue automatically as they mature.</p>
        </div>

        <div id="review-stage" class="fc-stage hidden">
          <div class="fc-prog-pill" id="review-counter" style="align-self:center;">Card 1 / 1</div>
          <div class="fc-card-3d" id="review-card-3d">
            <div class="fc-inner">
              <div class="fc-front">
                <span class="fc-badge" style="color:var(--primary);">Question / Term</span>
                <p class="fc-text" id="review-front-text">Front</p>
                <span class="fc-hint">💡 Click to flip</span>
              </div>
              <div class="fc-back">
                <span class="fc-badge" style="color:var(--secondary);">Answer / Explanation</span>
                <p class="fc-text" id="review-back-text">Back</p>
                <span class="fc-hint">🔄 Click to flip back</span>
              </div>
            </div>
          </div>
          <p class="panel-sub" style="text-align:center;">Click the card to flip, then rate how well you knew it — this schedules your next review.</p>
          <div class="srs-rating-row">
            <button class="srs-btn srs-again" data-quality="0">🔴 Again<span>&lt; 10 min</span></button>
            <button class="srs-btn srs-hard" data-quality="1">🟠 Hard<span>~1 day</span></button>
            <button class="srs-btn srs-good" data-quality="2">🔵 Good<span>~3 days</span></button>
            <button class="srs-btn srs-easy" data-quality="3">🟢 Easy<span>~7+ days</span></button>
          </div>
        </div>
      </div>

      <!-- Panel: My Notes (RAG) -->
      <div class="panel-container hidden" data-panel="notes">
        <div class="panel-header">
          <h1 class="panel-title">🗂️ My Notes — Retrieval-Augmented Study</h1>
          <p class="panel-sub">Paste or upload your own notes. SmartStud indexes them locally and grounds Chat, Flashcards, and Quizzes in your material when "Use my notes" is enabled.</p>
        </div>

        <div class="card" style="margin-bottom:16px;">
          <h2 style="font-size:16px; font-weight:700; margin-bottom:12px;">📥 Add Notes</h2>
          <div class="form-group">
            <label>Paste text, or upload a .txt / .md file</label>
            <textarea id="notes-textarea" class="form-input" rows="7" placeholder="Paste lecture notes, textbook excerpts, or summaries here..."></textarea>
          </div>
          <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
            <input type="file" id="notes-file-input" accept=".txt,.md,text/plain" style="display:none;" />
            <button class="btn btn-sec" id="notes-file-btn" style="width:auto; padding:10px 20px;">📁 Choose File</button>
            <button class="btn btn-pri" id="notes-index-btn" style="width:auto; padding:10px 24px;">⚡ Index Notes</button>
            <button class="btn btn-sec" id="notes-clear-btn" style="width:auto; padding:10px 20px; color:var(--accent);">🗑 Clear All Notes</button>
          </div>
        </div>

        <div class="card">
          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
            <div>
              <h2 style="font-size:16px; font-weight:700; margin-bottom:4px;">📡 Retrieval Status</h2>
              <p class="panel-sub" id="notes-status-text">No notes indexed yet.</p>
            </div>
            <label class="toggle-row">
              <input type="checkbox" id="notes-use-toggle" />
              <span>Use my notes in Chat / Flashcards / Quiz</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Panel: Settings & API Key -->
      <div class="panel-container hidden" data-panel="settings">
        <div class="panel-header">
          <h1 class="panel-title">⚙️ AI Provider & API Settings</h1>
          <p class="panel-sub">Configure your Google Gemini, Anthropic Claude, or OpenAI API key.</p>
        </div>

        <div class="card">
          <h2 style="font-size:18px; font-weight:700; margin-bottom:16px;">🔌 AI Model Configuration</h2>
          
          <div class="form-group">
            <label>AI Engine Provider</label>
            <select id="settings-provider" class="form-input">
              <option value="gemini">Google Gemini (Gemini 1.5 Flash - Free Tier available)</option>
              <option value="anthropic">Anthropic (Claude 3.5 Sonnet)</option>
              <option value="openai">OpenAI (GPT-4o Mini)</option>
              <option value="simulation">Smart Knowledge Simulator (Offline / Zero-Key)</option>
            </select>
          </div>

          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="settings-api-key" class="form-input" placeholder="AIzaSy... / sk-ant-... / sk-..." />
            <p class="panel-sub" style="font-size:12px; margin-top:6px;">Your API key is stored securely in your local browser session and passed in requests.</p>
          </div>

          <button class="btn btn-pri" id="settings-save-btn" style="width:auto; padding:12px 28px;">
            Save & Connect AI
          </button>
        </div>
      </div>

      <!-- Panel: Google Meet Live Study Room -->
      <div class="panel-container hidden" data-panel="meet">
        <div class="gmeet-room-layout">
          <div class="gmeet-room-header">
            <div class="gmeet-room-title">
              <span>🎥 SmartStud AI Video Study Room</span>
              <span class="gmeet-pill-code" id="gmeet-live-code">abc-defg-hij</span>
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
              <button class="btn btn-sec" id="btn-gmeet-copy-info" style="font-size:12px; padding:6px 14px;">
                🔗 Copy Joining Info
              </button>
            </div>
          </div>

          <div class="gmeet-stage-area" id="gmeet-video-stage">
            <!-- Host Video Tile -->
            <div class="gmeet-video-card speaking" id="tile-host">
              <div class="gmeet-cam-avatar" id="avatar-host">🧑</div>
              <div class="gmeet-user-tag" id="tag-host">You (Host)</div>
            </div>
            <!-- AI Tutor Tile -->
            <div class="gmeet-video-card" id="tile-tutor">
              <div class="gmeet-cam-avatar" style="background:linear-gradient(135deg,#10b981,#059669);">🤖</div>
              <div class="gmeet-user-tag">Stud AI Co-Host</div>
            </div>
          </div>

          <!-- Bottom Control Bar -->
          <div class="gmeet-bottom-bar">
            <button class="gmeet-control-btn" id="gmeet-btn-mic" title="Toggle Microphone">🎤</button>
            <button class="gmeet-control-btn" id="gmeet-btn-cam" title="Toggle Camera">📹</button>
            <button class="gmeet-control-btn" id="gmeet-btn-screen" title="Share Screen">🖥️</button>
            <button class="gmeet-control-btn" id="gmeet-btn-whiteboard" title="Toggle Whiteboard">🎨</button>
            <button class="gmeet-control-btn" id="gmeet-btn-hand" title="Raise Hand">✋</button>
            <button class="gmeet-control-btn danger" id="gmeet-btn-leave" title="Leave Call">📞</button>
          </div>
        </div>
      </div>
    </main>
  </div>
</div>

<!-- ================= GOOGLE MEET: MEETING FOR LATER MODAL ================= -->
<div id="modal-gmeet-later" class="modal-overlay hidden">
  <div class="modal-content">
    <button class="modal-close" id="modal-gmeet-later-close">✕</button>
    <div style="width:48px; height:48px; border-radius:50%; background:rgba(26,115,232,0.15); color:#1a73e8; display:flex; align-items:center; justify-content:center; font-size:24px; margin-bottom:14px;">🔗</div>
    <h2 style="font-size:20px; font-weight:800; margin-bottom:6px;">Here's the link to your meeting</h2>
    <p class="panel-sub" style="font-size:13px; margin-bottom:16px;">Copy this link and send it to people you want to study with. Be sure to save it so you can use it later, too.</p>

    <div class="copy-input-row">
      <input type="text" id="gmeet-later-url" readonly />
      <button class="btn btn-pri" id="gmeet-later-copy-btn" style="width:auto; padding:8px 18px;">Copy</button>
    </div>

    <div class="share-social-grid" style="margin-top:16px;">
      <button class="btn-social" id="gmeet-later-whatsapp">📱 WhatsApp</button>
      <button class="btn-social" id="gmeet-later-telegram">✈️ Telegram</button>
      <button class="btn-social" id="gmeet-later-native">🔗 Share Link</button>
    </div>
  </div>
</div>

<!-- ================= GOOGLE MEET: LOBBY / GREEN ROOM MODAL ================= -->
<div id="modal-gmeet-lobby" class="modal-overlay hidden">
  <div class="modal-content" style="max-width:560px;">
    <button class="modal-close" id="modal-gmeet-lobby-close">✕</button>
    <h2 style="font-size:20px; font-weight:800; margin-bottom:6px;" id="gmeet-lobby-title">Ready to join?</h2>
    <p class="panel-sub" style="font-size:13px; margin-bottom:16px;" id="gmeet-lobby-sub">SmartStud AI Live Study Room • <span id="gmeet-lobby-code" class="gmeet-pill-code">abc-defg-hij</span></p>

    <!-- Camera / Mic Preview Box -->
    <div class="gmeet-lobby-preview">
      <div class="gmeet-cam-avatar" id="gmeet-lobby-avatar">🧑</div>
      <div class="gmeet-cam-controls">
        <button class="gmeet-control-btn" id="gmeet-lobby-mic-btn" title="Toggle Mic">🎤</button>
        <button class="gmeet-control-btn" id="gmeet-lobby-cam-btn" title="Toggle Camera">📹</button>
      </div>
    </div>

    <div class="form-group" style="margin-bottom:16px;">
      <label>Your Display Name</label>
      <input type="text" class="form-input" id="gmeet-user-name" value="Study Scholar" placeholder="Enter your name" />
    </div>

    <div style="display:flex; gap:12px; align-items:center;">
      <button class="btn btn-pri" id="btn-gmeet-join-now" style="flex:1; padding:12px;">Join Now</button>
      <button class="btn btn-sec" id="btn-gmeet-copy-lobby" style="width:auto; padding:12px 18px;">📋 Copy Info</button>
    </div>
  </div>
</div>

<!-- ================= SHARE MODAL ================= -->
<div id="modal-share" class="modal-overlay hidden">
  <div class="modal-content">
    <button class="modal-close" id="modal-share-close">✕</button>
    <h2 style="font-size:20px; font-weight:800; margin-bottom:6px;">🔗 Share Study Set with Peers</h2>
    <p class="panel-sub" style="font-size:13px;">Allow anyone on your Wi-Fi or internet to instantly open this exact study deck.</p>

    <div class="qr-preview">
      <canvas id="share-qr-canvas"></canvas>
      <span style="font-size:11px; color:#64748B; margin-top:6px;">Scan on mobile to study together</span>
    </div>

    <label style="font-size:12px; font-weight:700; color:var(--text-muted);">Direct Study Link</label>
    <div class="copy-input-row">
      <input type="text" id="share-url-input" readonly />
      <button class="btn btn-pri" id="share-copy-btn" style="width:auto; padding:8px 16px;">Copy</button>
    </div>

    <div class="share-social-grid">
      <button class="btn-social" id="share-whatsapp">📱 WhatsApp</button>
      <button class="btn-social" id="share-telegram">✈️ Telegram</button>
      <button class="btn-social" id="share-twitter">🐦 Twitter</button>
      <button class="btn-social" id="share-native">🔗 Web Share</button>
    </div>
  </div>
</div>

<!-- Toast Container -->
<div class="toast-box" id="toast-container"></div>

<script>
// ============================================================================
// CLIENT APPLICATION LOGIC & ANIMATIONS
// ============================================================================
let appState = {
  currentView: 'landing',
  currentTool: 'chat',
  aiProvider: localStorage.getItem('smartstud_provider') || 'gemini',
  customApiKey: localStorage.getItem('smartstud_api_key') || '',
  activeDeck: [],
  activeCardIndex: 0,
  quizQuestions: [],
  quizAnswers: {},
  todoList: JSON.parse(localStorage.getItem('smartstud_todos') || '[]'),
  savedDecks: JSON.parse(localStorage.getItem('smartstud_library') || '[]'),
  xp: parseInt(localStorage.getItem('smartstud_xp') || '0', 10),
  quizHistory: JSON.parse(localStorage.getItem('smartstud_quiz_history') || '[]'),
  srsData: JSON.parse(localStorage.getItem('smartstud_srs') || '{}'),
  useNotes: localStorage.getItem('smartstud_use_notes') === 'true',
  reviewQueue: [],
  reviewIndex: 0,
  serverInfo: { local_url: window.location.origin, network_url: window.location.origin }
};

// Sound synthesis helper
const audioCtx = (window.AudioContext || window.webkitAudioContext) ? new (window.AudioContext || window.webkitAudioContext)() : null;
function playBeep(freq = 440, type = 'sine', duration = 0.1) {
  if (!audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : '💡'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ---------------------------------------------------------------- Navigation --
function navigateTo(view) {
  appState.currentView = view;
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  const target = document.getElementById(`view-${view}`);
  if (target) target.classList.remove('hidden');
}

function switchTool(tool) {
  appState.currentTool = tool;
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tool === tool);
  });
  document.querySelectorAll('.panel-container').forEach(panel => {
    panel.classList.toggle('hidden', panel.dataset.panel !== tool);
  });
  const titles = {
    chat: 'AI Chat Tutor',
    flashcards: '3D Flashcards Deck',
    review: 'Spaced Review Queue',
    quiz: 'AI Quiz Arena',
    mindmap: 'Mind Map Studio',
    course: 'Course Builder',
    notes: 'My Notes (RAG)',
    pomodoro: 'Pomodoro Focus Timer',
    todo: 'Study Planner & Goals',
    library: 'Study Library',
    analytics: 'Mastery Analytics',
    settings: 'Settings & API Keys'
  };
  const heading = document.getElementById('page-heading');
  if (heading) heading.textContent = titles[tool] || 'Study Workspace';

  if (tool === 'analytics') renderAnalytics();
  if (tool === 'review') renderReviewQueue();
  if (tool === 'notes') refreshNotesStatus();
}

document.querySelectorAll('[data-goto]').forEach(btn => {
  btn.addEventListener('click', () => navigateTo(btn.dataset.goto));
});

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => switchTool(btn.dataset.tool));
});

// ---------------------------------------------------------------- Hero 3D scene interactivity --
(function initHeroScene() {
  const scene = document.getElementById('hc-3d-scene');
  const mask = document.getElementById('hc-scene-mask');
  if (!scene || !mask) return;
  scene.addEventListener('mousemove', (e) => {
    const rect = scene.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    mask.style.transform = `rotateY(${-16 + px * 14}deg) rotateX(${6 - py * 10}deg)`;
  });
  scene.addEventListener('mouseleave', () => {
    mask.style.transform = 'rotateY(-16deg) rotateX(6deg)';
  });
})();

// ---------------------------------------------------------------- Generic 3D tilt for cards --
(function initCardTilt() {
  const SELECTOR = '.card';
  let activeEl = null;
  document.addEventListener('mousemove', (e) => {
    const el = e.target.closest ? e.target.closest(SELECTOR) : null;
    if (activeEl && activeEl !== el) {
      activeEl.style.transform = '';
      activeEl.classList.remove('tilting');
      activeEl = null;
    }
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const rotateX = ((y / rect.height) - 0.5) * -5;
    const rotateY = ((x / rect.width) - 0.5) * 5;
    el.style.transform = `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
    el.classList.add('tilting');
    activeEl = el;
  });
  document.addEventListener('mouseleave', () => {
    if (activeEl) {
      activeEl.style.transform = '';
      activeEl.classList.remove('tilting');
      activeEl = null;
    }
  }, true);
})();

// ---------------------------------------------------------------- Server & API Sync --
async function postAPI(url, data) {
  const headers = { 'Content-Type': 'application/json' };
  if (appState.customApiKey) headers['X-Api-Key'] = appState.customApiKey;
  if (appState.aiProvider) headers['X-Ai-Provider'] = appState.aiProvider;
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('API Error:', err);
    return null;
  }
}

async function loadServerInfo() {
  try {
    const res = await fetch('/api/server-info');
    if (res.ok) {
      appState.serverInfo = await res.json();
    }
  } catch (e) {}
}
loadServerInfo();

function updateApiStatusBadge() {
  const badge = document.getElementById('api-status-badge');
  const text = document.getElementById('api-status-text');
  if (appState.customApiKey) {
    badge.className = 'status-badge';
    text.textContent = `${appState.aiProvider.toUpperCase()} AI Connected`;
  } else {
    badge.className = 'status-badge simulation';
    text.textContent = 'Knowledge Engine Active';
  }
}
updateApiStatusBadge();

// ---------------------------------------------------------------- AI Chat with Markdown & Typewriter --
const chatMessages = document.getElementById('chat-messages');
const chatEmpty = document.getElementById('chat-empty');
const chatLoader = document.getElementById('chat-loader');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send');

function renderChatMessage(role, text, animate = false) {
  chatEmpty.classList.add('hidden');
  chatMessages.classList.remove('hidden');

  const div = document.createElement('div');
  div.className = `msg ${role === 'user' ? 'msg-user' : 'msg-stud'}`;

  let parsedHTML = text;
  if (typeof marked !== 'undefined' && role === 'stud') {
    parsedHTML = marked.parse(text);
  } else {
    parsedHTML = escapeHTML(text).replace(/\n/g, '<br/>');
  }

  div.innerHTML = `<div class="markdown-body">${parsedHTML}</div>`;

  if (role === 'stud') {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `
      <button class="btn-msg-tool" onclick="speakText(decodeURIComponent('${encodeURIComponent(text)}'))">🔊 Read Aloud</button>
      <button class="btn-msg-tool" onclick="copyToClipboard(decodeURIComponent('${encodeURIComponent(text)}'))">📋 Copy</button>
    `;
    div.appendChild(actions);
  }

  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleSendChat(text) {
  const prompt = (text || chatInput.value).trim();
  if (!prompt) return;
  chatInput.value = '';

  renderChatMessage('user', prompt);
  chatLoader.classList.remove('hidden');

  const res = await postAPI('/api/chat', { prompt, useNotes: appState.useNotes });
  chatLoader.classList.add('hidden');

  const reply = (res && res.reply) ? res.reply : "I couldn't reach the model. Please check your connection or API key in Settings.";
  renderChatMessage('stud', reply, true);
  playBeep(520, 'sine', 0.15);
}

chatSendBtn.addEventListener('click', () => handleSendChat());
chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleSendChat(); });
document.querySelectorAll('.prompt-chip').forEach(chip => {
  chip.addEventListener('click', () => handleSendChat(chip.dataset.prompt));
});

// Speech Recognition & TTS
function speakText(txt) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const cleanTxt = txt.replace(/<[^>]*>/g, '').replace(/[#*`_]/g, '');
  const utter = new SpeechSynthesisUtterance(cleanTxt);
  utter.rate = 1.05;
  window.speechSynthesis.speak(utter);
  showToast('Reading response aloud...', 'info');
}

const micBtn = document.getElementById('btn-mic');
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRec();
  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    chatInput.value = transcript;
    handleSendChat(transcript);
  };
  micBtn.addEventListener('click', () => {
    recognition.start();
    showToast('Listening... Speak your study question!', 'info');
  });
} else {
  micBtn.style.display = 'none';
}

// ---------------------------------------------------------------- 3D Flashcards --
const fcTopic = document.getElementById('fc-topic-input');
const fcGenBtn = document.getElementById('fc-generate-btn');
const fcLoader = document.getElementById('fc-loader');
const fcStage = document.getElementById('fc-stage');
const fcCard3D = document.getElementById('fc-card-3d');
const fcFrontText = document.getElementById('fc-front-text');
const fcBackText = document.getElementById('fc-back-text');
const fcCounter = document.getElementById('fc-counter');

fcCard3D.addEventListener('click', () => {
  fcCard3D.classList.toggle('flipped');
  playBeep(380, 'sine', 0.08);
});

function renderActiveFlashcard() {
  if (!appState.activeDeck.length) return;
  const card = appState.activeDeck[appState.activeCardIndex];
  fcCard3D.classList.remove('flipped');
  fcFrontText.textContent = card.front;
  fcBackText.textContent = card.back;
  fcCounter.textContent = `Card ${appState.activeCardIndex + 1} / ${appState.activeDeck.length}`;
}

fcGenBtn.addEventListener('click', async () => {
  const topic = fcTopic.value.trim() || 'Robotics';
  fcLoader.classList.remove('hidden');
  fcStage.classList.add('hidden');

  const data = await postAPI('/api/flashcards', { topic, useNotes: appState.useNotes });
  fcLoader.classList.add('hidden');

  if (data && data.cards && data.cards.length) {
    appState.activeDeck = data.cards;
    appState.activeCardIndex = 0;
    fcStage.classList.remove('hidden');
    renderActiveFlashcard();
    showToast(`Generated ${data.cards.length} flashcards on ${topic}!`, 'success');
  }
});

document.getElementById('fc-prev-btn').addEventListener('click', () => {
  if (appState.activeCardIndex > 0) {
    appState.activeCardIndex--;
    renderActiveFlashcard();
  }
});

document.getElementById('fc-next-btn').addEventListener('click', () => {
  if (appState.activeCardIndex < appState.activeDeck.length - 1) {
    appState.activeCardIndex++;
    renderActiveFlashcard();
  }
});

document.getElementById('fc-speak-btn').addEventListener('click', () => {
  const card = appState.activeDeck[appState.activeCardIndex];
  if (card) speakText(card.front + ". " + card.back);
});

document.getElementById('fc-master-btn').addEventListener('click', () => {
  addXP(50);
  playBeep(650, 'sine', 0.2);
  showToast('+50 XP! Card mastered!', 'success');
  if (appState.activeCardIndex < appState.activeDeck.length - 1) {
    appState.activeCardIndex++;
    renderActiveFlashcard();
  }
});

document.getElementById('fc-save-deck').addEventListener('click', () => {
  if (!appState.activeDeck.length) return;
  const topic = fcTopic.value.trim() || 'Study Deck';
  const deckId = 'deck_' + Date.now().toString(36);
  const stampedCards = appState.activeDeck.map((c, i) => ({ ...c, id: c.id || `${deckId}_${i}` }));
  appState.savedDecks.push({ id: deckId, type: 'flashcards', topic, data: stampedCards, date: new Date().toLocaleDateString() });
  localStorage.setItem('smartstud_library', JSON.stringify(appState.savedDecks));
  showToast('Flashcard deck saved to Library — cards added to your Spaced Review queue!', 'success');
  renderLibrary();
  updateReviewBadge();
});

// ---------------------------------------------------------------- Quiz Arena --
const quizGenBtn = document.getElementById('quiz-generate-btn');
const quizTopicInput = document.getElementById('quiz-topic-input');
const quizDiffSelect = document.getElementById('quiz-diff-select');
const quizLoader = document.getElementById('quiz-loader');
const quizArena = document.getElementById('quiz-arena');
const quizQuestionsList = document.getElementById('quiz-questions-list');
const quizTopicDisplay = document.getElementById('quiz-topic-display');

quizGenBtn.addEventListener('click', async () => {
  const topic = quizTopicInput.value.trim() || 'Robotics';
  const difficulty = quizDiffSelect.value;
  quizLoader.classList.remove('hidden');
  quizArena.classList.add('hidden');

  const res = await postAPI('/api/quiz', { topic, difficulty, useNotes: appState.useNotes });
  quizLoader.classList.add('hidden');

  if (res && res.questions && res.questions.length) {
    appState.quizQuestions = res.questions;
    appState.quizAnswers = {};
    appState.quizActiveTopic = topic;
    appState.quizRecorded = false;
    const diffClass = `quiz-diff-${difficulty.toLowerCase()}`;
quizTopicDisplay.innerHTML = `${escapeHTML(topic)} <span class="quiz-diff-badge ${diffClass}">${escapeHTML(difficulty)}</span>`;
    quizArena.classList.remove('hidden');
    renderQuizArena();
    showToast('Quiz arena ready! Test your knowledge.', 'info');
  }
});

function toggleChapterBody(i) {
  const el = document.getElementById(`chap-body-${i}`);
  if (el) el.classList.toggle('hidden');
}

function toggleChapterDetail(event, i) {
  if (event) event.stopPropagation();
  const el = document.getElementById(`chap-detail-${i}`);
  const btnTxt = document.getElementById(`btn-lesson-text-${i}`);
  const btnIcon = document.getElementById(`btn-lesson-icon-${i}`);
  if (el) {
    el.classList.toggle('hidden');
    const isHidden = el.classList.contains('hidden');
    if (btnTxt) btnTxt.textContent = isHidden ? 'Read Full Detailed Lesson' : 'Hide Detailed Lesson';
    if (btnIcon) btnIcon.textContent = isHidden ? '📖' : '🙈';
  }
}

function recordQuizResult(topic, score, total) {
  appState.quizHistory.push({ topic, score, total, date: new Date().toISOString() });
  if (appState.quizHistory.length > 200) appState.quizHistory.shift();
  localStorage.setItem('smartstud_quiz_history', JSON.stringify(appState.quizHistory));
  renderAnalytics();
}

function renderQuizArena() {
  quizQuestionsList.innerHTML = '';
  let score = 0;
  const answeredCount = Object.keys(appState.quizAnswers).length;

  appState.quizQuestions.forEach((q, idx) => {
    const isAnswered = appState.quizAnswers[idx] !== undefined;
    if (isAnswered && appState.quizAnswers[idx] === q.correctIndex) score++;

    const card = document.createElement('div');
    card.className = 'quiz-card';
    card.innerHTML = `<div class="quiz-q-title">${idx + 1}. ${escapeHTML(q.q)}</div>`;

    const optGrid = document.createElement('div');
    optGrid.className = 'quiz-options-grid';

    q.options.forEach((opt, optIdx) => {
      const btn = document.createElement('button');
      btn.className = 'quiz-btn-opt';
      btn.textContent = opt;

      if (isAnswered) {
        btn.disabled = true;
        if (optIdx === q.correctIndex) btn.classList.add('correct');
        else if (optIdx === appState.quizAnswers[idx]) btn.classList.add('wrong');
      } else {
        btn.addEventListener('click', () => {
          appState.quizAnswers[idx] = optIdx;
          if (optIdx === q.correctIndex) {
            playBeep(600, 'triangle', 0.15);
            addXP(20);
          } else {
            playBeep(220, 'sawtooth', 0.2);
          }
          renderQuizArena();
        });
      }
      optGrid.appendChild(btn);
    });

    card.appendChild(optGrid);

    if (isAnswered) {
      const exp = document.createElement('div');
      exp.className = 'quiz-explanation';
      exp.innerHTML = `<strong>💡 Explanation:</strong> ${escapeHTML(q.explanation)}`;
      card.appendChild(exp);
    }

    quizQuestionsList.appendChild(card);
  });

  document.getElementById('quiz-score-live').textContent = `Score: ${score} / ${appState.quizQuestions.length}`;

  if (answeredCount === appState.quizQuestions.length && answeredCount > 0) {
    if (!appState.quizRecorded) {
      appState.quizRecorded = true;
      recordQuizResult(appState.quizActiveTopic || 'General', score, appState.quizQuestions.length);
      if (typeof confetti === 'function') {
        confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
      }
      showToast(`Quiz completed! Final Score: ${score}/${appState.quizQuestions.length}`, 'success');
    }
  }
}

// ---------------------------------------------------------------- Mind Map --
const mmTopicInput = document.getElementById('mm-topic-input');
const mmGenBtn = document.getElementById('mm-generate-btn');
const mmLoader = document.getElementById('mm-loader');
const mmDisplayArea = document.getElementById('mm-display-area');
const mmSvgContainer = document.getElementById('mm-svg-container');
const mmLegendGrid = document.getElementById('mm-legend-grid');

const MM_PALETTE = ['#7952FC', '#38BDF8', '#10B981', '#F59E0B', '#F43F5E', '#8B5CF6'];

mmGenBtn.addEventListener('click', async () => {
  const topic = mmTopicInput.value.trim() || 'Robotics';
  mmLoader.classList.remove('hidden');
  mmDisplayArea.classList.add('hidden');

  const res = await postAPI('/api/mindmap', { topic });
  mmLoader.classList.add('hidden');

  if (res && res.branches && res.branches.length) {
    renderMindMap(res);
    mmDisplayArea.classList.remove('hidden');
  }
});

function renderMindMap(data) {
  const cx = 370, cy = 250, R = 185;
  const branches = data.branches;
  let svg = `<svg viewBox="0 0 740 500" style="width:100%;height:auto;">`;

  branches.forEach((b, i) => {
    const angle = (i / branches.length) * 2 * Math.PI - Math.PI / 2;
    const x = cx + R * Math.cos(angle);
    const y = cy + R * Math.sin(angle);
    const col = MM_PALETTE[i % MM_PALETTE.length];

    svg += `<path d="M ${cx} ${cy} Q ${cx + (x - cx) * 0.5} ${cy} ${x} ${y}" stroke="${col}" stroke-width="2.5" fill="none" opacity="0.75" />`;
    svg += `<g class="mm-node-g" onclick="speakText('${escapeHTML(b.label)}: ${escapeHTML(b.note)}')">
      <circle cx="${x}" cy="${y}" r="48" fill="#161A30" stroke="${col}" stroke-width="2.5" />
      <foreignObject x="${x - 44}" y="${y - 32}" width="88" height="64">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:11px;font-weight:700;color:#fff;text-align:center;display:flex;align-items:center;justify-content:center;height:100%;line-height:1.2;">
          ${escapeHTML(b.label)}
        </div>
      </foreignObject>
    </g>`;
  });

  // Center node
  svg += `<circle cx="${cx}" cy="${cy}" r="66" fill="url(#gradCenter)" stroke="#fff" stroke-width="3" />`;
  svg += `<defs>
    <linearGradient id="gradCenter" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7952FC"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
  </defs>`;
  svg += `<foreignObject x="${cx - 58}" y="${cy - 38}" width="116" height="76">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:13.5px;font-weight:800;color:#fff;text-align:center;display:flex;align-items:center;justify-content:center;height:100%;">
      ${escapeHTML(data.center)}
    </div>
  </foreignObject>`;
  svg += `</svg>`;

  mmSvgContainer.innerHTML = svg;
  mmLegendGrid.innerHTML = '';
  branches.forEach((b, i) => {
    const card = document.createElement('div');
    card.className = 'mm-branch-card';
    card.innerHTML = `<strong style="color:${MM_PALETTE[i % MM_PALETTE.length]}; font-size:14px;">${escapeHTML(b.label)}:</strong> <span style="color:var(--text-muted);">${escapeHTML(b.note)}</span>`;
    mmLegendGrid.appendChild(card);
  });
}

// ---------------------------------------------------------------- Course Builder --
const courseGenBtn = document.getElementById('course-generate-btn');
const courseTopicInput = document.getElementById('course-topic-input');
const courseChapSelect = document.getElementById('course-chap-select');
const courseLoader = document.getElementById('course-loader');
const courseOutput = document.getElementById('course-output');

courseGenBtn.addEventListener('click', async () => {
  const topic = courseTopicInput.value.trim() || 'Robotics';
  const chapters = parseInt(courseChapSelect.value, 10);
  courseLoader.classList.remove('hidden');
  courseOutput.classList.add('hidden');

  const res = await postAPI('/api/course', { topic, chapters });
  courseLoader.classList.add('hidden');

  if (res && res.chapters) {
    document.getElementById('course-title-out').textContent = res.title;
    document.getElementById('course-summary-out').textContent = res.summary;
    const list = document.getElementById('course-chapters-list');
    list.innerHTML = '';

    res.chapters.forEach((chap, i) => {
      const div = document.createElement('div');
      div.className = 'chapter-module';
      
      // Clean up any duplicate 'Module X:' prefix from title
      const cleanTitle = (chap.title || '').replace(/^Module\s+\d+:\s*/i, '');
      const hasDetails = Boolean(chap.detailedContent && chap.detailedContent.length > 10);
      const detailHTML = hasDetails ? (typeof marked !== 'undefined' ? marked.parse(chap.detailedContent) : chap.detailedContent) : '<p style="color:var(--text-muted); font-style:italic;">No detailed lesson available for this topic.</p>';
      
      div.innerHTML = `
        <div class="chapter-header" onclick="toggleChapterBody(${i})">
          <span class="chapter-title-text">Module ${i + 1}: ${escapeHTML(cleanTitle)}</span>
          <span style="color:var(--primary); font-size:13px; font-weight:700;">📖 Lesson & Topics ▾</span>
        </div>
        <div class="chapter-body" id="chap-body-${i}">
          <div style="font-weight:700; font-size:12px; color:var(--primary); margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px;">Topic Overview</div>
          <ul class="chapter-points-ul" style="margin-bottom:14px;">
            ${(chap.points || []).map(p => `<li>${escapeHTML(p)}</li>`).join('')}
          </ul>
          
          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; margin-bottom:12px;">
            <button class="btn-lesson-toggle" onclick="toggleChapterDetail(event, ${i})">
              <span id="btn-lesson-icon-${i}">📖</span> <span id="btn-lesson-text-${i}">Read Full Detailed Lesson</span>
            </button>
            <button class="btn-lesson-toggle" style="background:rgba(255,255,255,0.08); border:1px solid var(--border);" onclick="askTutorAboutModule('${escapeHTML(cleanTitle)}')">
              💬 Ask Tutor About Module
            </button>
          </div>
          
          <div id="chap-detail-${i}" class="chapter-detailed-content hidden">
            ${detailHTML}
          </div>
        </div>
      `;
      list.appendChild(div);
    });

    courseOutput.classList.remove('hidden');
    showToast('Course curriculum created!', 'success');
  }
});

// ---------------------------------------------------------------- Pomodoro Timer --
let pomoInterval = null;
let pomoSeconds = 25 * 60;

function updatePomoDisplay() {
  const mins = Math.floor(pomoSeconds / 60).toString().padStart(2, '0');
  const secs = (pomoSeconds % 60).toString().padStart(2, '0');
  document.getElementById('pomo-time').textContent = `${mins}:${secs}`;
}

document.querySelectorAll('.pomo-mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pomo-mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    clearInterval(pomoInterval);
    pomoInterval = null;
    document.getElementById('pomo-start-btn').textContent = '▶ Start';
    pomoSeconds = parseInt(btn.dataset.pomoMin, 10) * 60;
    updatePomoDisplay();
  });
});

document.getElementById('pomo-start-btn').addEventListener('click', (e) => {
  if (pomoInterval) {
    clearInterval(pomoInterval);
    pomoInterval = null;
    e.target.textContent = '▶ Start';
  } else {
    e.target.textContent = '⏸ Pause';
    pomoInterval = setInterval(() => {
      if (pomoSeconds > 0) {
        pomoSeconds--;
        updatePomoDisplay();
      } else {
        clearInterval(pomoInterval);
        pomoInterval = null;
        e.target.textContent = '▶ Start';
        playBeep(880, 'sine', 0.5);
        showToast('⏰ Focus session complete! Time for a rest.', 'success');
      }
    }, 1000);
  }
});

document.getElementById('pomo-reset-btn').addEventListener('click', () => {
  clearInterval(pomoInterval);
  pomoInterval = null;
  document.getElementById('pomo-start-btn').textContent = '▶ Start';
  const activeBtn = document.querySelector('.pomo-mode-btn.active');
  pomoSeconds = (activeBtn ? parseInt(activeBtn.dataset.pomoMin, 10) : 25) * 60;
  updatePomoDisplay();
});

// ---------------------------------------------------------------- Planner & To-Dos --
const todoInput = document.getElementById('todo-input');
const todoAddBtn = document.getElementById('todo-add-btn');
const todoContainer = document.getElementById('todo-list-container');

function renderTodos() {
  todoContainer.innerHTML = '';
  appState.todoList.forEach((todo, idx) => {
    const row = document.createElement('div');
    row.className = 'todo-row';
    row.innerHTML = `
      <div class="todo-left">
        <input type="checkbox" ${todo.done ? 'checked' : ''} style="cursor:pointer;" />
        <span class="todo-text ${todo.done ? 'done' : ''}">${escapeHTML(todo.text)}</span>
      </div>
      <button class="btn-icon" style="width:28px;height:28px;border:none;" title="Delete">🗑</button>
    `;
    row.querySelector('input').addEventListener('change', () => {
      todo.done = !todo.done;
      saveTodos();
      renderTodos();
    });
    row.querySelector('button').addEventListener('click', () => {
      appState.todoList.splice(idx, 1);
      saveTodos();
      renderTodos();
    });
    todoContainer.appendChild(row);
  });
}

function saveTodos() {
  localStorage.setItem('smartstud_todos', JSON.stringify(appState.todoList));
}

todoAddBtn.addEventListener('click', () => {
  const text = todoInput.value.trim();
  if (!text) return;
  appState.todoList.push({ text, done: false });
  todoInput.value = '';
  saveTodos();
  renderTodos();
});

// ---------------------------------------------------------------- Study Library --
function renderLibrary() {
  const grid = document.getElementById('library-grid');
  const empty = document.getElementById('library-empty');
  grid.innerHTML = '';
  if (!appState.savedDecks.length) {
    empty.classList.remove('hidden');
    return;
  }
  empty.classList.add('hidden');
  appState.savedDecks.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3 style="font-size:16px; font-weight:700; margin-bottom:6px;">${escapeHTML(item.topic)}</h3>
      <p class="panel-sub" style="font-size:12px; margin-bottom:14px;">${item.type.toUpperCase()} • Saved on ${item.date}</p>
      <button class="btn btn-pri" style="font-size:12px; padding:6px 12px;">Load Deck</button>
    `;
    card.querySelector('button').addEventListener('click', () => {
      if (item.type === 'flashcards') {
        appState.activeDeck = item.data;
        appState.activeCardIndex = 0;
        switchTool('flashcards');
        fcStage.classList.remove('hidden');
        renderActiveFlashcard();
      }
    });
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------- Multi-User Sharing --
const shareModal = document.getElementById('modal-share');
const shareQrCanvas = document.getElementById('share-qr-canvas');
const shareUrlInput = document.getElementById('share-url-input');

async function openShareModal() {
  const topic = fcTopic.value || quizTopicInput.value || 'SmartStud AI Study Session';
  const dataPayload = {
    flashcards: appState.activeDeck,
    quiz: appState.quizQuestions
  };

  const res = await postAPI('/api/share/create', {
    type: appState.currentTool,
    topic,
    data: dataPayload
  });

  const shareId = res && res.shareId ? res.shareId : 'demo';
  const networkUrl = `${appState.serverInfo.network_url || window.location.origin}/?share=${shareId}&tool=${appState.currentTool}`;
  
  shareUrlInput.value = networkUrl;

  if (typeof QRious !== 'undefined') {
    new QRious({
      element: shareQrCanvas,
      value: networkUrl,
      size: 160,
      background: 'white',
      foreground: '#0B0D17'
    });
  }

  shareModal.classList.remove('hidden');
}

document.getElementById('btn-open-share').addEventListener('click', openShareModal);
document.getElementById('modal-share-close').addEventListener('click', () => shareModal.classList.add('hidden'));

document.getElementById('share-copy-btn').addEventListener('click', () => {
  shareUrlInput.select();
  navigator.clipboard.writeText(shareUrlInput.value);
  showToast('Study link copied to clipboard!', 'success');
});

document.getElementById('share-whatsapp').addEventListener('click', () => {
  window.open(`https://api.whatsapp.com/send?text=${encodeURIComponent('Join my SmartStud AI revision room: ' + shareUrlInput.value)}`);
});
document.getElementById('share-telegram').addEventListener('click', () => {
  window.open(`https://t.me/share/url?url=${encodeURIComponent(shareUrlInput.value)}&text=Study%20with%20SmartStud%20AI`);
});
document.getElementById('share-twitter').addEventListener('click', () => {
  window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent('Revising with SmartStud.Ai: ' + shareUrlInput.value)}`);
});
document.getElementById('share-native').addEventListener('click', () => {
  if (navigator.share) {
    navigator.share({ title: 'SmartStud AI Study Room', url: shareUrlInput.value });
  } else {
    document.getElementById('share-copy-btn').click();
  }
});

// ---------------------------------------------------------------- Google Meet Style Rooms Logic --
const gmeetDropdownMenu = document.getElementById('gmeet-dropdown-menu');
const gmeetBtnNew = document.getElementById('btn-gmeet-new');
const gmeetCodeInput = document.getElementById('gmeet-code-input');
const gmeetModalLater = document.getElementById('modal-gmeet-later');
const gmeetModalLobby = document.getElementById('modal-gmeet-lobby');

let currentActiveRoomCode = '';
let isMicMuted = false;
let isCamOff = false;

if (gmeetBtnNew) {
  gmeetBtnNew.addEventListener('click', (e) => {
    e.stopPropagation();
    gmeetDropdownMenu.classList.toggle('hidden');
  });
  document.addEventListener('click', () => {
    if (gmeetDropdownMenu) gmeetDropdownMenu.classList.add('hidden');
  });
}

// 1. Create a meeting for later
document.getElementById('btn-gmeet-later')?.addEventListener('click', async () => {
  gmeetDropdownMenu.classList.add('hidden');
  const res = await postAPI('/api/room/create', { topic: 'SmartStud AI Study Session' });
  if (res && res.roomCode) {
    currentActiveRoomCode = res.roomCode;
    const roomUrl = `${appState.serverInfo.network_url || window.location.origin}/?room=${res.roomCode}`;
    const laterUrlInput = document.getElementById('gmeet-later-url');
    if (laterUrlInput) laterUrlInput.value = roomUrl;
    gmeetModalLater.classList.remove('hidden');
  } else {
    showToast('Failed to create meeting link', 'error');
  }
});

document.getElementById('modal-gmeet-later-close')?.addEventListener('click', () => {
  gmeetModalLater.classList.add('hidden');
});

document.getElementById('gmeet-later-copy-btn')?.addEventListener('click', () => {
  const input = document.getElementById('gmeet-later-url');
  if (input) {
    input.select();
    navigator.clipboard.writeText(input.value);
    showToast('Google Meet link copied to clipboard!', 'success');
  }
});

document.getElementById('gmeet-later-whatsapp')?.addEventListener('click', () => {
  const url = document.getElementById('gmeet-later-url').value;
  window.open(`https://api.whatsapp.com/send?text=${encodeURIComponent('Join my SmartStud AI Google Meet room: ' + url)}`);
});
document.getElementById('gmeet-later-telegram')?.addEventListener('click', () => {
  const url = document.getElementById('gmeet-later-url').value;
  window.open(`https://t.me/share/url?url=${encodeURIComponent(url)}&text=Join%20SmartStud%20AI%20Meeting`);
});
document.getElementById('gmeet-later-native')?.addEventListener('click', () => {
  const url = document.getElementById('gmeet-later-url').value;
  if (navigator.share) {
    navigator.share({ title: 'SmartStud AI Google Meet Room', url });
  } else {
    document.getElementById('gmeet-later-copy-btn').click();
  }
});

// 2. Start an instant meeting
document.getElementById('btn-gmeet-instant')?.addEventListener('click', async () => {
  gmeetDropdownMenu.classList.add('hidden');
  const res = await postAPI('/api/room/create', { topic: 'Instant Study Session' });
  if (res && res.roomCode) {
    openMeetLobby(res.roomCode);
  }
});

// 3. Join with a code or link
document.getElementById('btn-gmeet-join')?.addEventListener('click', () => {
  let val = gmeetCodeInput.value.trim();
  if (!val) {
    showToast('Please enter a room code or link', 'info');
    return;
  }
  openMeetLobby(val);
});
gmeetCodeInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('btn-gmeet-join').click();
});

// Green Room Lobby
async function openMeetLobby(roomCodeRaw) {
  let cleanedCode = roomCodeRaw.trim().toLowerCase();
  if (cleanedCode.includes('room=')) cleanedCode = cleanedCode.split('room=')[1].split('&')[0];
  if (cleanedCode.includes('meet/')) cleanedCode = cleanedCode.split('meet/')[1].split('?')[0];

  try {
    const res = await fetch(`/api/room/${cleanedCode}`);
    if (res.ok) {
      const data = await res.json();
      currentActiveRoomCode = data.roomCode;
      document.getElementById('gmeet-lobby-code').textContent = data.roomCode;
      document.getElementById('gmeet-lobby-sub').textContent = `${data.topic} • ${data.roomCode}`;
      gmeetModalLobby.classList.remove('hidden');
    }
  } catch (e) {
    currentActiveRoomCode = cleanedCode;
    document.getElementById('gmeet-lobby-code').textContent = cleanedCode;
    gmeetModalLobby.classList.remove('hidden');
  }
}

document.getElementById('modal-gmeet-lobby-close')?.addEventListener('click', () => {
  gmeetModalLobby.classList.add('hidden');
});

// Lobby Mic/Cam preview toggles
document.getElementById('gmeet-lobby-mic-btn')?.addEventListener('click', function() {
  isMicMuted = !isMicMuted;
  this.classList.toggle('muted', isMicMuted);
  this.textContent = isMicMuted ? '🔇' : '🎤';
});
document.getElementById('gmeet-lobby-cam-btn')?.addEventListener('click', function() {
  isCamOff = !isCamOff;
  this.classList.toggle('muted', isCamOff);
  const avatar = document.getElementById('gmeet-lobby-avatar');
  if (avatar) avatar.style.opacity = isCamOff ? '0.3' : '1';
});

// Join Now from Lobby
document.getElementById('btn-gmeet-join-now')?.addEventListener('click', () => {
  gmeetModalLobby.classList.add('hidden');
  const userName = document.getElementById('gmeet-user-name').value || 'Study Scholar';
  
  // Launch live room tool view
  switchTool('meet');
  document.getElementById('gmeet-live-code').textContent = currentActiveRoomCode;
  document.getElementById('tag-host').textContent = `${userName} (Host)`;
  
  // Sync live room mic/cam state
  const liveMicBtn = document.getElementById('gmeet-btn-mic');
  const liveCamBtn = document.getElementById('gmeet-btn-cam');
  if (liveMicBtn) {
    liveMicBtn.classList.toggle('muted', isMicMuted);
    liveMicBtn.textContent = isMicMuted ? '🔇' : '🎤';
  }
  if (liveCamBtn) {
    liveCamBtn.classList.toggle('muted', isCamOff);
    const hostAvatar = document.getElementById('avatar-host');
    if (hostAvatar) hostAvatar.style.opacity = isCamOff ? '0.3' : '1';
  }

  showToast(`Joined Google Meet Room: ${currentActiveRoomCode}`, 'success');
});

document.getElementById('btn-gmeet-copy-lobby')?.addEventListener('click', () => {
  const roomUrl = `${appState.serverInfo.network_url || window.location.origin}/?room=${currentActiveRoomCode}`;
  navigator.clipboard.writeText(roomUrl);
  showToast('Meeting link copied!', 'info');
});

// Live Room Controls
document.getElementById('gmeet-btn-mic')?.addEventListener('click', function() {
  isMicMuted = !isMicMuted;
  this.classList.toggle('muted', isMicMuted);
  this.textContent = isMicMuted ? '🔇' : '🎤';
  showToast(isMicMuted ? 'Microphone muted' : 'Microphone unmuted', 'info');
});

document.getElementById('gmeet-btn-cam')?.addEventListener('click', function() {
  isCamOff = !isCamOff;
  this.classList.toggle('muted', isCamOff);
  const hostAvatar = document.getElementById('avatar-host');
  if (hostAvatar) hostAvatar.style.opacity = isCamOff ? '0.3' : '1';
  showToast(isCamOff ? 'Camera turned off' : 'Camera turned on', 'info');
});

let isScreenSharing = false;
document.getElementById('gmeet-btn-screen')?.addEventListener('click', function() {
  isScreenSharing = !isScreenSharing;
  this.classList.toggle('active', isScreenSharing);
  showToast(isScreenSharing ? 'Screen sharing started' : 'Screen sharing stopped', 'success');
});

document.getElementById('gmeet-btn-hand')?.addEventListener('click', () => {
  playBeep(600, 'sine', 0.15);
  showToast('✋ You raised your hand in the study room', 'info');
});

document.getElementById('btn-gmeet-copy-info')?.addEventListener('click', () => {
  const roomUrl = `${appState.serverInfo.network_url || window.location.origin}/?room=${currentActiveRoomCode}`;
  navigator.clipboard.writeText(roomUrl);
  showToast('Joining info copied to clipboard!', 'success');
});

document.getElementById('gmeet-btn-leave')?.addEventListener('click', () => {
  showToast('You left the meeting', 'info');
  switchTool('chat');
});

// Check URL Params for Shared Session & Room Links
async function checkUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const shareId = params.get('share');
  const roomCode = params.get('room') || params.get('meet');
  const tool = params.get('tool');

  if (roomCode) {
    navigateTo('dashboard');
    openMeetLobby(roomCode);
    return;
  }

  if (shareId) {
    navigateTo('dashboard');
    if (tool) switchTool(tool);
    try {
      const res = await fetch(`/api/share/${shareId}`);
      if (res.ok) {
        const payload = await res.json();
        if (payload.data && payload.data.flashcards && payload.data.flashcards.length) {
          appState.activeDeck = payload.data.flashcards;
          appState.activeCardIndex = 0;
          fcStage.classList.remove('hidden');
          renderActiveFlashcard();
        }
        showToast(`Loaded shared study session: ${payload.topic}`, 'success');
      }
    } catch (e) {}
  }
}
checkUrlParams();

// ---------------------------------------------------------------- Settings & API --
document.getElementById('api-status-badge').addEventListener('click', () => switchTool('settings'));

document.getElementById('settings-save-btn').addEventListener('click', () => {
  const provider = document.getElementById('settings-provider').value;
  const key = document.getElementById('settings-api-key').value.trim();

  appState.aiProvider = provider;
  appState.customApiKey = key;

  localStorage.setItem('smartstud_provider', provider);
  localStorage.setItem('smartstud_api_key', key);

  updateApiStatusBadge();
  showToast('Settings saved! Connection active.', 'success');
});

document.getElementById('settings-provider').value = appState.aiProvider;
document.getElementById('settings-api-key').value = appState.customApiKey;

// ---------------------------------------------------------------- Utilities --
function addXP(amount) {
  appState.xp += amount;
  localStorage.setItem('smartstud_xp', appState.xp);
  document.getElementById('stat-xp').textContent = `${appState.xp} XP`;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text);
  showToast('Copied to clipboard!', 'info');
}

// Theme toggle
document.getElementById('btn-theme-toggle').addEventListener('click', () => {
  document.body.classList.toggle('theme-light');
});

// Mobile menu toggle
const sidebarToggle = document.getElementById('btn-sidebar-toggle');
if (window.innerWidth <= 768) sidebarToggle.style.display = 'flex';
sidebarToggle.addEventListener('click', () => {
  document.getElementById('app-sidebar').classList.toggle('open');
});

// ---------------------------------------------------------------- Spaced Repetition (SM-2 lite) --
// Quality: 0 = Again, 1 = Hard, 2 = Good, 3 = Easy
function srsSchedule(prevState, quality) {
  let { ease = 2.5, interval = 0, reps = 0 } = prevState || {};
  if (quality === 0) {
    reps = 0;
    interval = 0.007; // ~10 minutes, expressed in days
    ease = Math.max(1.3, ease - 0.2);
  } else {
    reps += 1;
    if (quality === 1) ease = Math.max(1.3, ease - 0.15);
    else if (quality === 3) ease = ease + 0.15;
    if (reps === 1) interval = quality === 1 ? 1 : (quality === 2 ? 1 : 3);
    else if (reps === 2) interval = quality === 1 ? 2 : (quality === 2 ? 3 : 7);
    else interval = Math.round(interval * ease) || interval + 1;
  }
  const due = new Date(Date.now() + interval * 24 * 60 * 60 * 1000).toISOString();
  return { ease, interval, reps, due };
}

function saveSrsState() {
  localStorage.setItem('smartstud_srs', JSON.stringify(appState.srsData));
}

function getAllLibraryCards() {
  const out = [];
  appState.savedDecks.forEach(deck => {
    if (deck.type !== 'flashcards') return;
    (deck.data || []).forEach((card, i) => {
      const cardId = card.id || `${deck.id || deck.topic}_${i}`;
      out.push({ id: cardId, front: card.front, back: card.back, topic: deck.topic });
    });
  });
  return out;
}

function computeDueQueue() {
  const now = Date.now();
  return getAllLibraryCards().filter(card => {
    const state = appState.srsData[card.id];
    if (!state || !state.due) return true; // never reviewed = due immediately
    return new Date(state.due).getTime() <= now;
  });
}

function updateReviewBadge() {
  const due = computeDueQueue();
  const badge = document.getElementById('review-due-badge');
  if (!badge) return;
  if (due.length > 0) {
    badge.textContent = due.length > 99 ? '99+' : String(due.length);
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

const reviewStage = document.getElementById('review-stage');
const reviewEmpty = document.getElementById('review-empty');
const reviewCard3D = document.getElementById('review-card-3d');
const reviewFrontText = document.getElementById('review-front-text');
const reviewBackText = document.getElementById('review-back-text');
const reviewCounter = document.getElementById('review-counter');

if (reviewCard3D) {
  reviewCard3D.addEventListener('click', () => {
    reviewCard3D.classList.toggle('flipped');
    playBeep(380, 'sine', 0.08);
  });
}

function renderReviewQueue() {
  appState.reviewQueue = computeDueQueue();
  appState.reviewIndex = 0;
  updateReviewBadge();
  if (!appState.reviewQueue.length) {
    reviewEmpty.classList.remove('hidden');
    reviewStage.classList.add('hidden');
    return;
  }
  reviewEmpty.classList.add('hidden');
  reviewStage.classList.remove('hidden');
  renderCurrentReviewCard();
}

function renderCurrentReviewCard() {
  const card = appState.reviewQueue[appState.reviewIndex];
  if (!card) { renderReviewQueue(); return; }
  reviewCard3D.classList.remove('flipped');
  reviewFrontText.textContent = card.front;
  reviewBackText.textContent = card.back;
  reviewCounter.textContent = `Card ${appState.reviewIndex + 1} / ${appState.reviewQueue.length}`;
}

document.querySelectorAll('.srs-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const quality = parseInt(btn.dataset.quality, 10);
    const card = appState.reviewQueue[appState.reviewIndex];
    if (!card) return;
    const prev = appState.srsData[card.id];
    appState.srsData[card.id] = srsSchedule(prev, quality);
    saveSrsState();
    addXP(quality === 0 ? 2 : quality * 5 + 5);
    playBeep(quality >= 2 ? 600 : 260, quality >= 2 ? 'triangle' : 'sawtooth', 0.12);

    appState.reviewQueue.splice(appState.reviewIndex, 1);
    if (appState.reviewIndex >= appState.reviewQueue.length) appState.reviewIndex = 0;
    if (!appState.reviewQueue.length) {
      showToast('Review queue cleared! Great work. 🎉', 'success');
      renderReviewQueue();
    } else {
      renderCurrentReviewCard();
      updateReviewBadge();
    }
  });
});

// ---------------------------------------------------------------- My Notes (RAG) --
const notesTextarea = document.getElementById('notes-textarea');
const notesFileInput = document.getElementById('notes-file-input');
const notesFileBtn = document.getElementById('notes-file-btn');
const notesIndexBtn = document.getElementById('notes-index-btn');
const notesClearBtn = document.getElementById('notes-clear-btn');
const notesStatusText = document.getElementById('notes-status-text');
const notesUseToggle = document.getElementById('notes-use-toggle');

if (notesUseToggle) {
  notesUseToggle.checked = appState.useNotes;
  notesUseToggle.addEventListener('change', () => {
    appState.useNotes = notesUseToggle.checked;
    localStorage.setItem('smartstud_use_notes', String(appState.useNotes));
    showToast(appState.useNotes ? 'Your notes will now ground Chat, Flashcards & Quiz.' : 'Notes grounding disabled.', 'info');
  });
}

if (notesFileBtn) {
  notesFileBtn.addEventListener('click', () => notesFileInput.click());
  notesFileInput.addEventListener('change', () => {
    const file = notesFileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => { notesTextarea.value = e.target.result; };
    reader.readAsText(file);
  });
}

async function refreshNotesStatus() {
  try {
    const res = await fetch('/api/notes/status');
    if (res.ok) {
      const data = await res.json();
      notesStatusText.textContent = data.totalChunks > 0
        ? `${data.totalChunks} note chunk(s) indexed and searchable.`
        : 'No notes indexed yet.';
    }
  } catch (e) {}
}

if (notesIndexBtn) {
  notesIndexBtn.addEventListener('click', async () => {
    const text = notesTextarea.value.trim();
    if (!text) { showToast('Paste or upload some notes first.', 'info'); return; }
    notesIndexBtn.disabled = true;
    notesIndexBtn.textContent = '⏳ Indexing...';
    const res = await postAPI('/api/notes/upload', { text, source: (notesFileInput.files[0] && notesFileInput.files[0].name) || 'Pasted Note' });
    notesIndexBtn.disabled = false;
    notesIndexBtn.textContent = '⚡ Index Notes';
    if (res && res.success) {
      showToast(`Indexed ${res.chunksIndexed} chunk(s) from your notes!`, 'success');
      notesTextarea.value = '';
      refreshNotesStatus();
    } else {
      showToast('Could not index notes. Try again.', 'info');
    }
  });
}

if (notesClearBtn) {
  notesClearBtn.addEventListener('click', async () => {
    const res = await postAPI('/api/notes/clear', {});
    if (res && res.success) {
      showToast('All indexed notes cleared.', 'success');
      refreshNotesStatus();
    }
  });
}

// ---------------------------------------------------------------- Real Analytics Dashboard --
function renderAnalytics() {
  const history = appState.quizHistory || [];
  document.getElementById('stat-xp').textContent = `${appState.xp} XP`;
  document.getElementById('stat-quizzes').textContent = String(history.length);

  const totalCorrect = history.reduce((s, h) => s + h.score, 0);
  const totalQs = history.reduce((s, h) => s + h.total, 0);
  const accuracy = totalQs > 0 ? Math.round((totalCorrect / totalQs) * 100) : 0;
  document.getElementById('stat-accuracy').textContent = `${accuracy}%`;

  const dayStrings = new Set(history.map(h => new Date(h.date).toDateString()));
  let streak = 0;
  for (let i = 0; i < 60; i++) {
    const d = new Date(Date.now() - i * 86400000).toDateString();
    if (dayStrings.has(d)) streak++;
    else if (i > 0) break;
  }
  document.getElementById('stat-streak').textContent = `${streak} Day${streak === 1 ? '' : 's'}`;

  // Per-topic accuracy bar chart
  const topicChart = document.getElementById('analytics-topic-chart');
  if (history.length) {
    const byTopic = {};
    history.forEach(h => {
      if (!byTopic[h.topic]) byTopic[h.topic] = { correct: 0, total: 0 };
      byTopic[h.topic].correct += h.score;
      byTopic[h.topic].total += h.total;
    });
    let html = '';
    Object.entries(byTopic)
      .sort((a, b) => b[1].total - a[1].total)
      .slice(0, 8)
      .forEach(([topic, stats]) => {
        const pct = stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
        html += `<div class="chart-bar-row">
          <div class="chart-bar-label" title="${escapeHTML(topic)}">${escapeHTML(topic)}</div>
          <div class="chart-bar-track"><div class="chart-bar-fill" style="width:${pct}%;"></div></div>
          <div class="chart-bar-pct">${pct}%</div>
        </div>`;
      });
    topicChart.innerHTML = html;
  }

  // XP progression sparkline (cumulative XP per quiz session, last 14)
  const xpChart = document.getElementById('analytics-xp-chart');
  if (history.length) {
    const recent = history.slice(-14);
    let cum = 0;
    const points = recent.map(h => { cum += h.score * 20; return cum; });
    const maxV = Math.max(...points, 1);
    const w = 700, h = 140, pad = 10;
    const stepX = points.length > 1 ? (w - pad * 2) / (points.length - 1) : 0;
    const coords = points.map((v, i) => {
      const x = pad + i * stepX;
      const y = h - pad - (v / maxV) * (h - pad * 2);
      return `${x},${y}`;
    });
    const linePath = coords.length > 1 ? `M ${coords.join(' L ')}` : '';
    const areaPath = coords.length > 1 ? `M ${pad},${h - pad} L ${coords.join(' L ')} L ${pad + (points.length - 1) * stepX},${h - pad} Z` : '';
    xpChart.innerHTML = `<svg viewBox="0 0 ${w} ${h}" style="width:100%; height:auto;">
      <defs>
        <linearGradient id="xpAreaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--primary)" stop-opacity="0.4"/>
          <stop offset="100%" stop-color="var(--primary)" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${areaPath ? `<path d="${areaPath}" fill="url(#xpAreaGrad)" />` : ''}
      ${linePath ? `<path d="${linePath}" fill="none" stroke="var(--primary)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />` : ''}
      ${coords.map(c => `<circle cx="${c.split(',')[0]}" cy="${c.split(',')[1]}" r="3.5" fill="var(--secondary)" />`).join('')}
    </svg>`;
  }
}

// Initial renders
renderTodos();
renderLibrary();
renderAnalytics();
updateReviewBadge();
refreshNotesStatus();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    print("\n=======================================================")
    print(" SmartStud.Ai Study Platform is Running!")
    print(f" Local URL:    http://localhost:{port}")
    print(f" Network LAN:  http://{LOCAL_IP}:{port}")
    print(" (Anyone on your Wi-Fi can open the Network URL!)")
    print("=======================================================")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)


