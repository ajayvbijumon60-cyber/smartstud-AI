"""
SmartStud.Ai -- single-file Flask app.

Everything (HTML, CSS, JS, backend) lives in this ONE file, so there is
no templates/ or static/ folder to lose track of. Just run:

    pip install "flask[async]" anthropic
    set ANTHROPIC_API_KEY=sk-ant-...        (Windows cmd)
    $env:ANTHROPIC_API_KEY="sk-ant-..."     (PowerShell)
    python app.py

Then open http://127.0.0.1:5000
"""

import os
import re
import json
import traceback
import asyncio

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Using the stable latest alias for Claude 3.5 Sonnet
MODEL = "claude-3-5-sonnet-latest"
TUTOR_SYSTEM = (
    "You are Stud, a warm, encouraging AI study tutor. Keep answers clear, "
    "concise, and structured with short paragraphs or bullet points."
)
JSON_SYSTEM = "You output ONLY valid JSON. No markdown fences, no preamble, no commentary."

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client = None
if API_KEY:
    try:
        import anthropic
        # Using Async client for non-blocking I/O performance
        _client = anthropic.AsyncAnthropic(api_key=API_KEY)
    except ImportError:
        _client = None


async def ask_claude(prompt, as_json=False, max_tokens=1200):
    """Mirrors the askClaude() helper from the original React app."""
    if _client is None:
        msg = (
            "Claude isn't configured yet: set the ANTHROPIC_API_KEY environment "
            "variable and make sure the 'anthropic' package is installed, then "
            "restart the app."
        )
        return None if as_json else msg

    system = JSON_SYSTEM if as_json else TUTOR_SYSTEM
    try:
        resp = await _client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text_block = next(
            (block.text for block in resp.content if block.type == "text"), ""
        )
        if as_json:
            cleaned = re.sub(r"```json|```", "", text_block).strip()
            try:
                return json.loads(cleaned)
            except Exception:
                match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", cleaned)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        return None
                return None
        return text_block
    except Exception as exc:
        traceback.print_exc()
        return None if as_json else f"I couldn't reach the model just now ({exc})."


# ---------------------------------------------------------------- pages ----

@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


# --------------------------------------------------------------- api -------

@app.route("/api/chat", methods=["POST"])
async def api_chat():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"reply": "Please enter a question or topic to study."})
    reply = await ask_claude(prompt, as_json=False)
    return jsonify({"reply": reply})


@app.route("/api/flashcards", methods=["POST"])
async def api_flashcards():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    prompt = (
        f'Create 6 flashcards for revising "{topic}". Return ONLY JSON: '
        '{{"cards":[{{"front":"term/question","back":"concise answer"}}]}}'
    )
    result = await ask_claude(prompt, as_json=True)
    cards = (result or {}).get("cards", [])
    return jsonify({"cards": cards})


@app.route("/api/quiz", methods=["POST"])
async def api_quiz():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    difficulty = data.get("difficulty", "beginner")
    prompt = (
        f'Create a 5 question multiple-choice quiz on "{topic}" at {difficulty} '
        'difficulty. Return ONLY JSON: {{"questions":[{{"q":"text",'
        '"options":["a","b","c","d"],"correctIndex":0,"explanation":"short reason"}}]}}'
    )
    result = await ask_claude(prompt, as_json=True, max_tokens=1500)
    questions = (result or {}).get("questions", [])
    return jsonify({"questions": questions})


@app.route("/api/mindmap", methods=["POST"])
async def api_mindmap():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    prompt = (
        f'Create a mind map for "{topic}": a central node and up to 6 branches, '
        "each with a short (max 4 word) label and a one sentence note. "
        'Return ONLY JSON: {{"center":"topic","branches":[{{"label":"short label",'
        '"note":"sentence"}}]}}'
    )
    result = await ask_claude(prompt, as_json=True)
    if result and result.get("branches"):
        return jsonify(result)
    return jsonify(None)


@app.route("/api/course", methods=["POST"])
async def api_course():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    chapters = data.get("chapters", 3)
    prompt = (
        f'Design a beginner-friendly course outline on "{topic}" with exactly '
        f"{chapters} chapters. Return ONLY JSON: "
        '{{"title":"course title","summary":"one sentence","chapters":'
        '[{{"title":"chapter title","points":["point1","point2","point3"]}}]}}'
    )
    result = await ask_claude(prompt, as_json=True, max_tokens=1600)
    if result and result.get("chapters"):
        return jsonify(result)
    return jsonify(None)


# ------------------------------------------------------------ the page -----

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SmartStud.Ai</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Caveat:wght@600;700&display=swap');

:root {
  --purple: #7B6EF6;
  --blue: #4FB6E8;
  --ink: #22232E;
  --muted: #6C6F80;
  --border: #ECE9F7;
  --bg: linear-gradient(135deg, #FFF4EA 0%, #FDEAF1 45%, #EAE7FB 100%);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: 'Poppins', sans-serif;
  color: var(--ink);
  min-height: 100vh;
  background: var(--bg);
  transition: background 0.3s ease, color 0.3s ease;
}

body.dashboard-active { background: #fff; }

/* Dark Mode Variables */
body.dark-mode {
  --bg: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
  --ink: #e0e0e0;
  --muted: #a0a0a0;
  --border: #2a2a40;
  --purple: #9D8DF1;
  --blue: #69C8F5;
  background: var(--bg);
  color: var(--ink);
}

body.dark-mode.dashboard-active { background: #12121c; }

body.dark-mode .card,
body.dark-mode .feature-card,
body.dark-mode .login-card,
body.dark-mode .sidebar,
body.dark-mode .quiz-opt,
body.dark-mode .fc-card,
body.dark-mode .chat-input-row,
body.dark-mode .msg-stud,
body.dark-mode .chapter-card,
body.dark-mode .todo-item {
  background: #1e1e2f !important;
  border-color: var(--border) !important;
  color: var(--ink) !important;
}

body.dark-mode input, body.dark-mode select {
  background: #2a2a40 !important;
  color: var(--ink) !important;
  border-color: var(--border) !important;
}

body.dark-mode .btn-outline {
  background: #2a2a40 !important;
  color: var(--ink) !important;
  border-color: var(--border) !important;
}

.hidden { display: none !important; }

.grad-text {
  background-image: linear-gradient(90deg, var(--purple), var(--blue));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.clickable { cursor: pointer; }
.muted { color: var(--muted); }
.link { color: var(--purple); font-weight: 600; }

button { font-family: 'Poppins', sans-serif; }

.btn {
  border: none;
  border-radius: 10px;
  padding: 13px 26px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s;
}
.btn:active { transform: scale(0.98); }
.btn-primary { background: linear-gradient(90deg, var(--purple), var(--blue)); color: #fff; }
.btn-outline { background: #fff; color: var(--ink); border: 1px solid var(--border); }
.btn-sm { padding: 9px 20px; font-size: 14px; }
.wide { width: 260px; }

input, select {
  font-family: 'Poppins', sans-serif;
  font-size: 14px;
}
input:focus, select:focus { outline: 2px solid var(--purple); outline-offset: 1px; }
button:focus-visible { outline: 2px solid var(--purple); outline-offset: 1px; }

/* Custom Scrollbars */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ---------------- Landing ---------------- */

.landing-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 48px;
}
.brand { display: flex; align-items: center; gap: 10px; }
.logo {
  width: 34px; height: 34px; border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), var(--blue));
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.logo.small { width: 30px; height: 30px; }
.brand-name { font-weight: 700; font-size: 19px; }

.landing-nav { display: flex; align-items: center; gap: 28px; font-size: 14.5px; color: var(--muted); }
.landing-nav .moon { font-size: 18px; cursor: pointer; user-select: none; }

.hero {
  display: flex; align-items: center; justify-content: space-between;
  padding: 60px 48px 90px; flex-wrap: wrap; gap: 40px;
}
.hero-copy { max-width: 520px; }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: #E6F7F5; color: #1D9E75; font-size: 12.5px; font-weight: 600;
  padding: 6px 14px; border-radius: 999px; margin-bottom: 20px;
}
body.dark-mode .pill { background: #1e3a34; color: #42b58e; }
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; }
.hero h1 { font-weight: 700; font-size: 52px; line-height: 1.08; margin: 0 0 20px; }
.hero-sub { color: var(--muted); font-size: 16px; line-height: 1.7; margin-bottom: 30px; }
.hero-actions { display: flex; gap: 14px; }

.hero-orbit { position: relative; width: 320px; height: 320px; flex-shrink: 0; }
.ring { position: absolute; border-radius: 50%; border: 1px solid var(--border); }
.ring-outer { inset: 0; }
.ring-inner { inset: 40px; }
.orbit-center {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 130px; height: 130px; border-radius: 50%; background: #fff;
  box-shadow: 0 20px 50px rgba(123,110,246,0.25);
  display: flex; align-items: center; justify-content: center;
}
body.dark-mode .orbit-center { background: #1e1e2f; }
.orbit-badge {
  position: absolute; width: 40px; height: 40px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 6px 16px rgba(0,0,0,0.06); font-size: 16px;
}
.b1 { top: 10px; left: 250px; background: #E6F7F5; }
.b2 { top: 130px; left: 285px; background: #FDEAF3; }
.b3 { top: 250px; left: 245px; background: #FCEFE1; }
.b4 { top: 250px; left: 40px; background: #EAE9FD; }

.features { padding: 10px 48px 80px; text-align: center; }
.features h2 { font-weight: 700; font-size: 34px; margin: 0 0 10px; }
.features-sub { color: var(--muted); margin-bottom: 44px; }
.features-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px; text-align: left; max-width: 1000px; margin: 0 auto;
}
.feature-card { background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 26px 24px; }
.feature-icon {
  width: 46px; height: 46px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 16px; font-size: 20px;
}
.feature-title { font-weight: 600; font-size: 16px; margin: 0 0 8px; }
.feature-desc { color: var(--muted); font-size: 14px; line-height: 1.6; margin: 0; }

/* ---------------- Get Started ---------------- */

.center-col {
  min-height: 100vh; display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 22px; padding: 40px;
}
.ring-avatar { 
  width: 64px; height: 64px; border-radius: 50%; border: 3px solid var(--purple); 
  display: flex; align-items: center; justify-content: center; font-size: 32px;
  background: #F1EEFE; color: var(--purple);
}
body.dark-mode .ring-avatar { background: #2a2a40; }
.center-col h1 { font-weight: 700; font-size: 30px; margin: 0; }

/* ---------------- Login ---------------- */

.login-wrap { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; }
.login-card {
  width: 420px; background: #fff; border-radius: 20px; padding: 40px 36px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.08);
}
.login-title { text-align: center; font-weight: 700; font-size: 26px; margin: 0 0 6px; }
.login-sub { text-align: center; color: var(--muted); font-size: 14px; margin-bottom: 26px; }
.login-card label { font-size: 13px; font-weight: 600; display: block; margin-top: 4px; }
.login-card input {
  width: 100%; padding: 11px 12px; margin: 6px 0 16px;
  border: 1px solid var(--border); border-radius: 8px;
}
.divider { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; margin: 20px 0; }
.divider::before, .divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.login-footer { text-align: center; font-size: 13px; color: var(--muted); margin-top: 18px; }

/* ---------------- Dashboard ---------------- */

.dash-shell { display: flex; min-height: 100vh; }
.sidebar {
  width: 230px; flex-shrink: 0; background: #fff; border-right: 1px solid var(--border);
  padding: 26px 16px; display: flex; flex-direction: column;
}
.sidebar-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 30px; padding: 0 8px; font-weight: 700; font-size: 16px; }
.nav-list { flex: 1; display: flex; flex-direction: column; gap: 2px; overflow-y: auto; }
.nav-item {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 10px;
  border: none; background: transparent; color: var(--muted); font-weight: 400; font-size: 13.5px;
  cursor: pointer; text-align: left; width: 100%;
}
.nav-item.active { background: #F1EEFE; color: var(--purple); font-weight: 600; }
body.dark-mode .nav-item.active { background: #2a2a40; }
.logout {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 10px;
  border: none; background: transparent; color: #D4534A; font-size: 13.5px; cursor: pointer;
}

.dash-main { flex: 1; padding: 30px 36px; background: var(--bg); overflow-y: auto; }

.card { background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 24px; }
.coming-soon { text-align: center; padding: 60px 24px; }
.cs-title { font-weight: 600; font-size: 16px; margin-bottom: 6px; }
.cs-sub { color: var(--muted); font-size: 14px; }

.topic-form { display: flex; gap: 10px; margin-bottom: 22px; flex-wrap: wrap; align-items: center; }
.topic-form input[type="text"], .topic-form input:not([type]) {
  flex: 1; min-width: 200px; padding: 11px 14px; border: 1px solid var(--border); border-radius: 10px; font-size: 14px;
}
#fc-topic, #quiz-topic, #mm-topic, #course-topic, #todo-text {
  flex: 1; min-width: 200px; padding: 11px 14px; border: 1px solid var(--border); border-radius: 10px; font-size: 14px;
}
#quiz-difficulty { padding: 11px 12px; border: 1px solid var(--border); border-radius: 10px; }
.inline-label { font-size: 13px; color: var(--muted); }
.num-input { width: 60px; padding: 11px 10px; border: 1px solid var(--border); border-radius: 10px; }

.loader { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
.spinner {
  width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--purple);
  border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Chat panel */
.chat-empty { text-align: center; padding-top: 60px; }
.hello { font-family: 'Caveat', cursive; font-weight: 700; font-size: 40px; margin: 0 0 4px;
  background-image: linear-gradient(90deg, var(--purple), var(--blue)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.chat-tagline { font-family: 'Caveat', cursive; font-weight: 600; font-size: 24px; color: var(--purple); margin: 0 0 30px; }
.chat-input-row {
  max-width: 640px; margin: 0 auto; display: flex; gap: 10px;
  background: #fff; border: 1px solid var(--border); border-radius: 999px; padding: 8px;
}
.chat-input-row input {
  flex: 1; border: none; outline: none; padding: 8px 14px; font-size: 14.5px; border-radius: 999px;
  background: transparent; color: var(--ink);
}
.chat-input-row.followup { max-width: none; margin: 0; }
.quick-actions { display: flex; gap: 10px; justify-content: center; margin-top: 22px; flex-wrap: wrap; }
.quick-actions button {
  display: flex; align-items: center; gap: 6px; background: #fff; border: 1px solid var(--border);
  border-radius: 10px; padding: 9px 16px; font-size: 13.5px; color: var(--ink); cursor: pointer;
}
.chat-thread { display: flex; flex-direction: column; height: 100%; }
.chat-messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-bottom: 16px; min-height: 300px; }
.msg { max-width: 78%; border-radius: 14px; padding: 10px 16px; font-size: 14.5px; line-height: 1.6; white-space: pre-wrap; }
.msg-user { align-self: flex-end; background: linear-gradient(90deg, var(--purple), var(--blue)); color: #fff; }
.msg-stud { align-self: flex-start; background: #fff; color: var(--ink); border: 1px solid var(--border); }
.send-round {
  background: linear-gradient(90deg, var(--purple), var(--blue)); border: none; border-radius: 50%;
  width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; cursor: pointer; color: #fff;
}

/* Flashcards */
.fc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.fc-card {
  cursor: pointer; min-height: 130px; border-radius: 14px; padding: 18px; background: #FAF9FE;
  border: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center;
  transition: transform 0.2s;
}
.fc-card:hover { transform: translateY(-2px); }
.fc-card.flipped { background: #F1EEFE; }
body.dark-mode .fc-card.flipped { background: #2a2a40 !important; }
.fc-label { font-size: 11px; font-weight: 600; color: var(--purple); text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px; }
.fc-text { margin: 0; font-size: 14.5px; line-height: 1.5; }

/* Quiz */
.quiz-score { font-size: 13px; color: var(--muted); margin-bottom: 18px; }
.quiz-q { margin-bottom: 24px; }
.quiz-q-text { font-weight: 600; font-size: 15px; margin: 0 0 10px; }
.quiz-options { display: flex; flex-direction: column; gap: 8px; }
.quiz-opt {
  text-align: left; padding: 10px 14px; border-radius: 10px; border: 1px solid var(--border);
  background: #FAF9FE; color: var(--ink); font-size: 14px; cursor: pointer;
  display: flex; justify-content: space-between; align-items: center;
}
.quiz-opt[disabled] { cursor: default; }
.quiz-opt.correct { background: #E9F6E3; border-color: #3B6D11; color: #27500A; }
.quiz-opt.wrong { background: #FDEAEA; border-color: #D4534A; color: #A32D2D; }
.quiz-explain { color: var(--muted); font-size: 13px; margin-top: 8px; }

/* Mind map */
.mm-legend p { font-size: 13px; color: var(--muted); margin: 0; }
.mm-legend { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }

/* Course */
.course-title { font-weight: 700; font-size: 18px; margin: 0 0 4px; }
.course-summary { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
.course-chapters { display: flex; flex-direction: column; gap: 12px; }
.chapter-card { border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; background: #fff; }
.chapter-title { font-weight: 600; font-size: 14.5px; margin: 0 0 8px; color: var(--purple); }
.chapter-points { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13.5px; line-height: 1.7; }

/* Todo */
.muted-text { color: var(--muted); font-size: 14px; }
.todo-list { display: flex; flex-direction: column; gap: 8px; }
.todo-item { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border: 1px solid var(--border); border-radius: 10px; background: #fff; }
.todo-item span { flex: 1; font-size: 14px; }
.todo-item span.done { color: var(--muted); text-decoration: line-through; }
.todo-item .del { cursor: pointer; color: var(--muted); }

</style>
</head>
<body>

<!-- ================= LANDING ================= -->
<div id="view-landing" class="view">
  <header class="landing-header">
    <div class="brand">
      <div class="logo"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2z"/></svg></div>
      <span class="brand-name">SmartStud.Ai</span>
    </div>
    <nav class="landing-nav">
      <span>Features</span>
      <span>How It Works</span>
      <span class="clickable" data-goto="getstarted">Log In</span>
      <span class="moon" title="Toggle Dark Mode">🌙</span>
      <button class="btn btn-primary btn-sm" data-goto="getstarted">Get Started</button>
    </nav>
  </header>

  <section class="hero">
    <div class="hero-copy">
      <div class="pill"><span class="dot"></span>AI-Powered Learning</div>
      <h1>Master Your<br /><span class="grad-text">Studies</span></h1>
      <p class="hero-sub">Experience the universe of knowledge with personalized AI courses, quizzes, and mind maps. Your journey to mastery starts here.</p>
      <div class="hero-actions">
        <button class="btn btn-primary" data-goto="getstarted">Start Learning Now</button>
        <button class="btn btn-outline" data-goto="getstarted">Watch Demo</button>
      </div>
    </div>
    <div class="hero-orbit">
      <div class="ring ring-outer"></div>
      <div class="ring ring-inner"></div>
      <div class="orbit-center"><svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="var(--purple)" stroke-width="1.6"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2z"/></svg></div>
      <div class="orbit-badge b1">✨</div>
      <div class="orbit-badge b2">🗂️</div>
      <div class="orbit-badge b3">📊</div>
      <div class="orbit-badge b4">📈</div>
    </div>
  </section>

  <section class="features">
    <h2><span class="grad-text">Explore Our Features</span></h2>
    <p class="features-sub">Powerful tools designed to supercharge your learning efficiency.</p>
    <div class="features-grid">
      <div class="feature-card"><div class="feature-icon" style="background:#F1EEFE;color:#7B6EF6;">🧠</div><p class="feature-title">Course Generator</p><p class="feature-desc">Generate complete courses on any topic with AI-curated modules and lessons.</p></div>
      <div class="feature-card"><div class="feature-icon" style="background:#E6F7F5;color:#1D9E75;">⚡</div><p class="feature-title">Quiz Generator</p><p class="feature-desc">Instantly create quizzes to test your knowledge and reinforce learning.</p></div>
      <div class="feature-card"><div class="feature-icon" style="background:#FDEAF3;color:#D4537E;">📚</div><p class="feature-title">Flashcards</p><p class="feature-desc">Smart flashcards that use spaced repetition to help you memorize faster.</p></div>
      <div class="feature-card"><div class="feature-icon" style="background:#EAE9FD;color:#534AB7;">🕸️</div><p class="feature-title">Mind Maps</p><p class="feature-desc">Visualize complex topics with AI-generated interactive mind maps.</p></div>
      <div class="feature-card"><div class="feature-icon" style="background:#E9F6E3;color:#3B6D11;">📈</div><p class="feature-title">Performance Tracking</p><p class="feature-desc">Detailed analytics to track your progress and identify weak areas.</p></div>
      <div class="feature-card"><div class="feature-icon" style="background:#FCEFE1;color:#B4881A;">🏆</div><p class="feature-title">Skill Assessment</p><p class="feature-desc">Adaptive assessments to gauge your current level and customize paths.</p></div>
    </div>
  </section>
</div>

<!-- ================= GET STARTED ================= -->
<div id="view-getstarted" class="view hidden">
  <div class="center-col">
    <div class="ring-avatar">👤</div>
    <h1>Get Started</h1>
    <button class="btn btn-primary wide" data-goto="login">Log In</button>
    <button class="btn btn-outline wide" data-goto="dashboard">Sign up free</button>
    <span class="clickable muted" data-goto="dashboard">Try it first</span>
  </div>
</div>

<!-- ================= LOGIN ================= -->
<div id="view-login" class="view hidden">
  <div class="login-wrap">
    <div class="login-card">
      <h1 class="login-title">Log in</h1>
      <p class="login-sub">Access your personalized dashboard</p>
      <button class="btn btn-outline wide" data-goto="dashboard">Continue with Google</button>
      <div class="divider"><span>OR</span></div>
      <label>Email</label>
      <input type="email" placeholder="you@example.com" />
      <label>Password</label>
      <input type="password" placeholder="********" />
      <button class="btn btn-primary wide" data-goto="dashboard">Continue</button>
      <p class="login-footer">Don't have an account? <span class="clickable link" data-goto="dashboard">Sign up</span></p>
    </div>
  </div>
</div>

<!-- ================= DASHBOARD ================= -->
<div id="view-dashboard" class="view hidden">
  <div class="dash-shell">
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div class="logo small"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="#fff" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24A2.5 2.5 0 0 0 14.5 2z"/></svg></div>
        <span>SmartStud.Ai</span>
      </div>
      <div class="nav-list">
        <button class="nav-item active" data-tool="chat">💬 Chat with Agent</button>
        <button class="nav-item" data-tool="companion">🧠 AI Companion</button>
        <button class="nav-item" data-tool="courses">🎓 Courses</button>
        <button class="nav-item" data-tool="aicourses">✨ AI Courses</button>
        <button class="nav-item" data-tool="flashcards">📚 Flashcards</button>
        <button class="nav-item" data-tool="mindmap">🕸️ Mind Map</button>
        <button class="nav-item" data-tool="quizzes">📋 Quizzes</button>
        <button class="nav-item" data-tool="learning">📖 My Learning</button>
        <button class="nav-item" data-tool="library">📗 Library</button>
        <button class="nav-item" data-tool="todo">✅ To-Do List</button>
        <button class="nav-item" data-tool="combine">👥 CombineStudies</button>
        <button class="nav-item" data-tool="performance">📊 Performance</button>
        <button class="nav-item" data-tool="settings">⚙️ Settings</button>
      </div>
      <button class="logout" data-goto="landing">🚪 Log out</button>
    </aside>

    <main class="dash-main">

      <!-- Chat / Companion panel (shared) -->
      <div class="panel" data-panel="chat">
        <div id="chat-empty" class="chat-empty">
          <p class="hello">Hello suri!</p>
          <p class="chat-tagline">Chat with Stud</p>
          <div class="chat-input-row">
            <input id="chat-empty-input" placeholder="Describe what you want to learn..." />
            <button class="btn btn-primary" id="chat-empty-send">Ask AI</button>
          </div>
          <div class="quick-actions">
            <button data-tool-jump="mindmap" data-icon="🕸️">🕸️ Mind Map</button>
            <button data-tool-jump="flashcards" data-icon="📚">📚 Flashcards</button>
            <button data-tool-jump="quizzes" data-icon="⚡">⚡ Quiz Generator</button>
            <button id="chat-clear">🗑️ Clear</button>
          </div>
        </div>
        <div id="chat-thread" class="chat-thread hidden">
          <div id="chat-messages" class="chat-messages"></div>
          <div id="chat-loader" class="loader hidden"><span class="spinner"></span>Stud is thinking...</div>
          <div class="chat-input-row followup">
            <input id="chat-followup-input" placeholder="Ask a follow-up..." />
            <button class="send-round" id="chat-followup-send">➤</button>
          </div>
        </div>
      </div>

      <!-- Flashcards panel -->
      <div class="panel hidden" data-panel="flashcards">
        <div class="card">
          <div class="topic-form">
            <input id="fc-topic" placeholder="Enter a topic..." />
            <button class="btn btn-primary" id="fc-go">Generate</button>
          </div>
          <div id="fc-loader" class="loader hidden"><span class="spinner"></span>Writing flashcards...</div>
          <div id="fc-grid" class="fc-grid"></div>
        </div>
      </div>

      <!-- Quiz panel -->
      <div class="panel hidden" data-panel="quiz">
        <div class="card">
          <div class="topic-form">
            <input id="quiz-topic" placeholder="Enter a topic..." />
            <select id="quiz-difficulty">
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
            <button class="btn btn-primary" id="quiz-go">Start Quiz</button>
          </div>
          <div id="quiz-loader" class="loader hidden"><span class="spinner"></span>Setting the question paper...</div>
          <p id="quiz-score" class="quiz-score hidden"></p>
          <div id="quiz-questions"></div>
        </div>
      </div>

      <!-- Mind map panel -->
      <div class="panel hidden" data-panel="mindmap">
        <div class="card">
          <div class="topic-form">
            <input id="mm-topic" placeholder="Enter a topic..." />
            <button class="btn btn-primary" id="mm-go">Draw It</button>
          </div>
          <div id="mm-loader" class="loader hidden"><span class="spinner"></span>Sketching the branches...</div>
          <div id="mm-svg-wrap"></div>
          <div id="mm-legend" class="mm-legend"></div>
        </div>
      </div>

      <!-- Course panel -->
      <div class="panel hidden" data-panel="course">
        <div class="card">
          <div class="topic-form">
            <input id="course-topic" placeholder="e.g. Data Structures in C" />
            <label class="inline-label">Chapters</label>
            <input id="course-chapters" type="number" min="2" max="8" value="3" class="num-input" />
            <button class="btn btn-primary" id="course-go">Generate Course</button>
          </div>
          <div id="course-loader" class="loader hidden"><span class="spinner"></span>Building your course...</div>
          <div id="course-result"></div>
        </div>
      </div>

      <!-- Todo panel -->
      <div class="panel hidden" data-panel="todo">
        <div class="card">
          <div class="topic-form">
            <input id="todo-text" placeholder="Add a study task..." />
            <button class="btn btn-primary" id="todo-add">+</button>
          </div>
          <p id="todo-empty" class="muted-text">No tasks yet. Add your first one above.</p>
          <div id="todo-list" class="todo-list"></div>
        </div>
      </div>

      <!-- Coming soon panels -->
      <div class="panel hidden" data-panel="coming-courses"><div class="card coming-soon"><p class="cs-title">Courses library</p><p class="cs-sub">This section is on the roadmap.</p></div></div>
      <div class="panel hidden" data-panel="coming-learning"><div class="card coming-soon"><p class="cs-title">My Learning</p><p class="cs-sub">This section is on the roadmap.</p></div></div>
      <div class="panel hidden" data-panel="coming-library"><div class="card coming-soon"><p class="cs-title">Library</p><p class="cs-sub">This section is on the roadmap.</p></div></div>
      <div class="panel hidden" data-panel="coming-combine"><div class="card coming-soon"><p class="cs-title">CombineStudies</p><p class="cs-sub">This section is on the roadmap.</p></div></div>
      <div class="panel hidden" data-panel="coming-performance"><div class="card coming-soon"><p class="cs-title">Performance analytics</p><p class="cs-sub">This section is on the roadmap.</p></div></div>
      <div class="panel hidden" data-panel="coming-settings"><div class="card coming-soon"><p class="cs-title">Settings</p><p class="cs-sub">This section is on the roadmap.</p></div></div>

    </main>
  </div>
</div>

<script>
// ---------------------------------------------------------------- routing --

const VIEWS = ["landing", "getstarted", "login", "dashboard"];

function goto(view) {
  VIEWS.forEach((v) => {
    document.getElementById(`view-${v}`).classList.toggle("hidden", v !== view);
  });
  document.body.classList.toggle("dashboard-active", view === "dashboard");
}

document.querySelectorAll("[data-goto]").forEach((el) => {
  el.addEventListener("click", () => goto(el.dataset.goto));
});

document.querySelector(".moon").addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
});

// ---------------------------------------------------------------- dashboard nav --

const TOOL_TO_PANEL = {
  chat: "chat",
  companion: "chat",
  courses: "coming-courses",
  aicourses: "course",
  flashcards: "flashcards",
  mindmap: "mindmap",
  learning: "coming-learning",
  library: "coming-library",
  quizzes: "quiz",
  todo: "todo",
  combine: "coming-combine",
  performance: "coming-performance",
  settings: "coming-settings",
};

function showTool(tool) {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tool === tool);
  });
  const panelName = TOOL_TO_PANEL[tool] || "chat";
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.panel !== panelName);
  });
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => showTool(btn.dataset.tool));
});

document.querySelectorAll("[data-tool-jump]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tool = btn.dataset.toolJump;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.tool === tool));
    showTool(tool);
  });
});

// ---------------------------------------------------------------- helpers --

async function postJSON(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

// ---------------------------------------------------------------- chat panel --

let chatMessages = [];

function renderChat() {
  const empty = document.getElementById("chat-empty");
  const thread = document.getElementById("chat-thread");
  if (chatMessages.length === 0) {
    empty.classList.remove("hidden");
    thread.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  thread.classList.remove("hidden");
  const box = document.getElementById("chat-messages");
  box.innerHTML = "";
  chatMessages.forEach((m) => {
    const div = document.createElement("div");
    div.className = "msg " + (m.role === "user" ? "msg-user" : "msg-stud");
    div.textContent = m.text;
    box.appendChild(div);
  });
  box.scrollTop = box.scrollHeight;
}

async function sendChat(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;
  chatMessages.push({ role: "user", text: trimmed });
  renderChat();
  document.getElementById("chat-loader").classList.remove("hidden");
  const data = await postJSON("/api/chat", { prompt: trimmed });
  document.getElementById("chat-loader").classList.add("hidden");
  chatMessages.push({ role: "stud", text: data && data.reply ? data.reply : "I couldn't reach the model just now. Please try again." });
  renderChat();
}

document.getElementById("chat-empty-send").addEventListener("click", () => {
  const input = document.getElementById("chat-empty-input");
  const text = input.value;
  input.value = "";
  sendChat(text);
});
document.getElementById("chat-empty-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("chat-empty-send").click();
});
document.getElementById("chat-followup-send").addEventListener("click", () => {
  const input = document.getElementById("chat-followup-input");
  const text = input.value;
  input.value = "";
  sendChat(text);
});
document.getElementById("chat-followup-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("chat-followup-send").click();
});
document.getElementById("chat-clear").addEventListener("click", () => {
  chatMessages = [];
  renderChat();
});

// ---------------------------------------------------------------- flashcards --

document.getElementById("fc-go").addEventListener("click", generateFlashcards);
document.getElementById("fc-topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter") generateFlashcards();
});

async function generateFlashcards() {
  const topic = document.getElementById("fc-topic").value.trim();
  if (!topic) return;
  const grid = document.getElementById("fc-grid");
  const loader = document.getElementById("fc-loader");
  grid.innerHTML = "";
  loader.classList.remove("hidden");
  const data = await postJSON("/api/flashcards", { topic });
  loader.classList.add("hidden");
  const cards = (data && data.cards) || [];
  cards.forEach((c) => {
    const div = document.createElement("div");
    div.className = "fc-card";
    div.innerHTML = `<p class="fc-label">Term</p><p class="fc-text">${escapeHTML(c.front)}</p>`;
    let flipped = false;
    div.addEventListener("click", () => {
      flipped = !flipped;
      div.classList.toggle("flipped", flipped);
      div.innerHTML = flipped
        ? `<p class="fc-label">Answer</p><p class="fc-text">${escapeHTML(c.back)}</p>`
        : `<p class="fc-label">Term</p><p class="fc-text">${escapeHTML(c.front)}</p>`;
    });
    grid.appendChild(div);
  });
}

// ---------------------------------------------------------------- quiz --

let quizData = [];
let quizAnswers = {};

document.getElementById("quiz-go").addEventListener("click", generateQuiz);
document.getElementById("quiz-topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter") generateQuiz();
});

async function generateQuiz() {
  const topic = document.getElementById("quiz-topic").value.trim();
  if (!topic) return;
  const difficulty = document.getElementById("quiz-difficulty").value;
  const loader = document.getElementById("quiz-loader");
  const container = document.getElementById("quiz-questions");
  const scoreEl = document.getElementById("quiz-score");
  container.innerHTML = "";
  scoreEl.classList.add("hidden");
  quizAnswers = {};
  loader.classList.remove("hidden");
  const data = await postJSON("/api/quiz", { topic, difficulty });
  loader.classList.add("hidden");
  quizData = (data && data.questions) || [];
  renderQuiz();
}

function renderQuiz() {
  const container = document.getElementById("quiz-questions");
  const scoreEl = document.getElementById("quiz-score");
  container.innerHTML = "";
  if (quizData.length === 0) return;
  let score = 0;
  quizData.forEach((q, i) => {
    if (quizAnswers[i] === q.correctIndex) score += 1;
  });
  scoreEl.textContent = `Score: ${score} / ${quizData.length}`;
  scoreEl.classList.remove("hidden");

  quizData.forEach((q, i) => {
    const wrap = document.createElement("div");
    wrap.className = "quiz-q";
    const qText = document.createElement("p");
    qText.className = "quiz-q-text";
    qText.textContent = `${i + 1}. ${q.q}`;
    wrap.appendChild(qText);

    const optWrap = document.createElement("div");
    optWrap.className = "quiz-options";
    const answered = quizAnswers[i] !== undefined;
    q.options.forEach((opt, oi) => {
      const btn = document.createElement("button");
      btn.className = "quiz-opt";
      btn.textContent = opt;
      if (answered) {
        btn.disabled = true;
        const isChosen = oi === quizAnswers[i];
        const correct = oi === q.correctIndex;
        if (isChosen) btn.classList.add(correct ? "correct" : "wrong");
      } else {
        btn.addEventListener("click", () => {
          quizAnswers[i] = oi;
          renderQuiz();
        });
      }
      optWrap.appendChild(btn);
    });
    wrap.appendChild(optWrap);

    if (answered) {
      const exp = document.createElement("p");
      exp.className = "quiz-explain";
      exp.textContent = q.explanation;
      wrap.appendChild(exp);
    }
    container.appendChild(wrap);
  });
}

// ---------------------------------------------------------------- mind map --

const MM_COLORS = ["#7B6EF6", "#1D9E75", "#D4537E", "#B4881A", "#4FB6E8", "#534AB7"];

document.getElementById("mm-go").addEventListener("click", generateMindMap);
document.getElementById("mm-topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter") generateMindMap();
});

async function generateMindMap() {
  const topic = document.getElementById("mm-topic").value.trim();
  if (!topic) return;
  const loader = document.getElementById("mm-loader");
  const svgWrap = document.getElementById("mm-svg-wrap");
  const legend = document.getElementById("mm-legend");
  svgWrap.innerHTML = "";
  legend.innerHTML = "";
  loader.classList.remove("hidden");
  const map = await postJSON("/api/mindmap", { topic });
  loader.classList.add("hidden");
  if (!map || !map.branches || map.branches.length === 0) return;

  const cx = 260, cy = 210, R = 155;
  const branches = map.branches;
  let svg = `<svg viewBox="0 0 520 420" style="width:100%;max-width:540px;height:auto;">`;
  branches.forEach((b, i) => {
    const angle = (i / branches.length) * 2 * Math.PI - Math.PI / 2;
    const x = cx + R * Math.cos(angle);
    const y = cy + R * Math.sin(angle);
    const col = MM_COLORS[i % MM_COLORS.length];
    svg += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#ECE9F7" stroke-width="2" />`;
    svg += `<circle cx="${x}" cy="${y}" r="44" fill="#fff" stroke="${col}" stroke-width="2" />`;
    svg += `<foreignObject x="${x - 40}" y="${y - 26}" width="80" height="52">
      <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:10.5px;line-height:1.25;color:#22232E;text-align:center;font-weight:600;">${escapeHTML(b.label)}</div>
    </foreignObject>`;
  });
  svg += `<circle cx="${cx}" cy="${cy}" r="58" fill="#7B6EF6" />`;
  svg += `<foreignObject x="${cx - 50}" y="${cy - 30}" width="100" height="60">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:12px;line-height:1.3;color:#fff;text-align:center;font-weight:700;">${escapeHTML(map.center)}</div>
  </foreignObject>`;
  svg += `</svg>`;
  svgWrap.innerHTML = svg;

  branches.forEach((b, i) => {
    const p = document.createElement("p");
    p.innerHTML = `<span style="color:${MM_COLORS[i % MM_COLORS.length]};font-weight:600;">${escapeHTML(b.label)}:</span> ${escapeHTML(b.note)}`;
    legend.appendChild(p);
  });
}

// ---------------------------------------------------------------- course --

document.getElementById("course-go").addEventListener("click", generateCourse);
document.getElementById("course-topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter") generateCourse();
});

async function generateCourse() {
  const topic = document.getElementById("course-topic").value.trim();
  if (!topic) return;
  let chapters = Number(document.getElementById("course-chapters").value) || 3;
  if (chapters < 2) chapters = 2;
  if (chapters > 8) chapters = 8;
  
  const loader = document.getElementById("course-loader");
  const result = document.getElementById("course-result");
  result.innerHTML = "";
  loader.classList.remove("hidden");
  const course = await postJSON("/api/course", { topic, chapters });
  loader.classList.add("hidden");
  if (!course || !course.chapters) return;

  let html = `<p class="course-title">${escapeHTML(course.title)}</p>
    <p class="course-summary">${escapeHTML(course.summary)}</p>
    <div class="course-chapters">`;
  course.chapters.forEach((c, i) => {
    html += `<div class="chapter-card">
      <p class="chapter-title">Chapter ${i + 1}: ${escapeHTML(c.title)}</p>
      <ul class="chapter-points">${(c.points || []).map((p) => `<li>${escapeHTML(p)}</li>`).join("")}</ul>
    </div>`;
  });
  html += `</div>`;
  result.innerHTML = html;
}

// ---------------------------------------------------------------- todo --

let todoItems = [];

document.getElementById("todo-add").addEventListener("click", addTodo);
document.getElementById("todo-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter") addTodo();
});

function addTodo() {
  const input = document.getElementById("todo-text");
  const text = input.value.trim();
  if (!text) return;
  todoItems.push({ text, done: false });
  input.value = "";
  renderTodo();
}

function renderTodo() {
  const empty = document.getElementById("todo-empty");
  const list = document.getElementById("todo-list");
  empty.classList.toggle("hidden", todoItems.length !== 0);
  list.innerHTML = "";
  todoItems.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "todo-item";
    row.innerHTML = `
      <input type="checkbox" ${item.done ? "checked" : ""} />
      <span class="${item.done ? "done" : ""}">${escapeHTML(item.text)}</span>
      <span class="del">🗑</span>
    `;
    row.querySelector('input[type="checkbox"]').addEventListener("change", () => {
      item.done = !item.done;
      renderTodo();
    });
    row.querySelector(".del").addEventListener("click", () => {
      todoItems.splice(i, 1);
      renderTodo();
    });
    list.appendChild(row);
  });
}

// ---------------------------------------------------------------- utils --

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

// initial render
renderChat();
renderTodo();

</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not API_KEY:
        print(
            "\n[!] ANTHROPIC_API_KEY is not set. The app will still run and the "
            "page will load, but AI features (chat, flashcards, quiz, mind map, "
            "course) will show a friendly error instead of real answers until "
            "you set it.\n"
        )
    app.run(debug=True, port=5000)