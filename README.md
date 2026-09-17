<div align="center">

# 🎓 SmartStud.Ai

### All-in-One Autonomous AI Study Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![AI Providers](https://img.shields.io/badge/AI-Gemini%20%7C%20Claude%20%7C%20GPT--4o-blueviolet?style=for-the-badge)](https://github.com/ajayvbijumon60-cyber/smartstud-AI)

> **SmartStud.Ai** is a powerful, single-file Flask web application that turns your browser into a complete AI-powered study environment. Supports multiple AI providers, runs locally, and is accessible across your entire Wi-Fi network.

</div>

---

## ✨ Features

### 🤖 Multi-Provider AI Engine

| Provider | Models | Key Format |
|----------|--------|------------|
| Google Gemini | Gemini 1.5 / 2.0 Flash | `AIzaSy...` |
| Anthropic | Claude 3.5 Sonnet | `sk-ant-...` |
| OpenAI | GPT-4o Mini | `sk-...` |
| Groq | Fast Inference | `gsk_...` |
| Built-in | Deep Knowledge Engine (offline) | No key needed |

### 📚 Study Tools

- **🃏 3D Flashcards** — Perspective-flip cards with smooth CSS animations
- **🧠 Interactive Mind Maps** — SVG radial mind maps generated from any topic
- **🎯 Adaptive Quiz Arena** — Multi-choice quizzes with XP system and confetti
- **⏱️ Pomodoro Timer** — Built-in focus timer to boost productivity
- **📝 Task Planner** — Local to-do manager with priority tracking
- **📖 Personal Notes RAG** — Upload your notes; AI answers from *your* content (offline!)
- **📊 Analytics Dashboard** — Track XP progress with an SVG chart
- **🔗 Link Sharing** — SQLite-backed multi-device session sharing over LAN

### 🎙️ Rich Media

- **Text-to-Speech (TTS)** — Reads explanations aloud via Web Speech API
- **Speech-to-Text (STT)** — Voice-input your questions
- **Sound Effects** — Subtle audio feedback via Web Audio API
- **Markdown Rendering** — Full Markdown support via Marked.js

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/ajayvbijumon60-cyber/smartstud-AI.git
cd smartstud-AI
```

### 2. Install Dependencies

```bash
pip install flask anthropic python-dotenv
```

### 3. Configure API Keys (Optional)

Create a `.env` file in the project root:

```env
# Use any one or more providers
GEMINI_API_KEY=AIzaSy...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

> ℹ️ If no keys are set, SmartStud.Ai uses the built-in **Deep Knowledge Engine** — no internet required!

### 4. Run the App

```bash
python app.py
```

### 5. Open in Browser

```
Local:    http://localhost:5000
Network:  http://<your-LAN-IP>:5000
```

---

## 🌐 Network / LAN Access

SmartStud.Ai automatically detects your local IP and binds to `0.0.0.0`:

```
=======================================================
 SmartStud.Ai Study Platform is Running!
 Local URL:    http://localhost:5000
 Network LAN:  http://172.16.x.x:5000
 (Anyone on your Wi-Fi can open the Network URL!)
=======================================================
```

---

## 🏗️ Architecture

```
smartstud-AI/
├── app.py          ← Entire application (Flask backend + HTML/CSS/JS frontend)
├── smartstud.db    ← SQLite database (auto-created on first run)
├── test_app.py     ← Unit tests
├── .env            ← Your API keys (DO NOT commit this!)
└── .gitignore
```

### Backend Components

| Component | Description |
|-----------|-------------|
| Flask Routes | REST API endpoints for all AI features |
| Notes RAG Engine | TF-IDF cosine similarity retrieval (pure Python, zero ML deps) |
| SQLite DB | Stores shared sessions, note chunks, and study rooms |
| Multi-provider AI | Auto-detects provider from API key prefix |
| LAN IP Detection | Discovers local network IP for multi-device access |

---

## 🔒 Security

- **Never hardcode API keys** — use `.env` instead
- `.env` is in `.gitignore` and will not be committed
- API keys are passed per-request via headers or request body
- GitHub Push Protection blocks accidental secret commits

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| Database | SQLite 3 |
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| Markdown | Marked.js |
| Speech | Web Speech API (TTS + STT) |
| Audio | Web Audio API |
| Mind Maps | SVG (server-rendered) |
| AI Providers | Google Gemini, Anthropic Claude, OpenAI GPT, Groq |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Made with ❤️ by [ajayvbijumon60-cyber](https://github.com/ajayvbijumon60-cyber)

⭐ Star this repo if you find it useful!

</div>
