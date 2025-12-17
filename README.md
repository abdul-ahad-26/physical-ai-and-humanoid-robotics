# Physical AI & Humanoid Robotics — AI-Native Technical Textbook

This repository contains a complete AI-native textbook for the Physical AI & Humanoid Robotics course, featuring an embedded RAG (Retrieval-Augmented Generation) chatbot. The textbook covers the full curriculum from ROS 2 fundamentals to advanced VLA systems and humanoid robotics, with structured chapters, hands-on labs, and code examples.

## Project Structure

This is a monorepo containing:

```
physical-ai-and-humanoid-robotics/
├── frontend/          # Docusaurus v3 textbook site with RAG chatbot widget
├── backend/           # FastAPI service with OpenAI Agents SDK
├── specs/             # Feature specifications and design documents
├── history/           # Prompt history records and ADRs
└── .specify/          # SpecKit Plus templates and scripts
```

### Frontend (Docusaurus Textbook)

Interactive textbook with embedded chat widget for AI-powered Q&A.

**Tech Stack**: Docusaurus v3, React, OpenAI ChatKit, Better Auth
**See**: [`frontend/README.md`](frontend/README.md) for setup and usage

### Backend (RAG API)

FastAPI service orchestrating multi-agent workflows for context-aware question answering.

**Tech Stack**: FastAPI, OpenAI Agents SDK, Qdrant, Neon Postgres
**See**: [`backend/README.md`](backend/README.md) for API documentation

## Features

### 1. Comprehensive Textbook Content

✅ **COMPLETED** - All 18 chapters implemented with comprehensive content, hands-on labs, and code examples.

**Topics Covered**:
1. Introduction to Physical AI
2. Embodied Intelligence Concepts
3. Sensor Fundamentals (LiDAR, Cameras, IMUs)
4. ROS 2 Foundations
5. ROS 2 Packages & Nodes
6. URDF / SDF & Robot Description
7. Gazebo Simulation
8. Unity Visualization
9. NVIDIA Isaac Sim
10. Isaac ROS (VSLAM, Navigation)
11. VLA Systems (LLM + Vision + Action)
12. Conversational Robotics (Whisper + GPT + ROS 2)
13. Humanoid Kinematics & Locomotion
14. Manipulation & Grasping
15. Human–Robot Interaction (HRI)
16. Capstone: Autonomous Humanoid
17. Hardware Requirements & Lab Architecture
18. Cloud vs. On-Prem Physical AI Lab

**Designed to be**:
- **Comprehensive**: 18 core topics from basic ROS 2 to advanced VLA systems
- **Practical**: Hands-on exercises, setup instructions, and real robotics examples
- **AI-Native**: Optimized for both human learners and AI agents with semantic markup for RAG systems
- **Modular**: Each chapter structured independently with clear interfaces

### 2. RAG Chatbot

Intelligent AI assistant embedded in the textbook that:
- Answers questions grounded in textbook content
- Provides citations linking to relevant book sections
- Supports contextual queries from selected text
- Requires authentication for access

### 3. Authentication System

Secure user authentication powered by Better Auth:
- Email/password login and signup
- Session management with Neon Postgres
- Protected chat access for authenticated users

## Quick Start

### Prerequisites

- **Node.js**: v18.0 or higher
- **Python**: 3.11 or higher
- **Accounts**: Neon Postgres, Qdrant Cloud, OpenAI API

### 1. Clone Repository

```bash
git clone https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics.git
cd physical-ai-and-humanoid-robotics
```

### 2. Set Up Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials

# Initialize database
python scripts/init_db.py

# Ingest textbook content
python scripts/ingest_docs.py

# Start API server
uvicorn src.main:app --reload
```

Backend runs at `http://localhost:8000`

### 3. Set Up Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

### 4. Use the Textbook

1. Open `http://localhost:3000` in your browser
2. Click the chat icon in the bottom-right corner
3. Sign up for a new account or log in
4. Ask questions about the textbook content

## Development

This project follows the **Spec-Driven Development (SDD)** methodology using SpecKit Plus.

### Key Commands

| Command | Description |
|---------|-------------|
| `/sp.specify` | Create feature specification |
| `/sp.plan` | Generate implementation plan |
| `/sp.tasks` | Break down into actionable tasks |
| `/sp.implement` | Execute implementation |
| `/sp.adr` | Document architectural decisions |

### Project Organization

- **Specs**: Feature specifications in `specs/<feature-name>/spec.md`
- **Plans**: Architecture plans in `specs/<feature-name>/plan.md`
- **Tasks**: Task breakdowns in `specs/<feature-name>/tasks.md`
- **History**: Prompt records in `history/prompts/`
- **ADRs**: Architecture decisions in `history/adr/`

## Database Schema

The system uses two databases:

**Neon Postgres** (Relational):
- `users` - User accounts and profiles
- `sessions` - Better Auth session management
- `messages` - Chat conversation history
- `retrieval_logs` - RAG retrieval tracking
- `performance_metrics` - System performance data

**Qdrant Cloud** (Vector):
- Textbook content embeddings
- Semantic search for RAG retrieval

See [`specs/002-rag-chatbot/spec.md`](specs/002-rag-chatbot/spec.md) for complete schema.

## Deployment

### Frontend (Static Site)

Deploy to any static hosting:
- **GitHub Pages**: `npm run deploy` (from frontend/)
- **Netlify**: Connect repository and set build command to `npm run build` in frontend/
- **Vercel**: Import project and configure build settings

### Backend (API Service)

Deploy to:
- **Render**: Use `backend/render.yaml` configuration
- **Railway**: Connect repository and configure root directory to `backend/`
- **AWS/GCP/Azure**: Deploy as containerized service using `backend/Dockerfile`

## Contributing

Contributions are welcome! Please:

1. Follow existing patterns for chapter structure and content
2. Use the SDD methodology for new features
3. Create feature branches: `NNN-feature-name` (e.g., `003-video-tutorials`)
4. Submit pull requests with clear descriptions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUSAURUS FRONTEND                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  OpenAI ChatKit Widget + Better Auth Integration    │    │
│  └─────────────────────┬───────────────────────────────┘    │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │     OpenAI Agents SDK Multi-Agent Orchestration     │    │
│  │  (Retrieval → Answer Generation → Citation)         │    │
│  └─────────────────────┬───────────────┬───────────────┘    │
└────────────────────────┼───────────────┼────────────────────┘
                         │               │
                ┌────────▼──────┐  ┌─────▼──────────┐
                │ QDRANT CLOUD  │  │ NEON POSTGRES  │
                │  (Vectors)    │  │  (Relational)  │
                └───────────────┘  └────────────────┘
```

## License

This textbook is open source and available under the [MIT License](LICENSE).

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics/issues)
- Review documentation in [`frontend/README.md`](frontend/README.md) and [`backend/README.md`](backend/README.md)
- Check feature specifications in `specs/` directory