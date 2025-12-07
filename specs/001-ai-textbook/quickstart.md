# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Overview
This guide will help you get started with the Physical AI & Humanoid Robotics textbook project. The textbook is built with Docusaurus v3 and follows the Spec-Kit Plus methodology for spec-driven development.

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git for version control
- Basic knowledge of Markdown and JavaScript/React (for advanced customizations)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd physical-ai-and-humanoid-robotics
```

### 2. Install Dependencies
```bash
npm install
# or
yarn install
```

### 3. Start the Development Server
```bash
npm run start
# or
yarn start
```

This will start the Docusaurus development server at `http://localhost:3000` with hot reloading.

### 4. Project Structure Overview
```
/
├── docs/                 # Textbook chapters organized by topic
│   ├── intro/           # Introduction chapter
│   ├── physical-ai/     # Physical AI concepts
│   ├── ros2-core/       # ROS 2 fundamentals
│   └── ...              # Additional chapters
├── code/                # Code examples organized by technology
│   ├── python/          # Python examples
│   ├── ros2/            # ROS 2 specific code
│   └── ...              # Other technology examples
├── assets/              # Images, diagrams, and figures
├── docusaurus.config.js # Docusaurus configuration
└── sidebars.js          # Navigation sidebar configuration
```

## Adding New Content

### 1. Create a New Chapter
To add a new chapter, create a new directory in `/docs/` with the chapter name:

```bash
mkdir docs/new-chapter
```

### 2. Add Chapter Files
Each chapter should have at least these three files:
- `index.md` - Main chapter content
- `lab.md` - Hands-on exercises
- `code.md` - Code examples

Example structure for a new chapter:
```
docs/new-chapter/
├── index.md
├── lab.md
└── code.md
```

### 3. Chapter File Template
**index.md**:
```markdown
---
title: Chapter Title
sidebar_position: 1
description: Brief description of the chapter
---

# Chapter Title

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Content
Your chapter content here...

## Troubleshooting Tips
- Tip 1
- Tip 2
```

## Running the Textbook Locally
- Development mode: `npm run start`
- Production build: `npm run build`
- Serve production build: `npm run serve`
- Deploy to GitHub Pages: `npm run deploy`

## Content Guidelines
1. Follow the writing style guidelines from the constitution
2. Include learning objectives in each chapter
3. Add hands-on exercises in the lab.md file
4. Provide code examples in the code.md file
5. Use consistent heading hierarchy
6. Include visual diagram placeholders where appropriate

## Next Steps
1. Explore the existing chapters in the `/docs/` directory
2. Review the code examples in the `/code/` directory
3. Familiarize yourself with the Docusaurus configuration
4. Start creating your first chapter following the established patterns