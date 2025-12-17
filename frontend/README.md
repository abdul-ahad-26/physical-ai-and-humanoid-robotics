# Physical AI & Humanoid Robotics - Frontend

An AI-native technical textbook built with Docusaurus v3, featuring an embedded RAG (Retrieval-Augmented Generation) chatbot powered by OpenAI Agents SDK.

## Overview

This frontend provides:

- **Interactive Textbook**: Comprehensive content on Physical AI, Robotics, and AI Integration organized by chapters
- **RAG Chatbot Widget**: Context-aware AI assistant that answers questions based on textbook content
- **Better Auth Integration**: Secure authentication system with email/password login and session management
- **Responsive Design**: Mobile-friendly interface with dark mode support

## Tech Stack

- **Framework**: Docusaurus v3.9.2 (React-based static site generator)
- **UI Components**: OpenAI ChatKit (`@openai/chatkit-react`) for chat interface
- **Authentication**: Better Auth (TypeScript framework-agnostic auth library)
- **Styling**: Custom CSS with green (#10B981) theme for chat components
- **Node**: v18.0+ required

## Project Structure

```
frontend/
├── docs/                          # Textbook markdown content
│   ├── intro/                     # Introduction chapters
│   ├── ros2-core/                 # ROS2 fundamentals
│   ├── gazebo/                    # Simulation with Gazebo
│   ├── isaac/                     # NVIDIA Isaac platform
│   ├── manipulation/              # Robot manipulation
│   ├── vla/                       # Vision-Language-Action models
│   ├── conversational-robotics/   # HRI and conversational AI
│   ├── capstone/                  # Final projects
│   └── ...                        # Additional chapters
├── src/
│   ├── components/
│   │   └── ChatWidget/            # RAG chatbot widget
│   │       ├── index.tsx          # Main chat component
│   │       ├── AuthGate.tsx       # Login prompt for unauthenticated users
│   │       └── SelectedText.tsx   # Contextual text selection
│   ├── theme/
│   │   └── Root.tsx               # Docusaurus root wrapper with ChatWidget
│   └── css/
│       └── custom.css             # Custom styling and theme overrides
├── static/                        # Static assets (images, favicon)
├── docusaurus.config.js           # Docusaurus configuration
├── sidebars.js                    # Sidebar navigation structure
└── package.json                   # Dependencies and scripts
```

## Getting Started

### Prerequisites

- Node.js v18.0 or higher
- npm or yarn package manager
- Backend API running (see `backend/README.md`)

### Installation

```bash
cd frontend
npm install
```

### Development

Start the development server with hot reload:

```bash
npm start
```

The site will be available at `http://localhost:3000`.

### Build for Production

Generate static files for deployment:

```bash
npm run build
```

Built files will be in the `build/` directory.

### Serve Production Build Locally

Test the production build:

```bash
npm run serve
```

## Key Features

### 1. Textbook Content

- **Structured Learning Path**: Content organized by chapters with progressive complexity
- **Code Examples**: Syntax-highlighted Python, C++, and ROS2 code samples
- **Labs & Exercises**: Hands-on exercises with step-by-step instructions
- **Diagrams**: Mermaid diagrams for system architecture and workflows

### 2. RAG Chatbot

The chat widget appears as a floating icon in the bottom-right corner of every page.

**Features**:
- **Context-Aware Answers**: Responses grounded in textbook content with citations
- **Text Selection Support**: Select text on the page to add context to your question
- **Authentication Required**: Users must log in to use the chat feature
- **Streaming Responses**: Real-time streaming of AI-generated answers
- **Citation Links**: Clickable links to relevant textbook sections

**Usage**:
1. Click the chat icon in the bottom-right corner
2. Log in with your email/password (or sign up for a new account)
3. Type your question in the input field
4. Receive AI-generated answers with citations linking to textbook sections

### 3. Authentication

**Better Auth** provides secure session management:

- **Email/Password Login**: Standard authentication flow
- **Session Cookies**: Secure session storage in Neon Postgres
- **Protected Routes**: Chat feature requires authentication
- **Logout**: Clear session and return to unauthenticated state

**Pages**:
- `/login` - Sign in page
- `/signup` - New user registration
- Logout button in navbar (when authenticated)

## Configuration

### Environment Variables

The frontend connects to the backend API. Configure the API endpoint:

```bash
# .env (create this file in frontend/)
REACT_APP_API_URL=http://localhost:8000
```

### Docusaurus Config

Key settings in `docusaurus.config.js`:

- **Site Title**: "Physical AI & Humanoid Robotics"
- **Base URL**: `/` (adjust for subdirectory deployment)
- **GitHub Integration**: Edit links point to repository
- **Mermaid Support**: Enabled for diagrams
- **Custom CSS**: Theme overrides in `src/css/custom.css`

## Development Guidelines

### Adding New Content

1. Create a new markdown file in the appropriate `docs/` subdirectory
2. Add frontmatter with metadata:
   ```markdown
   ---
   title: Chapter Title
   description: Brief description
   sidebar_position: 1
   ---
   ```
3. Update `sidebars.js` if creating a new chapter section

### Customizing the Chat Widget

The chat widget is implemented in `src/components/ChatWidget/`:

- **Styling**: Edit `src/css/custom.css` to modify colors and layout
- **Authentication Logic**: Update `src/theme/Root.tsx` to integrate Better Auth hooks
- **Chat Behavior**: Modify `ChatWidget/index.tsx` for custom message handling

### Theme Customization

Edit `src/css/custom.css` to customize:
- Primary colors (currently green #10B981 for chat)
- Font families
- Dark mode overrides
- Component-specific styles

## Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start development server with hot reload |
| `npm run build` | Build production static site |
| `npm run serve` | Serve production build locally |
| `npm run clear` | Clear Docusaurus cache |
| `npm run swizzle` | Eject Docusaurus components for customization |
| `npm run write-translations` | Generate translation files |
| `npm run write-heading-ids` | Add explicit heading IDs |

## Deployment

### GitHub Pages

```bash
npm run deploy
```

Ensure `docusaurus.config.js` has correct `organizationName` and `projectName`.

### Other Hosting

Build the site and deploy the `build/` directory to any static hosting service:

- Netlify
- Vercel
- AWS S3 + CloudFront
- Azure Static Web Apps

## Troubleshooting

### Chat Widget Not Appearing

1. Check that `src/theme/Root.tsx` includes `<ChatWidget />`
2. Verify backend API is running and accessible
3. Check browser console for errors

### Authentication Issues

1. Ensure Better Auth is configured with correct database connection
2. Check that session cookies are enabled in the browser
3. Verify CORS settings allow frontend domain in backend

### Content Not Updating

1. Clear Docusaurus cache: `npm run clear`
2. Restart development server
3. Check for markdown syntax errors in console

### Build Failures

1. Check Node.js version (must be v18.0+)
2. Delete `node_modules/` and reinstall: `rm -rf node_modules && npm install`
3. Review error messages for missing dependencies

## Contributing

When contributing content:

1. Follow existing markdown structure and frontmatter format
2. Include code examples with proper syntax highlighting
3. Add alt text to all images for accessibility
4. Test locally before submitting pull requests
5. Keep chapter sections focused and concise

## Related Documentation

- **Backend Setup**: See `backend/README.md` for API and agent configuration
- **Database Schema**: See `specs/002-rag-chatbot/spec.md` for schema details
- **Content Ingestion**: Run `backend/scripts/ingest_docs.py` to index textbook content

## License

Copyright © 2025 Physical AI & Humanoid Robotics Textbook. All rights reserved.

## Support

For issues or questions:
- Open an issue on GitHub
- Review existing documentation in `specs/` directory
- Check Docusaurus documentation: https://docusaurus.io/
