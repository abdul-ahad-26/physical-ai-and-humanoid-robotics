/**
 * ChatWidget - Main component wrapping OpenAI ChatKit for the RAG chatbot.
 *
 * Features:
 * - Floating widget positioned bottom-right
 * - Green (#10B981) accent color theme
 * - Better Auth session integration (shows login prompt when not authenticated)
 * - Keyboard accessibility (Tab/Enter/Escape)
 * - Selected text context support
 */

import React, { useState, useEffect, useCallback } from 'react';

// Note: These imports require installing the packages
// npm install @openai/chatkit-react
// For Better Auth: npm install better-auth (or use session context)

// Type definitions for when packages are not yet installed
interface ChatKitControl {
  // Control object from useChatKit
}

interface ChatKitOptions {
  api: {
    url: string;
    domainKey: string;
  };
  theme?: {
    colorScheme?: 'light' | 'dark';
    radius?: 'round' | 'square';
    color?: {
      accent?: {
        primary?: string;
        level?: number;
      };
    };
  };
  startScreen?: {
    greeting?: string;
    prompts?: Array<{
      label: string;
      prompt: string;
      icon?: string;
    }>;
  };
  composer?: {
    placeholder?: string;
  };
}

// Environment configuration
// Note: In Docusaurus, use docusaurus.config.js customFields for env vars
const API_URL = typeof window !== 'undefined'
  ? (window as any).__API_URL__ || 'http://localhost:8000'
  : 'http://localhost:8000';
const DOMAIN_KEY = 'textbook-rag';

// ChatKit configuration options
const chatKitOptions: ChatKitOptions = {
  api: {
    url: `${API_URL}/api/chatkit`,
    domainKey: DOMAIN_KEY,
  },
  theme: {
    colorScheme: 'light',
    radius: 'round',
    color: {
      accent: {
        primary: '#10B981', // Green accent color per spec
        level: 2,
      },
    },
  },
  startScreen: {
    greeting: 'Ask me about this textbook!',
    prompts: [
      {
        label: 'Explain a concept',
        prompt: 'Can you explain ',
        icon: 'lightbulb',
      },
      {
        label: 'Find information',
        prompt: 'Where can I learn about ',
        icon: 'search',
      },
      {
        label: 'Summarize section',
        prompt: 'Summarize the section on ',
        icon: 'document',
      },
    ],
  },
  composer: {
    placeholder: 'Ask a question about the book...',
  },
};

// Placeholder for actual ChatKit implementation
// Replace with actual import when package is installed:
// import { ChatKit, useChatKit } from '@openai/chatkit-react';

interface ChatWidgetProps {
  /** Whether the user is authenticated */
  isAuthenticated?: boolean;
  /** User session data */
  session?: {
    user?: {
      id: string;
      email: string;
      name?: string;
    };
  } | null;
  /** Custom API URL override */
  apiUrl?: string;
}

/**
 * ChatWidget component that renders the AI chat interface.
 * Shows login prompt when user is not authenticated (per FR-010).
 */
export function ChatWidget({
  isAuthenticated = false,
  session = null,
  apiUrl,
}: ChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState<string>('');

  // Check if user is logged in
  const isLoggedIn = isAuthenticated || !!session;

  // Handle text selection on the page
  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      setSelectedText(selection.toString().trim());
    }
  }, []);

  // Set up text selection listener
  useEffect(() => {
    document.addEventListener('mouseup', handleTextSelection);
    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
    };
  }, [handleTextSelection]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to close
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
      // Ctrl/Cmd + Shift + C to toggle chat
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  // Always render the widget icon (per FR-010)
  return (
    <div
      className="chat-widget-container"
      style={{
        position: 'fixed',
        bottom: '1rem',
        right: '1rem',
        zIndex: 9999,
      }}
    >
      {/* Toggle Button - Always visible */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          aria-label="Open chat"
          style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            backgroundColor: '#10B981',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'transform 0.2s, box-shadow 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'scale(1.05)';
            e.currentTarget.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.2)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
          }}
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      )}

      {/* Login Prompt Panel - Shown when not authenticated */}
      {isOpen && !isLoggedIn && (
        <div
          style={{
            width: '360px',
            height: 'auto',
            backgroundColor: 'white',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '1rem',
              borderBottom: '1px solid #e5e7eb',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#10B981',
              color: 'white',
            }}
          >
            <span style={{ fontWeight: 600 }}>Textbook Assistant</span>
            <button
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
              style={{
                background: 'none',
                border: 'none',
                color: 'white',
                cursor: 'pointer',
                padding: '4px',
              }}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          {/* Login Prompt Content */}
          <div
            style={{
              padding: '2rem',
              textAlign: 'center',
            }}
          >
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#10B981"
              strokeWidth="1.5"
              style={{ marginBottom: '1rem' }}
            >
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            <h3 style={{ margin: '0 0 0.5rem 0', color: '#1f2937' }}>
              Login Required
            </h3>
            <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: '0 0 1.5rem 0' }}>
              Please log in to use the AI textbook assistant and ask questions about the content.
            </p>
            <a
              href="/login"
              style={{
                display: 'inline-block',
                padding: '0.75rem 1.5rem',
                backgroundColor: '#10B981',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '8px',
                fontWeight: 500,
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#059669';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#10B981';
              }}
            >
              Log In
            </a>
            <p style={{ color: '#9ca3af', fontSize: '0.75rem', marginTop: '1rem' }}>
              Don't have an account?{' '}
              <a href="/signup" style={{ color: '#10B981' }}>
                Sign up
              </a>
            </p>
          </div>
        </div>
      )}

      {/* Chat Panel - Only shown when authenticated */}
      {isOpen && isLoggedIn && (
        <div
          style={{
            width: '360px',
            height: '600px',
            backgroundColor: 'white',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '1rem',
              borderBottom: '1px solid #e5e7eb',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#10B981',
              color: 'white',
            }}
          >
            <span style={{ fontWeight: 600 }}>Textbook Assistant</span>
            <button
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
              style={{
                background: 'none',
                border: 'none',
                color: 'white',
                cursor: 'pointer',
                padding: '4px',
              }}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          {/* Selected Text Context */}
          {selectedText && (
            <div
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#f0fdf4',
                borderBottom: '1px solid #bbf7d0',
                fontSize: '0.875rem',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span style={{ color: '#15803d', fontWeight: 500 }}>
                  Selected text context:
                </span>
                <button
                  onClick={() => setSelectedText('')}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#15803d',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                  }}
                >
                  Clear
                </button>
              </div>
              <div
                style={{
                  color: '#166534',
                  marginTop: '0.25rem',
                  maxHeight: '3rem',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                "{selectedText.substring(0, 100)}
                {selectedText.length > 100 ? '...' : ''}"
              </div>
            </div>
          )}

          {/* Chat Content Area - Placeholder */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              padding: '2rem',
              color: '#6b7280',
              textAlign: 'center',
            }}
          >
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#10B981"
              strokeWidth="1.5"
              style={{ marginBottom: '1rem' }}
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <p style={{ marginBottom: '0.5rem', fontWeight: 500 }}>
              Ask me about this textbook!
            </p>
            <p style={{ fontSize: '0.875rem' }}>
              I can help you understand concepts, find information, and answer
              questions about the content.
            </p>
            <p
              style={{
                fontSize: '0.75rem',
                marginTop: '1rem',
                color: '#9ca3af',
              }}
            >
              ChatKit integration pending - install @openai/chatkit-react
            </p>
          </div>

          {/* Input Area - Placeholder */}
          <div
            style={{
              padding: '1rem',
              borderTop: '1px solid #e5e7eb',
            }}
          >
            <div
              style={{
                display: 'flex',
                gap: '0.5rem',
              }}
            >
              <input
                type="text"
                placeholder="Ask a question about the book..."
                style={{
                  flex: 1,
                  padding: '0.75rem 1rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  outline: 'none',
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#10B981';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#d1d5db';
                }}
              />
              <button
                style={{
                  padding: '0.75rem 1rem',
                  backgroundColor: '#10B981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                }}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
