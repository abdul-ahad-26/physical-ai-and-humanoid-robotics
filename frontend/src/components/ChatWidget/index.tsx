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

import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './ChatWidget.module.css';

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

  // Get API URL from Docusaurus config (respects API_URL env var at build time)
  const { siteConfig } = useDocusaurusContext();
  const effectiveApiUrl = apiUrl || (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  // Check if user is logged in
  const isLoggedIn = isAuthenticated || !!session;

  // Simple chat UI state (fallback when ChatKit not available)
  const [messages, setMessages] = useState<Array<{role: 'user' | 'assistant', content: string, citations?: any[]}>>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoadingMessage, setIsLoadingMessage] = useState(false);
  const [sessionIdState, setSessionIdState] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message function
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoadingMessage) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoadingMessage(true);

    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      const response = await fetch(`${effectiveApiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionIdState,
          selected_text: selectedText || undefined,
        }),
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      if (!sessionIdState && data.session_id) {
        setSessionIdState(data.session_id);
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        citations: data.citations
      }]);

      if (selectedText) setSelectedText('');
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setIsLoadingMessage(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

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
        zIndex: 10000,
      }}
    >
      {/* Toggle Button - Always visible */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          aria-label="Open chat"
          className={styles.toggleButton}
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
        <div className={styles.loginPromptPanel}>
          {/* Header */}
          <div className={styles.chatHeader}>
            <span style={{ fontWeight: 600 }}>Textbook Assistant</span>
            <button
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
              className={styles.closeButton}
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
          <div className={styles.loginPromptContent}>
            <svg
              className={styles.loginPromptIcon}
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              strokeWidth="1.5"
            >
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            <h3 className={styles.loginPromptTitle}>
              Login Required
            </h3>
            <p className={styles.loginPromptText}>
              Please log in to use the AI textbook assistant and ask questions about the content.
            </p>
            <a
              href="/login"
              className={styles.loginButton}
            >
              Log In
            </a>
            <p className={styles.signupPrompt}>
              Don't have an account?{' '}
              <a href="/signup" className={styles.signupLink}>
                Sign up
              </a>
            </p>
          </div>
        </div>
      )}

      {/* Chat Panel - Only shown when authenticated */}
      {isOpen && isLoggedIn && (
        <div className={styles.chatPanel} style={{ height: '600px' }}>
          {/* Header */}
          <div className={styles.chatHeader}>
            <span style={{ fontWeight: 600 }}>Textbook Assistant</span>
            <button
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
              className={styles.closeButton}
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

          {/* Chat Messages Area */}
          <div className={styles.messagesArea}>
            {messages.length === 0 ? (
              <div className={styles.emptyState}>
                <p className={styles.emptyStateTitle}>Ask me about the textbook!</p>
                <div className={styles.promptButtons}>
                  {['💡 Can you explain ', '🔍 Where can I learn about ', '📄 Summarize the section on '].map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => setInputValue(prompt)}
                      className={styles.promptButton}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`${styles.messageContainer} ${msg.role === 'user' ? styles.messageContainerUser : styles.messageContainerAssistant}`}
                  >
                    <div
                      className={`${styles.message} ${msg.role === 'user' ? styles.messageUser : styles.messageAssistant}`}
                    >
                      <div style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
                        {msg.role === 'assistant' ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        ) : (
                          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {msg.content}
                          </div>
                        )}
                      </div>
                      {msg.citations && msg.citations.length > 0 && (
                        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#6b7280', borderTop: '1px solid #e5e7eb', paddingTop: '0.5rem' }}>
                          <strong>Sources:</strong>{' '}
                          {msg.citations.map((cit: any, cidx: number) => (
                            <a
                              key={cidx}
                              href={cit.anchor_url}
                              style={{
                                color: '#10B981',
                                marginRight: '0.5rem',
                                textDecoration: 'underline',
                                cursor: 'pointer'
                              }}
                              onClick={(e) => {
                                e.preventDefault();
                                window.location.href = cit.anchor_url;
                              }}
                            >
                              {cit.display_text}
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isLoadingMessage && (
                  <div style={{ textAlign: 'center', color: '#6b7280', fontSize: '0.875rem', fontStyle: 'italic' }}>
                    Thinking...
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input Area */}
          <div className={styles.inputArea}>
            <div className={styles.inputContainer}>
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask a question about the book..."
                disabled={isLoadingMessage}
                rows={2}
                className={styles.textarea}
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoadingMessage}
                className={`${styles.sendButton} ${inputValue.trim() && !isLoadingMessage ? styles.sendButtonEnabled : styles.sendButtonDisabled}`}
              >
                {isLoadingMessage ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
