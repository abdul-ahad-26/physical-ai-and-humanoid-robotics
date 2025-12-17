/**
 * SelectedText - Component for capturing and displaying user text selection.
 *
 * Features:
 * - Captures text selected on the page
 * - Visual indicator when text is selected
 * - Allows clearing the selection
 * - Integrates with ChatWidget for contextual questions
 */

import React, { useState, useEffect, useCallback, createContext, useContext, ReactNode } from 'react';

interface SelectedTextContextType {
  selectedText: string;
  setSelectedText: (text: string) => void;
  clearSelection: () => void;
  hasSelection: boolean;
}

const SelectedTextContext = createContext<SelectedTextContextType>({
  selectedText: '',
  setSelectedText: () => {},
  clearSelection: () => {},
  hasSelection: false,
});

/**
 * Hook to access selected text context.
 */
export function useSelectedText(): SelectedTextContextType {
  return useContext(SelectedTextContext);
}

interface SelectedTextProviderProps {
  children: ReactNode;
  /** Maximum length of selected text to capture */
  maxLength?: number;
}

/**
 * SelectedTextProvider - Provides text selection context to child components.
 */
export function SelectedTextProvider({
  children,
  maxLength = 2000,
}: SelectedTextProviderProps) {
  const [selectedText, setSelectedText] = useState('');

  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection) {
      const text = selection.toString().trim();
      if (text && text.length <= maxLength) {
        setSelectedText(text);
      }
    }
  }, [maxLength]);

  useEffect(() => {
    document.addEventListener('mouseup', handleTextSelection);
    document.addEventListener('keyup', handleTextSelection);

    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
      document.removeEventListener('keyup', handleTextSelection);
    };
  }, [handleTextSelection]);

  const clearSelection = useCallback(() => {
    setSelectedText('');
    window.getSelection()?.removeAllRanges();
  }, []);

  const value: SelectedTextContextType = {
    selectedText,
    setSelectedText,
    clearSelection,
    hasSelection: selectedText.length > 0,
  };

  return (
    <SelectedTextContext.Provider value={value}>
      {children}
    </SelectedTextContext.Provider>
  );
}

interface SelectedTextIndicatorProps {
  /** Maximum characters to display */
  maxDisplayLength?: number;
  /** Custom styles */
  style?: React.CSSProperties;
}

/**
 * SelectedTextIndicator - Visual component showing the currently selected text.
 */
export function SelectedTextIndicator({
  maxDisplayLength = 100,
  style,
}: SelectedTextIndicatorProps) {
  const { selectedText, clearSelection, hasSelection } = useSelectedText();

  if (!hasSelection) {
    return null;
  }

  const displayText =
    selectedText.length > maxDisplayLength
      ? `${selectedText.substring(0, maxDisplayLength)}...`
      : selectedText;

  return (
    <div
      style={{
        padding: '0.5rem 1rem',
        backgroundColor: '#f0fdf4',
        borderBottom: '1px solid #bbf7d0',
        fontSize: '0.875rem',
        ...style,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '0.25rem',
        }}
      >
        <span
          style={{
            color: '#15803d',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4 7V4h16v3" />
            <path d="M9 20h6" />
            <path d="M12 4v16" />
          </svg>
          Selected text context
        </span>
        <button
          onClick={clearSelection}
          aria-label="Clear selection"
          style={{
            background: 'none',
            border: 'none',
            color: '#15803d',
            cursor: 'pointer',
            fontSize: '0.75rem',
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
          }}
        >
          Clear
        </button>
      </div>
      <div
        style={{
          color: '#166534',
          fontStyle: 'italic',
          lineHeight: 1.4,
        }}
      >
        "{displayText}"
      </div>
      <div
        style={{
          color: '#9ca3af',
          fontSize: '0.75rem',
          marginTop: '0.25rem',
        }}
      >
        {selectedText.length} characters selected
      </div>
    </div>
  );
}

interface SelectedTextTooltipProps {
  /** Show tooltip when text is selected */
  show?: boolean;
  /** Callback when user wants to ask about selected text */
  onAskAbout?: (text: string) => void;
}

/**
 * SelectedTextTooltip - Floating tooltip that appears near selected text.
 */
export function SelectedTextTooltip({
  show = true,
  onAskAbout,
}: SelectedTextTooltipProps) {
  const { selectedText, hasSelection } = useSelectedText();
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!show) return;

    const handleSelectionChange = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim()) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setPosition({
          x: rect.left + rect.width / 2,
          y: rect.top - 10,
        });
        setVisible(true);
      } else {
        setVisible(false);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, [show]);

  if (!visible || !hasSelection) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        left: position.x,
        top: position.y,
        transform: 'translate(-50%, -100%)',
        backgroundColor: '#10B981',
        color: 'white',
        padding: '0.5rem 0.75rem',
        borderRadius: '6px',
        fontSize: '0.875rem',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        zIndex: 10000,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
      }}
      onClick={() => onAskAbout?.(selectedText)}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      Ask about this
    </div>
  );
}

export default SelectedTextIndicator;
