/**
 * Step 1: Display Name Input
 *
 * Captures the user's preferred display name.
 */

import React from 'react';

interface StepNameProps {
  displayName: string;
  onChange: (value: string) => void;
  error?: string;
}

export default function StepName({ displayName, onChange, error }: StepNameProps) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h3 style={{ marginBottom: '0.5rem', color: '#333' }}>
        Step 1: What should we call you?
      </h3>
      <p style={{ marginBottom: '1rem', color: '#666', fontSize: '0.9rem' }}>
        This name will be displayed in the navbar when you're logged in.
      </p>

      <div style={{ marginBottom: '1rem' }}>
        <label
          htmlFor="displayName"
          style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#333',
          }}
        >
          Display Name
        </label>
        <input
          id="displayName"
          type="text"
          value={displayName}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Enter your name"
          maxLength={100}
          style={{
            width: '100%',
            padding: '0.75rem',
            borderRadius: '8px',
            border: error ? '1px solid #dc3545' : '1px solid #ddd',
            fontSize: '1rem',
            outline: 'none',
            transition: 'border-color 0.2s',
          }}
        />
        {error && (
          <p style={{ color: '#dc3545', fontSize: '0.85rem', marginTop: '0.25rem' }}>
            {error}
          </p>
        )}
        <p style={{ color: '#888', fontSize: '0.8rem', marginTop: '0.25rem' }}>
          {displayName.length}/100 characters
        </p>
      </div>
    </div>
  );
}
