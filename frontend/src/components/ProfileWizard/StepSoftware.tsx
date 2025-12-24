/**
 * Step 2: Software Background
 *
 * Captures the user's software development experience:
 * - Experience level (beginner/intermediate/advanced)
 * - Programming languages
 * - Frameworks and libraries
 */

import React from 'react';
import {
  SoftwareBackground,
  SoftwareLevel,
  SOFTWARE_LEVELS,
  SOFTWARE_LANGUAGES,
  SOFTWARE_FRAMEWORKS,
} from './types';

interface StepSoftwareProps {
  background: SoftwareBackground;
  onChange: (value: SoftwareBackground) => void;
  error?: string;
}

export default function StepSoftware({ background, onChange, error }: StepSoftwareProps) {
  const handleLevelChange = (level: SoftwareLevel) => {
    onChange({ ...background, level });
  };

  const toggleLanguage = (lang: string) => {
    const languages = background.languages.includes(lang)
      ? background.languages.filter((l) => l !== lang)
      : [...background.languages, lang];
    onChange({ ...background, languages });
  };

  const toggleFramework = (framework: string) => {
    const frameworks = background.frameworks.includes(framework)
      ? background.frameworks.filter((f) => f !== framework)
      : [...background.frameworks, framework];
    onChange({ ...background, frameworks });
  };

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h3 style={{ marginBottom: '0.5rem', color: '#333' }}>
        Step 2: Software Background
      </h3>
      <p style={{ marginBottom: '1rem', color: '#666', fontSize: '0.9rem' }}>
        Help us tailor the content to your programming experience.
      </p>

      {/* Experience Level */}
      <div style={{ marginBottom: '1.25rem' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#333',
          }}
        >
          Experience Level <span style={{ color: '#dc3545' }}>*</span>
        </label>
        <select
          value={background.level}
          onChange={(e) => handleLevelChange(e.target.value as SoftwareLevel)}
          style={{
            width: '100%',
            padding: '0.75rem',
            borderRadius: '8px',
            border: error ? '1px solid #dc3545' : '1px solid #ddd',
            fontSize: '1rem',
            backgroundColor: 'white',
          }}
        >
          <option value="">Select your level</option>
          {SOFTWARE_LEVELS.map((level) => (
            <option key={level.value} value={level.value}>
              {level.label}
            </option>
          ))}
        </select>
        {error && (
          <p style={{ color: '#dc3545', fontSize: '0.85rem', marginTop: '0.25rem' }}>
            {error}
          </p>
        )}
      </div>

      {/* Programming Languages */}
      <div style={{ marginBottom: '1.25rem' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#333',
          }}
        >
          Programming Languages (optional)
        </label>
        <p style={{ color: '#888', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          Select languages you're familiar with
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {SOFTWARE_LANGUAGES.map((lang) => (
            <button
              key={lang}
              type="button"
              onClick={() => toggleLanguage(lang)}
              style={{
                padding: '0.375rem 0.75rem',
                borderRadius: '20px',
                border: 'none',
                backgroundColor: background.languages.includes(lang)
                  ? '#10B981'
                  : '#e9ecef',
                color: background.languages.includes(lang) ? 'white' : '#333',
                cursor: 'pointer',
                fontSize: '0.85rem',
                transition: 'all 0.2s',
              }}
            >
              {lang}
            </button>
          ))}
        </div>
      </div>

      {/* Frameworks */}
      <div style={{ marginBottom: '1rem' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#333',
          }}
        >
          Frameworks & Libraries (optional)
        </label>
        <p style={{ color: '#888', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          Select frameworks you've used
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {SOFTWARE_FRAMEWORKS.map((framework) => (
            <button
              key={framework}
              type="button"
              onClick={() => toggleFramework(framework)}
              style={{
                padding: '0.375rem 0.75rem',
                borderRadius: '20px',
                border: 'none',
                backgroundColor: background.frameworks.includes(framework)
                  ? '#10B981'
                  : '#e9ecef',
                color: background.frameworks.includes(framework) ? 'white' : '#333',
                cursor: 'pointer',
                fontSize: '0.85rem',
                transition: 'all 0.2s',
              }}
            >
              {framework}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
