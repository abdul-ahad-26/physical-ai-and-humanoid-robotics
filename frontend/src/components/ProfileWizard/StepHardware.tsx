/**
 * Step 3: Hardware Background
 *
 * Captures the user's hardware/robotics experience:
 * - Experience level (none/basic/intermediate/advanced)
 * - Hardware domains (electronics, robotics, GPUs, etc.)
 */

import React from 'react';
import {
  HardwareBackground,
  HardwareLevel,
  HARDWARE_LEVELS,
  HARDWARE_DOMAINS,
} from './types';

interface StepHardwareProps {
  background: HardwareBackground;
  onChange: (value: HardwareBackground) => void;
  error?: string;
}

export default function StepHardware({ background, onChange, error }: StepHardwareProps) {
  const handleLevelChange = (level: HardwareLevel) => {
    onChange({ ...background, level });
  };

  const toggleDomain = (domain: string) => {
    const domains = background.domains.includes(domain)
      ? background.domains.filter((d) => d !== domain)
      : [...background.domains, domain];
    onChange({ ...background, domains });
  };

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h3 style={{ marginBottom: '0.5rem', color: '#333' }}>
        Step 3: Hardware Background
      </h3>
      <p style={{ marginBottom: '1rem', color: '#666', fontSize: '0.9rem' }}>
        Help us understand your experience with hardware and robotics.
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
          onChange={(e) => handleLevelChange(e.target.value as HardwareLevel)}
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
          {HARDWARE_LEVELS.map((level) => (
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

      {/* Hardware Domains */}
      <div style={{ marginBottom: '1rem' }}>
        <label
          style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#333',
          }}
        >
          Hardware Domains (optional)
        </label>
        <p style={{ color: '#888', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          Select areas you have experience with
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {HARDWARE_DOMAINS.map((domain) => (
            <button
              key={domain.value}
              type="button"
              onClick={() => toggleDomain(domain.value)}
              style={{
                padding: '0.375rem 0.75rem',
                borderRadius: '20px',
                border: 'none',
                backgroundColor: background.domains.includes(domain.value)
                  ? '#10B981'
                  : '#e9ecef',
                color: background.domains.includes(domain.value) ? 'white' : '#333',
                cursor: 'pointer',
                fontSize: '0.85rem',
                transition: 'all 0.2s',
              }}
            >
              {domain.label}
            </button>
          ))}
        </div>
      </div>

      {/* Completion message */}
      <div
        style={{
          marginTop: '1.5rem',
          padding: '1rem',
          backgroundColor: '#f0fdf4',
          borderRadius: '8px',
          border: '1px solid #86efac',
        }}
      >
        <p style={{ color: '#166534', fontSize: '0.9rem', margin: 0 }}>
          <strong>Almost done!</strong> Click "Complete Profile" to finish setup and unlock
          personalized content.
        </p>
      </div>
    </div>
  );
}
