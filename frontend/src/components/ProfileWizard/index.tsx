/**
 * ProfileWizard Component
 *
 * A 3-step modal wizard for capturing user's technical background:
 * - Step 1: Display name
 * - Step 2: Software background (level, languages, frameworks)
 * - Step 3: Hardware background (level, domains)
 *
 * Shows when profile_completed=false after login.
 * Can be skipped but user will be reminded on subsequent logins.
 */

import React, { useState, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import StepName from './StepName';
import StepSoftware from './StepSoftware';
import StepHardware from './StepHardware';
import {
  WizardState,
  SoftwareBackground,
  HardwareBackground,
  ProfileUpdateRequest,
} from './types';

interface ProfileWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
  initialDisplayName?: string;
}

const initialSoftwareBackground: SoftwareBackground = {
  level: '' as any,
  languages: [],
  frameworks: [],
};

const initialHardwareBackground: HardwareBackground = {
  level: '' as any,
  domains: [],
};

export default function ProfileWizard({
  isOpen,
  onClose,
  onComplete,
  initialDisplayName = '',
}: ProfileWizardProps) {
  const [state, setState] = useState<WizardState>({
    step: 1,
    displayName: initialDisplayName,
    softwareBackground: initialSoftwareBackground,
    hardwareBackground: initialHardwareBackground,
    isOpen: isOpen,
    isSkipped: false,
    isLoading: false,
    error: null,
  });

  const [stepErrors, setStepErrors] = useState<{ [key: string]: string }>({});

  // Get API URL from Docusaurus context
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  useEffect(() => {
    setState((prev) => ({ ...prev, isOpen, displayName: initialDisplayName || prev.displayName }));
  }, [isOpen, initialDisplayName]);

  const validateStep = (step: number): boolean => {
    const errors: { [key: string]: string } = {};

    if (step === 1) {
      if (!state.displayName.trim()) {
        errors.displayName = 'Display name is required';
      } else if (state.displayName.trim().length > 100) {
        errors.displayName = 'Display name must be 100 characters or less';
      }
    }

    if (step === 2) {
      if (!state.softwareBackground.level) {
        errors.softwareLevel = 'Please select your software experience level';
      }
    }

    if (step === 3) {
      if (!state.hardwareBackground.level) {
        errors.hardwareLevel = 'Please select your hardware experience level';
      }
    }

    setStepErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleNext = () => {
    if (!validateStep(state.step)) return;

    if (state.step < 3) {
      setState((prev) => ({ ...prev, step: (prev.step + 1) as 1 | 2 | 3 }));
    }
  };

  const handleBack = () => {
    if (state.step > 1) {
      setState((prev) => ({ ...prev, step: (prev.step - 1) as 1 | 2 | 3 }));
      setStepErrors({});
    }
  };

  const handleSkip = () => {
    const confirmSkip = window.confirm(
      'Skipping profile setup will limit personalization features. ' +
        'You can complete your profile later from the Profile Settings page.\n\n' +
        'Are you sure you want to skip?'
    );
    if (confirmSkip) {
      setState((prev) => ({ ...prev, isSkipped: true }));
      onClose();
    }
  };

  const handleComplete = async () => {
    if (!validateStep(3)) return;

    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const request: ProfileUpdateRequest = {
        display_name: state.displayName.trim(),
        software_background: state.softwareBackground,
        hardware_background: state.hardwareBackground,
      };

      const response = await fetch(`${apiUrl}/api/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error?.message || 'Failed to save profile');
      }

      onComplete();
      onClose();
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        error: err.message || 'An error occurred while saving your profile',
        isLoading: false,
      }));
    }
  };

  if (!state.isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: '1rem',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          // Don't close on backdrop click - require explicit skip
        }
      }}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '12px',
          padding: '2rem',
          maxWidth: '500px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#333' }}>Complete Your Profile</h2>
          <p style={{ margin: '0.5rem 0 0', color: '#666', fontSize: '0.9rem' }}>
            Help us personalize your learning experience
          </p>
        </div>

        {/* Progress Indicator */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            marginBottom: '1.5rem',
            gap: '0.5rem',
          }}
        >
          {[1, 2, 3].map((step) => (
            <div
              key={step}
              style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor:
                  state.step === step
                    ? '#10B981'
                    : state.step > step
                    ? '#10B981'
                    : '#e5e7eb',
                color: state.step >= step ? 'white' : '#6b7280',
                fontWeight: 600,
                transition: 'all 0.2s',
              }}
            >
              {state.step > step ? '✓' : step}
            </div>
          ))}
        </div>

        {/* Error Message */}
        {state.error && (
          <div
            style={{
              marginBottom: '1rem',
              padding: '0.75rem 1rem',
              backgroundColor: '#fef2f2',
              border: '1px solid #fecaca',
              borderRadius: '8px',
              color: '#dc2626',
              fontSize: '0.9rem',
            }}
          >
            {state.error}
          </div>
        )}

        {/* Step Content */}
        {state.step === 1 && (
          <StepName
            displayName={state.displayName}
            onChange={(value) => setState((prev) => ({ ...prev, displayName: value }))}
            error={stepErrors.displayName}
          />
        )}

        {state.step === 2 && (
          <StepSoftware
            background={state.softwareBackground}
            onChange={(value) =>
              setState((prev) => ({ ...prev, softwareBackground: value }))
            }
            error={stepErrors.softwareLevel}
          />
        )}

        {state.step === 3 && (
          <StepHardware
            background={state.hardwareBackground}
            onChange={(value) =>
              setState((prev) => ({ ...prev, hardwareBackground: value }))
            }
            error={stepErrors.hardwareLevel}
          />
        )}

        {/* Navigation Buttons */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: '1.5rem',
            paddingTop: '1rem',
            borderTop: '1px solid #e5e7eb',
          }}
        >
          <button
            type="button"
            onClick={handleSkip}
            disabled={state.isLoading}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'transparent',
              color: '#6b7280',
              border: 'none',
              cursor: 'pointer',
              fontSize: '0.9rem',
            }}
          >
            Skip for now
          </button>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {state.step > 1 && (
              <button
                type="button"
                onClick={handleBack}
                disabled={state.isLoading}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Back
              </button>
            )}

            {state.step < 3 ? (
              <button
                type="button"
                onClick={handleNext}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#10B981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Next
              </button>
            ) : (
              <button
                type="button"
                onClick={handleComplete}
                disabled={state.isLoading}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: state.isLoading ? '#9ca3af' : '#10B981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: state.isLoading ? 'not-allowed' : 'pointer',
                  fontWeight: 500,
                }}
              >
                {state.isLoading ? 'Saving...' : 'Complete Profile'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
