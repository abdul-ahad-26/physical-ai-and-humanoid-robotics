/**
 * Personalization API Client
 *
 * Client functions for profile management, content personalization,
 * and translation features (005-user-personalization).
 */

import {
  UserProfile,
  ProfileUpdateRequest,
  ProfileUpdateResponse,
  SoftwareBackground,
  HardwareBackground,
} from '../components/ProfileWizard/types';

// Re-export types for convenience
export type {
  UserProfile,
  ProfileUpdateRequest,
  ProfileUpdateResponse,
  SoftwareBackground,
  HardwareBackground,
};

/**
 * Get the API URL from the window context or environment
 */
function getApiUrl(): string {
  // Try to get from Docusaurus site config if available
  if (typeof window !== 'undefined' && (window as any).__DOCUSAURUS_SITE_CONFIG) {
    const config = (window as any).__DOCUSAURUS_SITE_CONFIG;
    return config.customFields?.apiUrl || 'http://localhost:8000';
  }
  return 'http://localhost:8000';
}

// =============================================================================
// Profile API
// =============================================================================

/**
 * Get the current user's profile
 */
export async function getProfile(apiUrl?: string): Promise<UserProfile> {
  const url = apiUrl || getApiUrl();
  const response = await fetch(`${url}/api/profile`, {
    credentials: 'include',
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error?.message || 'Failed to fetch profile');
  }

  const data = await response.json();
  return data.user;
}

/**
 * Update the current user's profile
 */
export async function updateProfile(
  request: ProfileUpdateRequest,
  apiUrl?: string
): Promise<ProfileUpdateResponse> {
  const url = apiUrl || getApiUrl();
  const response = await fetch(`${url}/api/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error?.message || 'Failed to update profile');
  }

  return await response.json();
}

// =============================================================================
// Personalization API
// =============================================================================

export interface PersonalizeRequest {
  chapter_id: string;
  content: string;
  title: string;
}

export interface PersonalizeResponse {
  success: boolean;
  personalized_content: string;
  metadata: {
    user_level: string;
    adaptations_made: string[];
    cached: boolean;
    response_time_ms: number;
  };
}

/**
 * Personalize chapter content based on user's profile
 */
export async function personalizeContent(
  request: PersonalizeRequest,
  apiUrl?: string
): Promise<PersonalizeResponse> {
  const url = apiUrl || getApiUrl();
  const response = await fetch(`${url}/api/personalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));

    // Handle specific error codes
    if (response.status === 403) {
      throw new Error('Profile must be completed before personalizing content');
    }
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After') || '60';
      throw new Error(`Rate limit exceeded. Try again in ${retryAfter} seconds.`);
    }

    throw new Error(data.error?.message || 'Failed to personalize content');
  }

  return await response.json();
}

// =============================================================================
// Translation API
// =============================================================================

export interface TranslateRequest {
  chapter_id: string;
  content: string;
  target_language: string;
}

export interface TranslateResponse {
  success: boolean;
  translated_content: string;
  metadata: {
    source_language: string;
    target_language: string;
    preserved_blocks: number;
    cached: boolean;
    response_time_ms: number;
  };
}

/**
 * Translate chapter content to Urdu
 */
export async function translateContent(
  request: TranslateRequest,
  apiUrl?: string
): Promise<TranslateResponse> {
  const url = apiUrl || getApiUrl();
  const response = await fetch(`${url}/api/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));

    // Handle specific error codes
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After') || '60';
      throw new Error(`Rate limit exceeded. Try again in ${retryAfter} seconds.`);
    }

    throw new Error(data.error?.message || 'Failed to translate content');
  }

  return await response.json();
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Check if user has completed their profile
 */
export async function isProfileComplete(apiUrl?: string): Promise<boolean> {
  try {
    const profile = await getProfile(apiUrl);
    return profile.profile_completed;
  } catch {
    return false;
  }
}

/**
 * Get user's software level for display
 */
export function getSoftwareLevelLabel(level: string): string {
  const labels: Record<string, string> = {
    beginner: 'Beginner',
    intermediate: 'Intermediate',
    advanced: 'Advanced',
  };
  return labels[level] || level;
}

/**
 * Get user's hardware level for display
 */
export function getHardwareLevelLabel(level: string): string {
  const labels: Record<string, string> = {
    none: 'No Experience',
    basic: 'Basic',
    intermediate: 'Intermediate',
    advanced: 'Advanced',
  };
  return labels[level] || level;
}
