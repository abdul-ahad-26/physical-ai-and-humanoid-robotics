/**
 * Configuration utilities for accessing Docusaurus custom fields.
 *
 * This module provides a consistent way to access environment configuration
 * across the frontend, properly handling both SSR and client-side rendering.
 */

/**
 * Get the API URL from Docusaurus customFields.
 *
 * Priority:
 * 1. Docusaurus siteConfig.customFields.apiUrl (set via process.env.API_URL at build time)
 * 2. Fallback to localhost for development
 *
 * @returns The backend API URL
 */
export function getApiUrl(): string {
  // In browser, access via Docusaurus global
  if (typeof window !== 'undefined') {
    // Access Docusaurus site config - this is available after hydration
    const docusaurus = (window as any).docusaurus;
    if (docusaurus?.siteConfig?.customFields?.apiUrl) {
      return docusaurus.siteConfig.customFields.apiUrl as string;
    }
  }

  // Fallback for SSR or if customFields not available
  return 'http://localhost:8000';
}

/**
 * React hook to get the API URL.
 * This is useful in components where you need the URL reactively.
 */
export function useApiUrl(): string {
  // For now, just return the static value
  // In a more complex setup, this could use useEffect to update after hydration
  return getApiUrl();
}
