/**
 * Better Auth Client Configuration
 *
 * This file configures the Better Auth client for the Docusaurus frontend.
 * It provides hooks for authentication: useSession, signIn, signUp, signOut.
 */

import { createAuthClient } from "better-auth/react";

// Get API URL from Docusaurus customFields (injected at build time)
// @ts-ignore - customFields is injected by Docusaurus
const apiUrl = (typeof window !== 'undefined' && window.docusaurus?.siteConfig?.customFields?.apiUrl) || "http://localhost:8000";

export const authClient = createAuthClient({
  baseURL: apiUrl,
  // Fetch options to include credentials (cookies)
  fetchOptions: {
    credentials: "include",
  },
});

// Export destructured methods for convenience
export const useSession = authClient.useSession;
export const signIn = authClient.signIn;
export const signUp = authClient.signUp;
export const signOut = authClient.signOut;
