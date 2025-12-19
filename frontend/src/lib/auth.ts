/**
 * Better Auth Client Configuration
 *
 * This file configures the Better Auth client for the Docusaurus frontend.
 * It provides hooks for authentication: useSession, signIn, signUp, signOut.
 *
 * The API URL is determined at runtime using Docusaurus's useDocusaurusContext.
 * A wrapper hook (useAuth) is provided that properly initializes the client.
 */

import { useState, useEffect, useMemo } from "react";
import { createAuthClient } from "better-auth/react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

// Type for the auth client
type AuthClient = ReturnType<typeof createAuthClient>;

/**
 * Hook that provides the properly configured auth client.
 * Uses Docusaurus context to get the API URL.
 */
export function useAuthClient(): AuthClient {
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || "http://localhost:8000";

  // Create client with memoization to prevent recreating on every render
  const client = useMemo(() => {
    console.log("[Auth] Creating auth client with baseURL:", apiUrl);
    return createAuthClient({
      baseURL: apiUrl,
      fetchOptions: {
        credentials: "include",
      },
    });
  }, [apiUrl]);

  return client;
}

/**
 * Hook for session management that properly uses Docusaurus context.
 * This is the primary hook to use for authentication state.
 */
export function useSession() {
  const client = useAuthClient();
  return client.useSession();
}

/**
 * Hook for sign in functionality.
 */
export function useSignIn() {
  const client = useAuthClient();
  return client.signIn;
}

/**
 * Hook for sign up functionality.
 */
export function useSignUp() {
  const client = useAuthClient();
  return client.signUp;
}

/**
 * Hook for sign out functionality.
 */
export function useSignOut() {
  const client = useAuthClient();
  return client.signOut;
}

// For backwards compatibility, also export a default client
// This uses a fallback URL and should only be used where hooks can't be used
let _fallbackClient: AuthClient | null = null;

function getFallbackClient(): AuthClient {
  if (!_fallbackClient) {
    // Try to get URL from window if available
    let apiUrl = "http://localhost:8000";
    if (typeof window !== "undefined") {
      const docusaurus = (window as any).docusaurus;
      if (docusaurus?.siteConfig?.customFields?.apiUrl) {
        apiUrl = docusaurus.siteConfig.customFields.apiUrl;
      }
    }
    _fallbackClient = createAuthClient({
      baseURL: apiUrl,
      fetchOptions: {
        credentials: "include",
      },
    });
  }
  return _fallbackClient;
}

// Legacy exports for backwards compatibility
export const authClient = {
  get useSession() {
    return getFallbackClient().useSession;
  },
  get signIn() {
    return getFallbackClient().signIn;
  },
  get signUp() {
    return getFallbackClient().signUp;
  },
  get signOut() {
    return getFallbackClient().signOut;
  },
};

export const signIn = getFallbackClient().signIn;
export const signUp = getFallbackClient().signUp;
export const signOut = getFallbackClient().signOut;
