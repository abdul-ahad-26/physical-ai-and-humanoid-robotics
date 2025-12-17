/**
 * AuthGate - Better Auth session check component.
 *
 * Wraps children and only renders them when the user is authenticated.
 * Uses Better Auth's useSession hook for session management.
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Type definitions for Better Auth session
interface User {
  id: string;
  email: string;
  name?: string;
  image?: string;
}

interface Session {
  user: User;
  expires: string;
}

interface SessionContextType {
  session: Session | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

// Create session context
const SessionContext = createContext<SessionContextType>({
  session: null,
  isLoading: true,
  isAuthenticated: false,
});

/**
 * Hook to access the session context.
 */
export function useSession(): SessionContextType {
  return useContext(SessionContext);
}

interface SessionProviderProps {
  children: ReactNode;
  /** Better Auth API URL */
  authUrl?: string;
}

/**
 * SessionProvider - Provides session context to child components.
 *
 * In production, this should use @better-auth/react's built-in provider.
 * This is a simplified version for development/demo purposes.
 */
export function SessionProvider({
  children,
  authUrl = 'http://localhost:3000',
}: SessionProviderProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch session from Better Auth API
    const fetchSession = async () => {
      try {
        const response = await fetch(`${authUrl}/api/auth/get-session`, {
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          if (data.user) {
            setSession(data);
          }
        }
      } catch (error) {
        console.error('Failed to fetch session:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSession();
  }, [authUrl]);

  const value: SessionContextType = {
    session,
    isLoading,
    isAuthenticated: !!session?.user,
  };

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}

interface AuthGateProps {
  children: ReactNode;
  /** Content to show while loading */
  loading?: ReactNode;
  /** Content to show when not authenticated */
  fallback?: ReactNode;
}

/**
 * AuthGate - Conditionally renders children based on authentication status.
 *
 * Usage:
 * ```tsx
 * <AuthGate>
 *   <ChatWidget />
 * </AuthGate>
 * ```
 */
export function AuthGate({
  children,
  loading = null,
  fallback = null,
}: AuthGateProps) {
  const { isAuthenticated, isLoading } = useSession();

  if (isLoading) {
    return <>{loading}</>;
  }

  if (!isAuthenticated) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

/**
 * Higher-order component version of AuthGate.
 *
 * Usage:
 * ```tsx
 * const ProtectedChatWidget = withAuth(ChatWidget);
 * ```
 */
export function withAuth<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options?: {
    loading?: ReactNode;
    fallback?: ReactNode;
  }
) {
  return function WithAuthComponent(props: P) {
    return (
      <AuthGate loading={options?.loading} fallback={options?.fallback}>
        <WrappedComponent {...props} />
      </AuthGate>
    );
  };
}

export default AuthGate;
