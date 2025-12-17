/**
 * Root.tsx - Docusaurus Root theme component with ChatWidget.
 *
 * The ChatWidget is always visible (icon shown to all users).
 * Unauthenticated users see a login prompt when they click it.
 * Authenticated users get the full chat experience.
 */

import React, { ReactNode } from 'react';
import { ChatWidget } from '../components/ChatWidget';

interface RootProps {
  children: ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  // TODO: Replace with actual Better Auth session check
  // For now, pass isAuthenticated=false to show login prompt
  // When Better Auth is configured, use:
  // const { data: session } = useSession();
  // <ChatWidget isAuthenticated={!!session} session={session} />

  return (
    <>
      {children}
      <ChatWidget isAuthenticated={false} />
    </>
  );
}
