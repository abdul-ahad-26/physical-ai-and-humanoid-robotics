/**
 * Root.tsx - Docusaurus Root theme component with ChatWidget.
 *
 * The ChatWidget is always visible (icon shown to all users).
 * Unauthenticated users see a login prompt when they click it.
 * Authenticated users get the full chat experience.
 */

import React, { ReactNode } from 'react';
import { ChatWidget } from '../components/ChatWidget';
import { useSession } from '@site/src/lib/auth';

interface RootProps {
  children: ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  const { data: session, isPending } = useSession();

  return (
    <>
      {children}
      <ChatWidget
        isAuthenticated={!!session && !isPending}
        session={session}
      />
    </>
  );
}
