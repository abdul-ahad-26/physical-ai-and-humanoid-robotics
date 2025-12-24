/**
 * Root.tsx - Docusaurus Root theme component with ChatWidget and ProfileWizard.
 *
 * The ChatWidget is only visible on docs/book pages (not homepage, login, signup, etc).
 * Unauthenticated users see a login prompt when they click it.
 * Authenticated users get the full chat experience.
 *
 * ProfileWizard shows when user is authenticated but profile_completed=false.
 */

import React, { ReactNode, useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import { ChatWidget } from '../components/ChatWidget';
import ProfileWizard from '../components/ProfileWizard';
import { useSession } from '@site/src/lib/auth';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

interface RootProps {
  children: ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  const { data: session, isPending } = useSession();
  const location = useLocation();
  const [showProfileWizard, setShowProfileWizard] = useState(false);
  const [profileChecked, setProfileChecked] = useState(false);

  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  // Determine if we should show the ChatWidget
  // Show only on docs/book pages (not homepage, login, signup, profile pages)
  const isDocPage = location.pathname.startsWith('/docs/') ||
                    location.pathname.startsWith('/book/');
  const shouldShowChat = isDocPage;

  // Check if profile is complete when user logs in
  useEffect(() => {
    const checkProfile = async () => {
      if (session && !isPending && !profileChecked) {
        try {
          const response = await fetch(`${apiUrl}/api/profile`, {
            credentials: 'include',
          });

          if (response.ok) {
            const data = await response.json();
            // Show wizard if profile not completed
            if (!data.user.profile_completed) {
              setShowProfileWizard(true);
            }
          }
        } catch (error) {
          console.error('Failed to check profile status:', error);
        }
        setProfileChecked(true);
      }
    };

    checkProfile();
  }, [session, isPending, profileChecked, apiUrl]);

  // Reset profile check when user logs out
  useEffect(() => {
    if (!session && !isPending) {
      setProfileChecked(false);
      setShowProfileWizard(false);
    }
  }, [session, isPending]);

  const handleProfileComplete = () => {
    setShowProfileWizard(false);
    // Reload to update navbar with new display name
    window.location.reload();
  };

  return (
    <>
      {children}
      {shouldShowChat && (
        <ChatWidget
          isAuthenticated={!!session && !isPending}
          session={session}
        />
      )}
      {session && !isPending && (
        <ProfileWizard
          isOpen={showProfileWizard}
          onClose={() => setShowProfileWizard(false)}
          onComplete={handleProfileComplete}
          initialDisplayName={session.user?.display_name || ''}
        />
      )}
    </>
  );
}
