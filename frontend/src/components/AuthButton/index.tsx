import React from 'react';
import { useSession, signOut } from '../../lib/auth';
import { useHistory } from '@docusaurus/router';

export default function AuthButton() {
  const { data: session, isPending } = useSession();
  const history = useHistory();

  const handleSignOut = async () => {
    try {
      await signOut();
      // Redirect to home and reload to clear session state
      window.location.href = '/';
    } catch (error) {
      console.error('Sign out error:', error);
      // Force clear by reloading anyway
      window.location.href = '/';
    }
  };

  const handleSignIn = () => {
    history.push('/login');
  };

  if (isPending) {
    return null;
  }

  if (session) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span
          style={{
            fontSize: '0.875rem',
            color: 'var(--ifm-navbar-link-color)',
            marginRight: '0.5rem',
          }}
        >
          {session.user?.email}
        </span>
        <button
          onClick={handleSignOut}
          style={{
            padding: '0.375rem 0.75rem',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '0.375rem',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 500,
          }}
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={handleSignIn}
      style={{
        padding: '0.375rem 0.75rem',
        backgroundColor: '#10B981',
        color: 'white',
        border: 'none',
        borderRadius: '0.375rem',
        cursor: 'pointer',
        fontSize: '0.875rem',
        fontWeight: 500,
      }}
    >
      Sign In
    </button>
  );
}
