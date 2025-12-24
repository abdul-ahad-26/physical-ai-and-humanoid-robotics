import React from 'react';
import { useSession, useSignOut } from '../../lib/auth';
import { useHistory } from '@docusaurus/router';

export default function AuthButton() {
  const { data: session, isPending } = useSession();
  const history = useHistory();
  const signOut = useSignOut();

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
    // Get display name with fallback to email
    const displayName = session.user?.display_name || session.user?.email || 'User';

    // Truncate long names (>20 chars) with ellipsis
    const truncatedName = displayName.length > 20
      ? displayName.substring(0, 17) + '...'
      : displayName;

    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <a
          href="/profile"
          style={{
            fontSize: '0.875rem',
            color: 'var(--ifm-navbar-link-color)',
            marginRight: '0.5rem',
            textDecoration: 'none',
            cursor: 'pointer',
          }}
          title={`${displayName} - Click to edit profile`}
          onMouseEnter={(e) => {
            e.currentTarget.style.textDecoration = 'underline';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.textDecoration = 'none';
          }}
        >
          Welcome, {truncatedName}
        </a>
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
        backgroundColor: '#2d7a4e',
        color: 'white',
        border: 'none',
        borderRadius: '0.375rem',
        cursor: 'pointer',
        fontSize: '0.875rem',
        fontWeight: 500,
        transition: 'background-color 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = '#276942';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = '#2d7a4e';
      }}
    >
      Sign In
    </button>
  );
}
