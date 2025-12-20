/**
 * Signup Page
 *
 * Allows new users to create an account with email/password or OAuth (Google/GitHub).
 * On successful signup, user is automatically logged in and redirected to homepage.
 */

import React, { useState, useEffect } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { useSignUp } from "@site/src/lib/auth";
import styles from "./login.module.css";

export default function SignupPage(): JSX.Element {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState<string | null>(null);
  const signUp = useSignUp();

  // Get API URL from Docusaurus context
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || "http://localhost:8000";

  // Handle OAuth errors from URL params (redirect back from failed OAuth)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const oauthError = params.get("error");
    const oauthMessage = params.get("message");

    if (oauthError) {
      setError(oauthMessage || "OAuth sign-up failed. Please try again.");
      // Clean URL without refreshing
      window.history.replaceState({}, "", "/signup");
    }
  }, []);

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return emailRegex.test(email);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Client-side validation
    if (!validateEmail(email)) {
      setError("Please enter a valid email address");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    setLoading(true);

    try {
      await signUp.email({ email, password });
      // Redirect to homepage on success
      window.location.href = "/";
    } catch (err: any) {
      // Handle error from API
      if (err.response?.data?.error) {
        setError(err.response.data.error.message);
      } else {
        setError(err.message || "Signup failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    setError("");
    setOauthLoading("google");
    // Redirect to backend OAuth initiation endpoint
    window.location.href = `${apiUrl}/api/auth/oauth/google`;
  };

  const handleGitHubSignIn = () => {
    setError("");
    setOauthLoading("github");
    // Redirect to backend OAuth initiation endpoint
    window.location.href = `${apiUrl}/api/auth/oauth/github`;
  };

  const isLoading = loading || oauthLoading !== null;

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Sign Up</h1>
        <p className={styles.subtitle}>Create an account to use the AI chatbot</p>

        {/* OAuth Buttons */}
        <div className={styles.oauthButtons}>
          <button
            type="button"
            onClick={handleGoogleSignIn}
            disabled={isLoading}
            className={`${styles.oauthButton} ${styles.googleButton}`}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" className={styles.oauthIcon}>
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            {oauthLoading === "google" ? "Redirecting..." : "Continue with Google"}
          </button>

          <button
            type="button"
            onClick={handleGitHubSignIn}
            disabled={isLoading}
            className={`${styles.oauthButton} ${styles.githubButton}`}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" className={styles.oauthIcon}>
              <path
                fill="currentColor"
                d="M12 1.27a11 11 0 00-3.48 21.46c.55.09.73-.28.73-.55v-1.84c-3.03.64-3.67-1.46-3.67-1.46-.55-1.29-1.28-1.65-1.28-1.65-.92-.65.1-.65.1-.65 1.1 0 1.73 1.1 1.73 1.1.92 1.65 2.57 1.2 3.21.92a2 2 0 01.64-1.47c-2.47-.27-5.04-1.19-5.04-5.5 0-1.1.46-2.1 1.2-2.84a3.76 3.76 0 010-2.93s.91-.28 3.11 1.1c1.8-.49 3.7-.49 5.5 0 2.1-1.38 3.02-1.1 3.02-1.1a3.76 3.76 0 010 2.93c.83.74 1.2 1.74 1.2 2.84 0 4.32-2.58 5.23-5.04 5.5.45.37.82.92.82 2.02v3.03c0 .27.1.64.73.55A11 11 0 0012 1.27"
              />
            </svg>
            {oauthLoading === "github" ? "Redirecting..." : "Continue with GitHub"}
          </button>
        </div>

        {/* Divider */}
        <div className={styles.divider}>
          <span>or sign up with email</span>
        </div>

        {/* Email/Password Form */}
        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              disabled={isLoading}
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Minimum 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              autoComplete="new-password"
              disabled={isLoading}
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" disabled={isLoading} className={styles.button}>
            {loading ? "Creating account..." : "Sign Up"}
          </button>

          <p className={styles.link}>
            Already have an account? <a href="/login">Log in</a>
          </p>
        </form>
      </div>
    </div>
  );
}
