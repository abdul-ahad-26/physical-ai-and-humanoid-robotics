/**
 * Login Page
 *
 * Allows existing users to log in with email and password.
 * On successful login, user is redirected to homepage.
 */

import React, { useState } from "react";
import { signIn } from "@site/src/lib/auth";
import styles from "./login.module.css";

export default function LoginPage(): JSX.Element {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await signIn.email({ email, password });
      // Redirect to homepage on success
      window.location.href = "/";
    } catch (err: any) {
      // Handle error from API
      if (err.response?.data?.error) {
        setError(err.response.data.error.message);
      } else {
        setError(err.message || "Invalid email or password");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Log In</h1>
        <p className={styles.subtitle}>Sign in to access the AI chatbot</p>

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
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" disabled={loading} className={styles.button}>
            {loading ? "Logging in..." : "Log In"}
          </button>

          <p className={styles.link}>
            Don't have an account? <a href="/signup">Sign up</a>
          </p>
        </form>
      </div>
    </div>
  );
}
