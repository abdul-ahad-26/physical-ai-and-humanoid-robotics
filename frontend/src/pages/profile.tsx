/**
 * Profile Settings Page
 *
 * Allows authenticated users to view and update their profile settings.
 * Email/password users can set their display name here.
 * OAuth users see their provider-sourced display name (read-only explanation).
 */

import React, { useState, useEffect } from "react";
import { useHistory } from "@docusaurus/router";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { useSession } from "@site/src/lib/auth";
import styles from "./login.module.css";

interface ProfileData {
  id: string;
  email: string;
  display_name: string | null;
  auth_provider: string;
}

export default function ProfilePage(): JSX.Element {
  const { data: session, isPending } = useSession();
  const history = useHistory();
  const [displayName, setDisplayName] = useState("");
  const [originalDisplayName, setOriginalDisplayName] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState<ProfileData | null>(null);

  // Get API URL from Docusaurus context
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || "http://localhost:8000";

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isPending && !session) {
      history.push("/login");
    }
  }, [isPending, session, history]);

  // Load profile data
  useEffect(() => {
    const loadProfile = async () => {
      if (!session) return;

      try {
        const response = await fetch(`${apiUrl}/api/auth/profile`, {
          credentials: "include",
        });

        if (response.ok) {
          const data = await response.json();
          setProfile(data.user);
          setDisplayName(data.user.display_name || "");
          setOriginalDisplayName(data.user.display_name || "");
        }
      } catch (err) {
        console.error("Failed to load profile:", err);
      }
    };

    loadProfile();
  }, [session, apiUrl]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (displayName.trim().length === 0) {
      setError("Display name cannot be empty");
      return;
    }

    if (displayName.trim().length > 100) {
      setError("Display name must be 100 characters or less");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/api/auth/profile`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ display_name: displayName.trim() }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error?.message || "Failed to update profile");
      }

      const data = await response.json();
      setProfile(data.user);
      setOriginalDisplayName(data.user.display_name || "");
      setSuccess("Profile updated successfully! Refresh to see changes in navbar.");
    } catch (err: any) {
      setError(err.message || "Failed to update profile");
    } finally {
      setLoading(false);
    }
  };

  const hasChanges = displayName.trim() !== originalDisplayName;

  if (isPending) {
    return (
      <div className={styles.container}>
        <div className={styles.card}>
          <p style={{ textAlign: "center", color: "#666" }}>Loading...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return null; // Will redirect
  }

  const isOAuthUser = profile?.auth_provider && profile.auth_provider !== "email";

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Profile Settings</h1>
        <p className={styles.subtitle}>Manage your account information</p>

        {/* Account Info */}
        <div style={{ marginBottom: "1.5rem" }}>
          <div
            style={{
              padding: "1rem",
              backgroundColor: "#f8f9fa",
              borderRadius: "8px",
              marginBottom: "1rem",
            }}
          >
            <div style={{ marginBottom: "0.5rem" }}>
              <strong style={{ color: "#333" }}>Email:</strong>{" "}
              <span style={{ color: "#666" }}>{profile?.email || session.user?.email}</span>
            </div>
            <div>
              <strong style={{ color: "#333" }}>Sign-in method:</strong>{" "}
              <span
                style={{
                  display: "inline-block",
                  padding: "0.125rem 0.5rem",
                  backgroundColor:
                    profile?.auth_provider === "google"
                      ? "#ea4335"
                      : profile?.auth_provider === "github"
                      ? "#24292f"
                      : "#10B981",
                  color: "white",
                  borderRadius: "4px",
                  fontSize: "0.75rem",
                  textTransform: "capitalize",
                }}
              >
                {profile?.auth_provider || "email"}
              </span>
            </div>
          </div>
        </div>

        {/* Display Name Form */}
        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label htmlFor="displayName">Display Name</label>
            <input
              id="displayName"
              type="text"
              placeholder="Enter your display name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              maxLength={100}
              disabled={loading}
            />
            <p
              style={{
                fontSize: "0.8rem",
                color: "#666",
                marginTop: "0.25rem",
              }}
            >
              This name will be shown in the navbar when you're logged in.
              {isOAuthUser && (
                <span style={{ display: "block", marginTop: "0.25rem" }}>
                  Your name was imported from {profile?.auth_provider}, but you can change it here.
                </span>
              )}
            </p>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          {success && (
            <div
              style={{
                marginBottom: "1rem",
                padding: "0.875rem 1rem",
                background: "#d4edda",
                border: "1px solid #c3e6cb",
                borderLeft: "4px solid #28a745",
                borderRadius: "8px",
                color: "#155724",
                fontSize: "0.9rem",
              }}
            >
              {success}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !hasChanges}
            className={styles.button}
            style={{
              opacity: loading || !hasChanges ? 0.6 : 1,
              cursor: loading || !hasChanges ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Saving..." : "Save Changes"}
          </button>
        </form>

        <p className={styles.link} style={{ marginTop: "2rem" }}>
          <a href="/">Back to Home</a>
        </p>
      </div>
    </div>
  );
}
