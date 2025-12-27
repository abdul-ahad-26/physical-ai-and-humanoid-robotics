/**
 * Profile Settings Page
 *
 * Allows authenticated users to view and update their profile settings:
 * - Display name
 * - Software background (level, languages, frameworks)
 * - Hardware background (level, domains)
 *
 * Email/password users can set their display name here.
 * OAuth users see their provider-sourced display name (can be changed).
 *
 * Updated for 005-user-personalization feature.
 */

import React, { useState, useEffect } from "react";
import { useHistory } from "@docusaurus/router";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { useSession } from "@site/src/lib/auth";
import {
  getProfile,
  updateProfile,
  UserProfile,
  ProfileUpdateRequest,
} from "@site/src/lib/personalization";
import styles from "./login.module.css";

// Options for dropdowns - same as ProfileWizard
const SOFTWARE_LEVELS = [
  { value: "beginner", label: "Beginner - New to programming" },
  { value: "intermediate", label: "Intermediate - 1-3 years experience" },
  { value: "advanced", label: "Advanced - 3+ years experience" },
];

const PROGRAMMING_LANGUAGES = [
  "Python",
  "JavaScript",
  "TypeScript",
  "C++",
  "C",
  "Java",
  "Go",
  "Rust",
  "Ruby",
  "Other",
];

const FRAMEWORKS = [
  "ROS/ROS2",
  "PyTorch",
  "TensorFlow",
  "React",
  "FastAPI",
  "Django",
  "Express",
  "Unity",
  "Other",
];

const HARDWARE_LEVELS = [
  { value: "none", label: "No Experience" },
  { value: "basic", label: "Basic - Hobbyist level" },
  { value: "intermediate", label: "Intermediate - Some projects" },
  { value: "advanced", label: "Advanced - Professional experience" },
];

const HARDWARE_DOMAINS = [
  "Arduino/Microcontrollers",
  "Raspberry Pi",
  "Robotics",
  "Embedded Systems",
  "PCB Design",
  "Sensors/Actuators",
  "3D Printing",
  "Other",
];

export default function ProfilePage(): JSX.Element {
  const { data: session, isPending } = useSession();
  const history = useHistory();

  // Get API URL from Docusaurus context
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || "http://localhost:8000";

  // State
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Profile data
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [originalDisplayName, setOriginalDisplayName] = useState("");

  // Software background
  const [softwareLevel, setSoftwareLevel] = useState("");
  const [languages, setLanguages] = useState<string[]>([]);
  const [frameworks, setFrameworks] = useState<string[]>([]);

  // Hardware background
  const [hardwareLevel, setHardwareLevel] = useState("");
  const [domains, setDomains] = useState<string[]>([]);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isPending && !session) {
      history.push("/login");
    }
  }, [isPending, session, history]);

  // Load profile data
  useEffect(() => {
    const loadProfile = async () => {
      if (!session || isPending) return;

      try {
        setLoading(true);
        const profileData = await getProfile(apiUrl);
        setProfile(profileData);

        // Set form values
        setDisplayName(profileData.display_name || "");
        setOriginalDisplayName(profileData.display_name || "");

        if (profileData.software_background) {
          setSoftwareLevel(profileData.software_background.level || "");
          setLanguages(profileData.software_background.languages || []);
          setFrameworks(profileData.software_background.frameworks || []);
        }

        if (profileData.hardware_background) {
          setHardwareLevel(profileData.hardware_background.level || "");
          setDomains(profileData.hardware_background.domains || []);
        }
      } catch (err: any) {
        console.error("Failed to load profile:", err);
        setError(err.message || "Failed to load profile");
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [session, isPending, apiUrl]);

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    // Validation
    if (!displayName.trim()) {
      setError("Display name is required");
      return;
    }
    if (displayName.length > 100) {
      setError("Display name must be 100 characters or less");
      return;
    }
    if (!softwareLevel) {
      setError("Please select your software experience level");
      return;
    }
    if (!hardwareLevel) {
      setError("Please select your hardware experience level");
      return;
    }

    setSaving(true);

    try {
      const request: ProfileUpdateRequest = {
        display_name: displayName.trim(),
        software_background: {
          level: softwareLevel as "beginner" | "intermediate" | "advanced",
          languages,
          frameworks,
        },
        hardware_background: {
          level: hardwareLevel as "none" | "basic" | "intermediate" | "advanced",
          domains,
        },
      };

      const response = await updateProfile(request, apiUrl);
      setProfile(response.user);
      setOriginalDisplayName(response.user.display_name || "");
      setSuccess("Profile updated successfully!");

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      setError(err.message || "Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  // Handle multi-select toggle
  const toggleSelection = (
    item: string,
    current: string[],
    setter: React.Dispatch<React.SetStateAction<string[]>>
  ) => {
    if (current.includes(item)) {
      setter(current.filter((i) => i !== item));
    } else {
      setter([...current, item]);
    }
  };

  // Check if form has changes
  const hasChanges =
    displayName.trim() !== originalDisplayName ||
    softwareLevel !== (profile?.software_background?.level || "") ||
    hardwareLevel !== (profile?.hardware_background?.level || "") ||
    JSON.stringify(languages.sort()) !==
      JSON.stringify((profile?.software_background?.languages || []).sort()) ||
    JSON.stringify(frameworks.sort()) !==
      JSON.stringify((profile?.software_background?.frameworks || []).sort()) ||
    JSON.stringify(domains.sort()) !==
      JSON.stringify((profile?.hardware_background?.domains || []).sort());

  if (isPending || loading) {
    return (
      <div className={styles.container}>
        <div className={styles.card} style={{ maxWidth: "700px" }}>
          <p className={styles.infoValue} style={{ textAlign: "center" }}>Loading...</p>
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
      <div className={styles.card} style={{ maxWidth: "700px" }}>
        <h1>Profile Settings</h1>
        <p className={styles.subtitle}>
          Update your profile to personalize your learning experience
        </p>

        {/* Account Info */}
        <div className={styles.infoBox}>
          <div style={{ marginBottom: "0.5rem" }}>
            <strong className={styles.infoLabel}>Email:</strong>{" "}
            <span className={styles.infoValue}>{profile?.email || session.user?.email}</span>
          </div>
          <div style={{ marginBottom: "0.5rem" }}>
            <strong className={styles.infoLabel}>Sign-in method:</strong>{" "}
            <span
              style={{
                display: "inline-block",
                padding: "0.125rem 0.5rem",
                backgroundColor:
                  profile?.auth_provider === "google"
                    ? "#ea4335"
                    : profile?.auth_provider === "github"
                    ? "#24292f"
                    : "#2d7a4e",
                color: "white",
                borderRadius: "4px",
                fontSize: "0.75rem",
                textTransform: "capitalize",
              }}
            >
              {profile?.auth_provider || "email"}
            </span>
          </div>
          <div>
            <strong className={styles.infoLabel}>Profile Status:</strong>{" "}
            <span
              style={{
                display: "inline-block",
                padding: "0.125rem 0.5rem",
                backgroundColor: profile?.profile_completed ? "#2d7a4e" : "#f59e0b",
                color: "white",
                borderRadius: "4px",
                fontSize: "0.75rem",
              }}
            >
              {profile?.profile_completed ? "Complete" : "Incomplete"}
            </span>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Display Name */}
          <div className={styles.field}>
            <label htmlFor="displayName">Display Name *</label>
            <input
              id="displayName"
              type="text"
              placeholder="Enter your display name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              maxLength={100}
              disabled={saving}
            />
            <p className={styles.infoValue} style={{ fontSize: "0.8rem", marginTop: "0.25rem" }}>
              {displayName.length}/100 characters
              {isOAuthUser && (
                <span style={{ display: "block", marginTop: "0.25rem" }}>
                  Your name was imported from {profile?.auth_provider}, but you can change it.
                </span>
              )}
            </p>
          </div>

          {/* Software Background Section */}
          <div className={styles.infoBox} style={{ padding: "1.5rem" }}>
            <h3 className={styles.sectionTitle} style={{ marginTop: 0 }}>Software Background</h3>

            {/* Software Level */}
            <div className={styles.field}>
              <label htmlFor="softwareLevel">Experience Level *</label>
              <select
                id="softwareLevel"
                value={softwareLevel}
                onChange={(e) => setSoftwareLevel(e.target.value)}
                disabled={saving}
                className={styles.select}
              >
                <option value="">Select your level...</option>
                {SOFTWARE_LEVELS.map((level) => (
                  <option key={level.value} value={level.value}>
                    {level.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Languages */}
            <div style={{ marginBottom: "1rem" }}>
              <label className={styles.infoLabel} style={{ display: "block", marginBottom: "0.5rem" }}>
                Programming Languages
              </label>
              <div className={styles.chipContainer}>
                {PROGRAMMING_LANGUAGES.map((lang) => (
                  <button
                    key={lang}
                    type="button"
                    onClick={() => toggleSelection(lang, languages, setLanguages)}
                    disabled={saving}
                    className={`${styles.chip} ${languages.includes(lang) ? styles.chipSelected : ''}`}
                    style={{ opacity: saving ? 0.7 : 1 }}
                  >
                    {lang}
                  </button>
                ))}
              </div>
            </div>

            {/* Frameworks */}
            <div>
              <label className={styles.infoLabel} style={{ display: "block", marginBottom: "0.5rem" }}>
                Frameworks & Tools
              </label>
              <div className={styles.chipContainer}>
                {FRAMEWORKS.map((fw) => (
                  <button
                    key={fw}
                    type="button"
                    onClick={() => toggleSelection(fw, frameworks, setFrameworks)}
                    disabled={saving}
                    className={`${styles.chip} ${frameworks.includes(fw) ? styles.chipSelected : ''}`}
                    style={{ opacity: saving ? 0.7 : 1 }}
                  >
                    {fw}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Hardware Background Section */}
          <div className={styles.infoBox} style={{ padding: "1.5rem" }}>
            <h3 className={styles.sectionTitle} style={{ marginTop: 0 }}>Hardware Background</h3>

            {/* Hardware Level */}
            <div className={styles.field}>
              <label htmlFor="hardwareLevel">Experience Level *</label>
              <select
                id="hardwareLevel"
                value={hardwareLevel}
                onChange={(e) => setHardwareLevel(e.target.value)}
                disabled={saving}
                className={styles.select}
              >
                <option value="">Select your level...</option>
                {HARDWARE_LEVELS.map((level) => (
                  <option key={level.value} value={level.value}>
                    {level.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Domains */}
            <div>
              <label className={styles.infoLabel} style={{ display: "block", marginBottom: "0.5rem" }}>
                Hardware Domains
              </label>
              <div className={styles.chipContainer}>
                {HARDWARE_DOMAINS.map((domain) => (
                  <button
                    key={domain}
                    type="button"
                    onClick={() => toggleSelection(domain, domains, setDomains)}
                    disabled={saving}
                    className={`${styles.chip} ${domains.includes(domain) ? styles.chipSelected : ''}`}
                    style={{ opacity: saving ? 0.7 : 1 }}
                  >
                    {domain}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && <div className={styles.error}>{error}</div>}

          {/* Success Message */}
          {success && (
            <div className={styles.successMessage}>
              {success}
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: "flex", gap: "1rem", justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={() => history.push("/")}
              className={styles.buttonSecondary}
              style={{ width: "auto", padding: "0.75rem 1.5rem" }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving || !hasChanges}
              className={styles.button}
              style={{
                opacity: saving || !hasChanges ? 0.6 : 1,
                cursor: saving || !hasChanges ? "not-allowed" : "pointer",
              }}
            >
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        </form>

        <p className={styles.link} style={{ marginTop: "2rem", textAlign: "center" }}>
          <a href="/">Back to Home</a>
        </p>
      </div>
    </div>
  );
}
