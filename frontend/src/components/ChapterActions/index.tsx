/**
 * ChapterActions Component
 *
 * Provides "Personalize Content" and "Translate to Urdu" toggle buttons
 * for chapter pages. Shows loading states, error handling, and login prompts.
 *
 * Features:
 * - Personalize Content: AI-adapts content based on user's technical background
 * - Translate to Urdu: Translates prose to Urdu with RTL support (preserves code)
 * - Caching: Results are cached server-side for quick toggling
 * - Rate limiting: 10 requests per minute per user
 */

import React, { useState, useCallback, useRef } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useSession } from '@site/src/lib/auth';
import {
  personalizeContent,
  translateContent,
  PersonalizeResponse,
  TranslateResponse,
} from '@site/src/lib/personalization';
import styles from './styles.module.css';

interface ChapterActionsProps {
  /** Unique identifier for the chapter */
  chapterId: string;
  /** Original chapter content (markdown) */
  content: string;
  /** Chapter title */
  title: string;
  /** Callback when content changes (personalized or translated) */
  onContentChange: (newContent: string, type: 'original' | 'personalized' | 'translated') => void;
  /** Whether user has completed their profile */
  profileCompleted?: boolean;
}

type ContentState = 'original' | 'personalized' | 'translated';
type LoadingState = 'idle' | 'personalizing' | 'translating';

export default function ChapterActions({
  chapterId,
  content,
  title,
  onContentChange,
  profileCompleted = false,
}: ChapterActionsProps) {
  const { data: session, isPending } = useSession();
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  // State
  const [contentState, setContentState] = useState<ContentState>('original');
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [error, setError] = useState<string | null>(null);

  // Cache the results locally for quick toggling
  const personalizedContentRef = useRef<string | null>(null);
  const translatedContentRef = useRef<string | null>(null);
  const originalContentRef = useRef<string>(content);

  // Debounce ref to prevent rapid clicks
  const lastClickRef = useRef<number>(0);
  const DEBOUNCE_MS = 500;

  const isDebounced = useCallback(() => {
    const now = Date.now();
    if (now - lastClickRef.current < DEBOUNCE_MS) {
      return true;
    }
    lastClickRef.current = now;
    return false;
  }, []);

  /**
   * Handle Personalize Content toggle
   */
  const handlePersonalize = useCallback(async () => {
    if (isDebounced()) return;
    setError(null);

    // If already personalized, toggle back to original
    if (contentState === 'personalized') {
      setContentState('original');
      onContentChange(originalContentRef.current, 'original');
      return;
    }

    // If we have cached personalized content, use it
    if (personalizedContentRef.current) {
      setContentState('personalized');
      onContentChange(personalizedContentRef.current, 'personalized');
      return;
    }

    // Call the API
    setLoadingState('personalizing');
    try {
      const response: PersonalizeResponse = await personalizeContent(
        {
          chapter_id: chapterId,
          content: originalContentRef.current,
          title,
        },
        apiUrl
      );

      personalizedContentRef.current = response.personalized_content;
      setContentState('personalized');
      onContentChange(response.personalized_content, 'personalized');
    } catch (err: any) {
      setError(err.message || 'Failed to personalize content');
    } finally {
      setLoadingState('idle');
    }
  }, [chapterId, content, title, apiUrl, contentState, onContentChange, isDebounced]);

  /**
   * Handle Translate to Urdu toggle
   */
  const handleTranslate = useCallback(async () => {
    if (isDebounced()) return;
    setError(null);

    // If already translated, toggle back to original
    if (contentState === 'translated') {
      setContentState('original');
      onContentChange(originalContentRef.current, 'original');
      return;
    }

    // If we have cached translated content, use it
    if (translatedContentRef.current) {
      setContentState('translated');
      onContentChange(translatedContentRef.current, 'translated');
      return;
    }

    // Call the API
    setLoadingState('translating');
    try {
      const response: TranslateResponse = await translateContent(
        {
          chapter_id: chapterId,
          content: originalContentRef.current,
          target_language: 'ur',
        },
        apiUrl
      );

      translatedContentRef.current = response.translated_content;
      setContentState('translated');
      onContentChange(response.translated_content, 'translated');
    } catch (err: any) {
      setError(err.message || 'Failed to translate content');
    } finally {
      setLoadingState('idle');
    }
  }, [chapterId, content, apiUrl, contentState, onContentChange, isDebounced]);

  // Show loading state while checking session
  if (isPending) {
    return null;
  }

  // Not authenticated - show login prompt
  if (!session) {
    return (
      <div className={styles.chapterActions}>
        <div className={styles.loginPrompt}>
          <span>Log in to personalize content for your experience level</span>
        </div>
      </div>
    );
  }

  // Authenticated but profile not completed - show warning (only for personalize)
  const showProfileWarning = !profileCompleted;

  return (
    <div className={styles.chapterActions}>
      {/* Personalize Button */}
      <button
        type="button"
        className={`${styles.actionButton} ${contentState === 'personalized' ? styles.active : ''}`}
        onClick={handlePersonalize}
        disabled={loadingState !== 'idle' || showProfileWarning}
        title={
          showProfileWarning
            ? 'Complete your profile to personalize content'
            : contentState === 'personalized'
            ? 'Show original content'
            : 'Personalize content for your experience level'
        }
      >
        {loadingState === 'personalizing' ? (
          <>
            <span className={styles.spinner} />
            Personalizing...
          </>
        ) : contentState === 'personalized' ? (
          'Show Original'
        ) : (
          'Personalize Content'
        )}
      </button>

      {/* Translate Button */}
      <button
        type="button"
        className={`${styles.actionButton} ${styles.translateButton} ${
          contentState === 'translated' ? styles.active : ''
        }`}
        onClick={handleTranslate}
        disabled={loadingState !== 'idle'}
        title={
          contentState === 'translated'
            ? 'Show English content'
            : 'Translate content to Urdu'
        }
      >
        {loadingState === 'translating' ? (
          <>
            <span className={styles.spinner} />
            Translating...
          </>
        ) : contentState === 'translated' ? (
          'Show English'
        ) : (
          'Translate to Urdu'
        )}
      </button>

      {/* Profile Warning */}
      {showProfileWarning && (
        <div className={styles.profileWarning}>
          <span>
            Complete your <a href="/profile">profile</a> to enable personalization
          </span>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className={`${styles.statusMessage} ${styles.error}`}>
          {error}
        </div>
      )}
    </div>
  );
}
