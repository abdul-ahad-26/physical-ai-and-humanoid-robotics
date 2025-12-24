/**
 * DocItem Content Wrapper
 *
 * Automatically adds personalization and translation buttons to all doc pages.
 * Extracts content from the DOM and integrates with ChapterActions.
 */

import React from 'react';
import Content from '@theme-original/DocItem/Content';
import type ContentType from '@theme/DocItem/Content';
import type { WrapperProps } from '@docusaurus/types';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import ChapterActions from '@site/src/components/ChapterActions';
import { useSession } from '@site/src/lib/auth';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import ReactMarkdown from 'react-markdown';

type Props = WrapperProps<typeof ContentType>;

export default function ContentWrapper(props: Props): JSX.Element {
  const doc = useDoc();
  const { data: session, isPending } = useSession();
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  const [profileCompleted, setProfileCompleted] = React.useState<boolean>(false);
  const [modifiedContent, setModifiedContent] = React.useState<string | null>(null);
  const [contentType, setContentType] = React.useState<'original' | 'personalized' | 'translated'>('original');
  const [pageContent, setPageContent] = React.useState<string>('');
  const contentRef = React.useRef<HTMLDivElement>(null);

  const chapterId = doc.metadata.id;
  const title = doc.metadata.title;

  // Extract content from the rendered DOM after it mounts
  React.useEffect(() => {
    // Small delay to ensure content is rendered
    const timer = setTimeout(() => {
      if (contentRef.current) {
        // Get the parent article element
        const article = contentRef.current.closest('article');
        if (article) {
          // Extract text content (this includes headers, paragraphs, code, etc.)
          const text = article.innerText || article.textContent || '';
          if (text && text.length > 100) {
            setPageContent(text);
          }
        }
      }
    }, 300);

    return () => clearTimeout(timer);
  }, []);

  // Check profile status
  React.useEffect(() => {
    const checkProfile = async () => {
      if (session && !isPending) {
        try {
          const response = await fetch(`${apiUrl}/api/profile`, {
            credentials: 'include',
          });
          if (response.ok) {
            const data = await response.json();
            setProfileCompleted(data.user?.profile_completed ?? false);
          }
        } catch (error) {
          console.error('Failed to check profile status:', error);
        }
      }
    };
    checkProfile();
  }, [session, isPending, apiUrl]);

  const handleContentChange = React.useCallback((newContent: string, type: 'original' | 'personalized' | 'translated') => {
    if (type === 'original') {
      setModifiedContent(null);
      setContentType('original');
    } else {
      setModifiedContent(newContent);
      setContentType(type);
    }
  }, []);

  // Show actions on all doc pages (pageContent check ensures we have content)
  const showActions = true;

  return (
    <div ref={contentRef}>
      {showActions && pageContent && (
        <div style={{ marginBottom: '1.5rem' }}>
          <ChapterActions
            chapterId={chapterId}
            content={pageContent}
            title={title}
            onContentChange={handleContentChange}
            profileCompleted={profileCompleted}
          />
        </div>
      )}

      {modifiedContent ? (
        <div
          className={`personalized-content-container ${contentType === 'translated' ? 'translated-content-urdu' : ''}`}
          style={{
            direction: contentType === 'translated' ? 'rtl' : 'ltr',
            fontFamily: contentType === 'translated' ? "'Noto Nastaliq Urdu', serif" : 'inherit',
          }}
        >
          <div className="personalized-content-badge">
            {contentType === 'personalized' ? '✨ Personalized for your level' : '🌍 Translated to Urdu'}
          </div>
          <div className="markdown">
            <ReactMarkdown>{modifiedContent}</ReactMarkdown>
          </div>
        </div>
      ) : (
        <Content {...props} />
      )}
    </div>
  );
}
