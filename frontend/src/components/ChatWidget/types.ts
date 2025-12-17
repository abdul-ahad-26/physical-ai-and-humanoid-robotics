/**
 * Type definitions for the ChatWidget components.
 */

/**
 * Citation reference to a textbook section.
 */
export interface Citation {
  chapter_id: string;
  section_id: string;
  anchor_url: string;
  display_text: string;
}

/**
 * Chat message structure.
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at: string;
}

/**
 * Chat request payload.
 */
export interface ChatRequest {
  message: string;
  session_id?: string | null;
  selected_text?: string | null;
}

/**
 * Chat response payload.
 */
export interface ChatResponse {
  answer: string;
  citations: Citation[];
  session_id: string;
  found_relevant_content: boolean;
  latency_ms: number;
}

/**
 * Session summary for history list.
 */
export interface SessionSummary {
  id: string;
  created_at: string;
  last_activity: string;
  is_active: boolean;
}

/**
 * User session from Better Auth.
 */
export interface UserSession {
  user: {
    id: string;
    email: string;
    name?: string;
    image?: string;
  };
  expires: string;
}

/**
 * ChatKit theme configuration.
 */
export interface ChatKitTheme {
  colorScheme: 'light' | 'dark';
  radius?: 'round' | 'square';
  color?: {
    accent?: {
      primary?: string;
      level?: number;
    };
  };
}

/**
 * ChatKit API configuration.
 */
export interface ChatKitApi {
  url: string;
  domainKey: string;
}

/**
 * ChatKit options for useChatKit hook.
 */
export interface ChatKitOptions {
  api: ChatKitApi;
  theme?: ChatKitTheme;
  startScreen?: {
    greeting?: string;
    prompts?: Array<{
      label: string;
      prompt: string;
      icon?: string;
    }>;
  };
  composer?: {
    placeholder?: string;
  };
  history?: {
    enabled?: boolean;
    showDelete?: boolean;
    showRename?: boolean;
  };
  header?: {
    enabled?: boolean;
    rightAction?: {
      icon: string;
      onClick: () => void;
    };
  };
}

/**
 * Error response from API.
 */
export interface ApiError {
  detail: string;
  status_code?: number;
}

/**
 * Rate limit error response.
 */
export interface RateLimitError extends ApiError {
  retry_after?: number;
}
