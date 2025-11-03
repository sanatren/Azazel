/**
 * Configuration file for Azazel AI frontend
 */

const CONFIG = {
    // API URL - will be set based on environment
    // For Railway deployment, set BACKEND_URL in HTML or use environment variable
    API_URL: window.BACKEND_URL || (window.location.hostname === 'localhost'
        ? 'http://localhost:8000'
        : 'https://your-backend-name.railway.app'),  // Replace with your Railway backend URL

    // Default settings
    DEFAULT_MODEL: 'gpt-4o-mini',
    DEFAULT_TEMPERATURE: 0.7,

    // API endpoints
    ENDPOINTS: {
        CHAT_MESSAGE: '/api/chat/message',
        CHAT_STREAM: '/api/chat/stream',
        CODE_EXECUTE: '/api/chat/code',
        DOCUMENT_UPLOAD: '/api/documents/upload',
        DOCUMENT_QUERY: '/api/documents/query',
        DOCUMENT_CLEAR: '/api/documents/clear',
        SEARCH_WEB: '/api/search/web',
        SEARCH_DETERMINE: '/api/search/determine-need',
        SESSION_CREATE: '/api/sessions/create',
        SESSION_HISTORY: '/api/sessions/history',
    },

    // LocalStorage keys
    STORAGE_KEYS: {
        API_KEY: 'azazel_api_key',
        MODEL: 'azazel_model',
        TEMPERATURE: 'azazel_temperature',
        SESSION_ID: 'azazel_session_id',
        MESSAGES: 'azazel_messages',
    },

    // Features
    FEATURES: {
        WEB_SEARCH: 'web_search',
        CODE_EXECUTION: 'code_execution',
        DOCUMENT_MODE: 'document_mode',
    }
};

// Export for use in other files
window.CONFIG = CONFIG;