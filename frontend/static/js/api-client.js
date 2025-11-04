/**
 * API Client for Azazel AI
 */

class AzazelAPIClient {
    constructor() {
        this.baseURL = CONFIG.API_URL;
    }

    /**
     * Send a chat message
     */
    async sendMessage(message, sessionId, apiKey, options = {}) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.CHAT_MESSAGE}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                session_id: sessionId,
                api_key: apiKey,
                language: options.language || 'English',
                personality: options.personality || null,
                use_rag: options.useRAG || false,
                force_search: options.forceSearch || false,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Stream a chat response
     */
    async* streamMessage(message, sessionId, apiKey, chatHistory = [], options = {}) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.CHAT_STREAM}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                session_id: sessionId,
                api_key: apiKey,
                language: options.language || 'English',
                personality: options.personality || null,
                chat_history: chatHistory,
                use_rag: options.useRAG || false,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        return;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.content) {
                            yield parsed.content;
                        } else if (parsed.error) {
                            throw new Error(parsed.error);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
    }

    /**
     * Execute code
     */
    async executeCode(question, sessionId, apiKey, chatHistory = [], options = {}) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.CODE_EXECUTE}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question,
                session_id: sessionId,
                api_key: apiKey,
                language: options.language || 'English',
                personality: options.personality || null,
                chat_history: chatHistory,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Upload document
     */
    async uploadDocument(file, sessionId, apiKey) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);
        formData.append('api_key', apiKey);

        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.DOCUMENT_UPLOAD}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(`API error: ${errorData.detail || response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Query documents
     */
    async queryDocuments(query, sessionId, apiKey, k = 8) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.DOCUMENT_QUERY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                session_id: sessionId,
                api_key: apiKey,
                k,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Clear documents
     */
    async clearDocuments(sessionId) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.DOCUMENT_CLEAR}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Perform web search
     */
    async searchWeb(query, sessionId, apiKey, chatHistory = [], language = 'English') {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.SEARCH_WEB}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                session_id: sessionId,
                api_key: apiKey,
                language,
                chat_history: chatHistory,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Create a new session
     */
    async createSession(userId = null) {
        const response = await fetch(`${this.baseURL}${CONFIG.ENDPOINTS.SESSION_CREATE}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            return response.ok;
        } catch (e) {
            return false;
        }
    }
}

// Export the client
window.AzazelAPIClient = AzazelAPIClient;