/**
 * Main Application Logic for Azazel AI
 */

class AzazelApp {
    constructor() {
        this.apiClient = new AzazelAPIClient();
        this.messages = [];
        this.sessionId = this.generateSessionId();
        this.uploadedFiles = [];

        // Initialize
        this.initializeElements();
        this.loadSettings();
        this.attachEventListeners();
        this.checkAPIHealth();

        // Configure marked.js for markdown
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true,
        });
    }

    initializeElements() {
        // Input elements
        this.apiKeyInput = document.getElementById('api-key-input');
        this.modelSelect = document.getElementById('model-select');
        this.temperatureSlider = document.getElementById('temperature-slider');
        this.tempValue = document.getElementById('temp-value');
        this.messageInput = document.getElementById('message-input');
        this.personalitySelect = document.getElementById('personality-select');
        this.languageSelect = document.getElementById('language-select');

        // Buttons
        this.sendBtn = document.getElementById('send-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
        this.mobileMenuBtn = document.getElementById('mobile-menu-btn');
        this.imageUploadBtn = document.getElementById('image-upload-btn');
        this.audioUploadBtn = document.getElementById('audio-upload-btn');
        this.removeAttachmentBtn = document.getElementById('remove-attachment-btn');

        // File inputs
        this.imageFileInput = document.getElementById('image-file-input');
        this.audioFileInput = document.getElementById('audio-file-input');

        // Toggles
        this.webSearchToggle = document.getElementById('web-search-toggle');
        this.codeExecutionToggle = document.getElementById('code-execution-toggle');
        this.documentModeToggle = document.getElementById('document-mode-toggle');

        // Containers
        this.messagesContainer = document.getElementById('messages-container');
        this.statusText = document.getElementById('status-text');
        this.sidebar = document.getElementById('sidebar');
        this.documentUploadSection = document.getElementById('document-upload-section');
        this.fileUpload = document.getElementById('file-upload');
        this.uploadedFilesDiv = document.getElementById('uploaded-files');
        this.attachmentPreview = document.getElementById('attachment-preview');
        this.attachmentName = document.getElementById('attachment-name');
        this.attachmentIcon = document.getElementById('attachment-icon');

        // State for current attachment
        this.currentAttachment = null;
    }

    attachEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });

        // Settings
        this.apiKeyInput.addEventListener('change', () => this.saveSettings());
        this.modelSelect.addEventListener('change', () => this.saveSettings());
        this.temperatureSlider.addEventListener('input', (e) => {
            this.tempValue.textContent = e.target.value;
            this.saveSettings();
        });
        this.personalitySelect.addEventListener('change', () => this.saveSettings());
        this.languageSelect.addEventListener('change', () => this.saveSettings());

        // Buttons
        this.newChatBtn.addEventListener('click', () => this.newChat());
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.mobileMenuBtn.addEventListener('click', () => this.toggleSidebar());

        // Audio/Image upload buttons
        this.imageUploadBtn.addEventListener('click', () => this.imageFileInput.click());
        this.audioUploadBtn.addEventListener('click', () => this.audioFileInput.click());
        this.removeAttachmentBtn.addEventListener('click', () => this.removeAttachment());

        // File inputs
        this.imageFileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.audioFileInput.addEventListener('change', (e) => this.handleAudioUpload(e));

        // Document mode toggle
        this.documentModeToggle.addEventListener('change', (e) => {
            this.documentUploadSection.classList.toggle('hidden', !e.target.checked);
        });

        // File upload
        this.fileUpload.addEventListener('change', (e) => this.handleFileUpload(e));
    }

    loadSettings() {
        const apiKey = localStorage.getItem(CONFIG.STORAGE_KEYS.API_KEY);
        const model = localStorage.getItem(CONFIG.STORAGE_KEYS.MODEL);
        const temperature = localStorage.getItem(CONFIG.STORAGE_KEYS.TEMPERATURE);

        if (apiKey) this.apiKeyInput.value = apiKey;
        if (model) this.modelSelect.value = model;
        if (temperature) {
            this.temperatureSlider.value = temperature;
            this.tempValue.textContent = temperature;
        }

        // Load messages
        const savedMessages = localStorage.getItem(CONFIG.STORAGE_KEYS.MESSAGES);
        if (savedMessages) {
            this.messages = JSON.parse(savedMessages);
            this.renderMessages();
        }
    }

    saveSettings() {
        localStorage.setItem(CONFIG.STORAGE_KEYS.API_KEY, this.apiKeyInput.value);
        localStorage.setItem(CONFIG.STORAGE_KEYS.MODEL, this.modelSelect.value);
        localStorage.setItem(CONFIG.STORAGE_KEYS.TEMPERATURE, this.temperatureSlider.value);
    }

    saveMessages() {
        localStorage.setItem(CONFIG.STORAGE_KEYS.MESSAGES, JSON.stringify(this.messages));
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    async checkAPIHealth() {
        const healthy = await this.apiClient.healthCheck();
        if (healthy) {
            this.statusText.textContent = 'Ready to chat';
            this.statusText.classList.remove('text-red-500');
            this.statusText.classList.add('text-green-500');
        } else {
            this.statusText.textContent = 'API offline';
            this.statusText.classList.remove('text-green-500');
            this.statusText.classList.add('text-red-500');
        }
    }

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        const apiKey = this.apiKeyInput.value.trim();
        if (!apiKey) {
            alert('Please enter your OpenAI API key');
            return;
        }

        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Add user message
        this.addMessage('user', message);

        // Disable send button
        this.sendBtn.disabled = true;
        this.statusText.textContent = 'Thinking...';

        try {
            // Get features
            const webSearch = this.webSearchToggle.checked;
            const codeExecution = this.codeExecutionToggle.checked;
            const documentMode = this.documentModeToggle.checked;

            let response;
            let assistantMessage = '';

            // Create typing indicator
            const typingId = this.addTypingIndicator();

            if (documentMode && this.uploadedFiles.length > 0) {
                // Use RAG
                response = await this.apiClient.sendMessage(message, this.sessionId, apiKey, {
                    useRAG: true
                });
                assistantMessage = response.response;

            } else if (webSearch) {
                // Use web search
                const chatHistory = this.messages.map(m => ({
                    role: m.role,
                    message: m.content
                }));
                response = await this.apiClient.searchWeb(message, this.sessionId, apiKey, chatHistory);
                assistantMessage = response.answer;

            } else if (codeExecution) {
                // Check if code execution
                const chatHistory = this.messages.map(m => ({
                    role: m.role,
                    message: m.content
                }));
                response = await this.apiClient.executeCode(message, this.sessionId, apiKey, chatHistory);
                assistantMessage = response.answer;

                if (response.code) {
                    assistantMessage += `\n\n\`\`\`python\n${response.code}\n\`\`\``;
                }
                if (response.execution_result) {
                    assistantMessage += `\n\n**Output:**\n\`\`\`\n${response.execution_result}\n\`\`\``;
                }

            } else {
                // Regular streaming chat
                const chatHistory = this.messages.map(m => ({
                    role: m.role,
                    message: m.content
                }));

                assistantMessage = '';
                const messageElement = this.addMessage('assistant', '');

                for await (const chunk of this.apiClient.streamMessage(message, this.sessionId, apiKey, chatHistory)) {
                    assistantMessage += chunk;
                    this.updateMessage(messageElement, assistantMessage);
                }

                this.removeTypingIndicator(typingId);
                this.statusText.textContent = 'Ready to chat';
                this.sendBtn.disabled = false;
                this.saveMessages();
                this.scrollToBottom();
                return;
            }

            // Remove typing indicator and add message
            this.removeTypingIndicator(typingId);
            this.addMessage('assistant', assistantMessage);

        } catch (error) {
            console.error('Error:', error);
            this.addMessage('assistant', `Error: ${error.message}`);
        }

        this.statusText.textContent = 'Ready to chat';
        this.sendBtn.disabled = false;
        this.saveMessages();
        this.scrollToBottom();
    }

    addMessage(role, content) {
        const message = { role, content, timestamp: new Date().toISOString() };
        this.messages.push(message);

        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = `max-w-3xl ${role === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-900'} rounded-2xl px-6 py-4 shadow-sm`;

        const headerDiv = document.createElement('div');
        headerDiv.className = 'flex items-center gap-2 mb-2';

        const icon = document.createElement('div');
        icon.className = 'font-semibold';
        icon.textContent = role === 'user' ? '👤 You' : '🤖 Azazel';
        headerDiv.appendChild(icon);

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content prose prose-sm max-w-none';
        messageContent.innerHTML = marked.parse(content);

        // Highlight code blocks
        messageContent.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        contentDiv.appendChild(headerDiv);
        contentDiv.appendChild(messageContent);
        messageDiv.appendChild(contentDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    updateMessage(messageElement, content) {
        const messageContent = messageElement.querySelector('.message-content');
        messageContent.innerHTML = marked.parse(content);

        // Highlight code blocks
        messageContent.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        this.scrollToBottom();
    }

    addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.className = 'flex justify-start';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'bg-white rounded-2xl px-6 py-4 shadow-sm';

        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';

        contentDiv.appendChild(typingDiv);
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);

        this.scrollToBottom();
        return id;
    }

    removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    renderMessages() {
        this.messagesContainer.innerHTML = '';
        this.messages.forEach(msg => {
            this.addMessage(msg.role, msg.content);
        });
    }

    newChat() {
        this.clearChat();
        this.sessionId = this.generateSessionId();
    }

    clearChat() {
        this.messages = [];
        this.messagesContainer.innerHTML = `
            <div class="text-center text-gray-500 mt-20">
                <div class="text-6xl mb-4">🤖</div>
                <h3 class="text-2xl font-semibold text-gray-900 mb-2">Welcome to Azazel AI</h3>
                <p class="text-gray-600">Your advanced AI assistant. How can I help you today?</p>
            </div>
        `;
        this.saveMessages();
    }

    async handleFileUpload(event) {
        const files = Array.from(event.target.files);
        const apiKey = this.apiKeyInput.value.trim();

        if (!apiKey) {
            alert('Please enter your OpenAI API key first');
            return;
        }

        for (const file of files) {
            try {
                this.statusText.textContent = `Uploading ${file.name}...`;
                const response = await this.apiClient.uploadDocument(file, this.sessionId, apiKey);

                if (response.success) {
                    this.uploadedFiles.push(file);
                    this.renderUploadedFiles();
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert(`Failed to upload ${file.name}: ${error.message}`);
            }
        }

        this.statusText.textContent = 'Ready to chat';
        event.target.value = ''; // Reset file input
    }

    renderUploadedFiles() {
        this.uploadedFilesDiv.innerHTML = '';

        this.uploadedFiles.forEach((file, index) => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'flex items-center justify-between bg-gray-100 px-3 py-2 rounded-lg';

            const fileName = document.createElement('span');
            fileName.className = 'text-sm text-gray-700 truncate';
            fileName.textContent = file.name;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'text-red-500 hover:text-red-700';
            removeBtn.innerHTML = '×';
            removeBtn.onclick = () => this.removeFile(index);

            fileDiv.appendChild(fileName);
            fileDiv.appendChild(removeBtn);
            this.uploadedFilesDiv.appendChild(fileDiv);
        });
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.renderUploadedFiles();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }

    // Audio/Image upload handlers
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const apiKey = this.apiKeyInput.value.trim();
        if (!apiKey) {
            alert('Please enter your OpenAI API key first');
            return;
        }

        try {
            this.statusText.textContent = `Uploading image ${file.name}...`;

            // Upload image to server immediately (same as document upload)
            const response = await this.apiClient.uploadDocument(file, this.sessionId, apiKey);

            if (response.success) {
                this.uploadedFiles.push(file);
                this.renderUploadedFiles();
                this.statusText.textContent = 'Image uploaded successfully';
            }
        } catch (error) {
            console.error('Image upload error:', error);
            alert(`Failed to upload image: ${error.message}`);
            this.statusText.textContent = 'Ready to chat';
        }

        event.target.value = ''; // Reset input
    }

    async handleAudioUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Show loading state
        this.statusText.textContent = 'Transcribing audio...';

        try {
            const apiKey = this.apiKeyInput.value;
            if (!apiKey) {
                alert('Please enter your OpenAI API key first');
                return;
            }

            // Transcribe audio
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('api_key', apiKey);

            const response = await fetch(`${CONFIG.API_URL}/api/chat/transcribe`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to transcribe audio');
            }

            const result = await response.json();

            // Set the transcribed text as the message
            this.messageInput.value = result.transcript;
            this.statusText.textContent = 'Audio transcribed successfully';

        } catch (error) {
            console.error('Audio transcription error:', error);
            alert('Failed to transcribe audio: ' + error.message);
            this.statusText.textContent = 'Ready to chat';
        }

        event.target.value = ''; // Reset input
    }

    showAttachmentPreview(icon, name) {
        this.attachmentIcon.textContent = icon;
        this.attachmentName.textContent = name;
        this.attachmentPreview.classList.remove('hidden');
    }

    removeAttachment() {
        // Removed currentAttachment - images are now uploaded immediately
        this.attachmentPreview.classList.add('hidden');
        this.imageFileInput.value = '';
        this.audioFileInput.value = '';
    }

    getPersonalityPrompt() {
        const personalities = {
            'helpful': 'You are a helpful AI assistant.',
            'professional': 'You are a professional AI assistant. Be formal, precise, and business-oriented in your responses.',
            'friendly': 'You are a friendly AI assistant. Be warm, casual, and approachable in your responses.',
            'concise': 'You are a concise AI assistant. Provide brief, to-the-point responses without unnecessary elaboration.',
            'creative': 'You are a creative AI assistant. Think outside the box and provide imaginative, innovative solutions.',
            'technical': 'You are a technical expert AI assistant. Provide detailed, accurate technical information with examples and best practices.'
        };

        const selected = this.personalitySelect.value;
        return personalities[selected] || personalities['helpful'];
    }

    getLanguage() {
        return this.languageSelect.value || 'English';
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AzazelApp();
});