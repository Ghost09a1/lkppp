// API Configuration
const API_BASE = '';  // Same origin

// State Management
const state = {
    currentView: 'characters',
    characters: [],
    currentCharacter: null,
    messages: [],
    settings: null,
    isLoading: false
};

// Utility Functions
function showLoading(element) {
    const spinner = document.createElement('div');
    spinner.className = 'loading';
    element.appendChild(spinner);
}

function hideLoading(element) {
    const spinner = element.querySelector('.loading');
    if (spinner) spinner.remove();
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Auto-dismiss after 4 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function showError(message) {
    showToast(message, 'error');
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showInfo(message) {
    showToast(message, 'info');
}

// Navigation
const navTabs = document.querySelectorAll('.nav-tab');
const views = document.querySelectorAll('.view');

navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const viewName = tab.dataset.view;
        switchView(viewName);
    });
});

function switchView(viewName) {
    navTabs.forEach(t => t.classList.remove('active'));
    views.forEach(v => v.classList.remove('active'));
    
    const activeTab = document.querySelector(`[data-view="${viewName}"]`);
    const activeView = document.getElementById(`${viewName}-view`);
    
    if (activeTab) activeTab.classList.add('active');
    if (activeView) activeView.classList.add('active');
    
    state.currentView = viewName;
    
    // Load data for view
    if (viewName === 'characters') {
        loadCharacters();
    } else if (viewName === 'chat') {
        loadCharactersForChat();
    } else if (viewName === 'settings') {
        loadSettings();
    }
}

// Characters View
async function loadCharacters() {
    try {
        const response = await fetch(`${API_BASE}/characters`);
        const data = await response.json();
        state.characters = data.characters || [];
        renderCharacters();
    } catch (error) {
        showError('Failed to load characters: ' + error.message);
    }
}

function renderCharacters() {
    const grid = document.getElementById('char-grid');
    grid.innerHTML = '';
    
    if (state.characters.length === 0) {
        grid.innerHTML = '<p class="text-muted">No characters yet. Create your first character!</p>';
        return;
    }
    
    state.characters.forEach(char => {
        const card = createCharacterCard(char);
        grid.appendChild(card);
    });
}

function createCharacterCard(char) {
    const card = document.createElement('div');
    card.className = 'char-card';
    
    const avatarUrl = char.avatar_path ? `/avatars/${char.avatar_path}` : '';
    const avatarContent = avatarUrl 
        ? `<img src="${avatarUrl}" alt="${char.name}">`
        : char.name.charAt(0).toUpperCase();
    
    card.innerHTML = `
        <div class="char-avatar">${avatarContent}</div>
        <div class="char-name">${char.name}</div>
        <div class="char-description">${char.description || 'No description'}</div>
        <div class="char-meta">
            <span class="char-tag">${char.language || 'en'}</span>
            ${char.relationship_type ? `<span class="char-tag">${char.relationship_type}</span>` : ''}
        </div>
        <div class="char-actions">
            <button class="btn btn-primary btn-chat" data-id="${char.id}">Chat</button>
            <button class="btn btn-secondary btn-edit" data-id="${char.id}">Edit</button>
        </div>
    `;
    
    card.querySelector('.btn-chat').addEventListener('click', (e) => {
        e.stopPropagation();
        startChat(char.id);
    });
    
    card.querySelector('.btn-edit').addEventListener('click', (e) => {
        e.stopPropagation();
        editCharacter(char.id);
    });
    
    return card;
}

function startChat(charId) {
    const char = state.characters.find(c => c.id === charId);
    if (char) {
        state.currentCharacter = char;
        state.messages = [];
        switchView('chat');
        document.getElementById('current-char-name').textContent = char.name;
    }
}

function editCharacter(charId) {
    const char = state.characters.find(c => c.id === charId);
    if (char) {
        openCharacterModal(char);
    }
}

// Character Modal
const charModal = document.getElementById('char-modal');
const charForm = document.getElementById('char-form');
const btnNewChar = document.getElementById('btn-new-char');

btnNewChar.addEventListener('click', () => {
    openCharacterModal();
});

document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', () => {
        closeAllModals();
    });
});

charModal.addEventListener('click', (e) => {
    if (e.target === charModal) {
        closeAllModals();
    }
});

function openCharacterModal(char = null) {
    const modalTitle = document.getElementById('modal-title');
    const form = charForm;
    
    if (char) {
        modalTitle.textContent = 'Edit Character';
        document.getElementById('char-id').value = char.id;
        document.getElementById('char-name').value = char.name || '';
        document.getElementById('char-description').value = char.description || '';
        document.getElementById('char-personality').value = char.personality || '';
        document.getElementById('char-backstory').value = char.backstory || '';
        document.getElementById('char-visual').value = char.visual_style || '';
        document.getElementById('char-relationship').value = char.relationship_type || '';
        document.getElementById('char-language').value = char.language || 'en';
    } else {
        modalTitle.textContent = 'New Character';
        form.reset();
        document.getElementById('char-id').value = '';
    }
    
    charModal.classList.add('active');
}

function closeAllModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
}

charForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const charId = document.getElementById('char-id').value;
    const formData = {
        name: document.getElementById('char-name').value,
        description: document.getElementById('char-description').value,
        personality: document.getElementById('char-personality').value,
        backstory: document.getElementById('char-backstory').value,
        visual_style: document.getElementById('char-visual').value,
        appearance_notes: '',
        relationship_type: document.getElementById('char-relationship').value,
        dos: '',
        donts: '',
        voice_style: '',
        voice_pitch_shift: 0.0,
        voice_speed: 1.0,
        voice_ref_path: '',
        voice_youtube_url: '',
        voice_model_path: '',
        voice_training_status: '',
        voice_error: '',
        language: document.getElementById('char-language').value
    };
    
    try {
        let response;
        if (charId) {
            // Update existing character
            response = await fetch(`${API_BASE}/characters/${charId}/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new character
            response = await fetch(`${API_BASE}/characters`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
        }
        
        if (!response.ok) throw new Error('Failed to save character');
        
        const result = await response.json();
        
        // Handle avatar upload if present
        const avatarFile = document.getElementById('char-avatar').files[0];
        if (avatarFile) {
            const avatarFormData = new FormData();
            avatarFormData.append('file', avatarFile);
            
            const avatarCharId = charId || result.id;
            await fetch(`${API_BASE}/characters/${avatarCharId}/avatar`, {
                method: 'POST',
                body: avatarFormData
            });
        }
        
        closeAllModals();
        loadCharacters();
        showSuccess(charId ? 'Character updated' : 'Character created');
    } catch (error) {
        showError('Failed to save character: ' + error.message);
    }
});

// Chat View
async function loadCharactersForChat() {
    try {
        const response = await fetch(`${API_BASE}/characters`);
        const data = await response.json();
        state.characters = data.characters || [];
        renderCharacterList();
    } catch (error) {
        showError('Failed to load characters: ' + error.message);
    }
}

function renderCharacterList() {
    const list = document.getElementById('char-list');
    list.innerHTML = '';
    
    if (state.characters.length === 0) {
        list.innerHTML = '<p class="text-muted">No characters available</p>';
        return;
    }
    
    state.characters.forEach(char => {
        const item = document.createElement('div');
        item.className = 'char-list-item';
        if (state.currentCharacter && state.currentCharacter.id === char.id) {
            item.classList.add('active');
        }
        
        item.innerHTML = `
            <div class="char-list-item-name">${char.name}</div>
        `;
        
        item.addEventListener('click', () => {
            selectCharacterForChat(char);
        });
        
        list.appendChild(item);
    });
}

function selectCharacterForChat(char) {
    state.currentCharacter = char;
    state.messages = [];
    document.getElementById('current-char-name').textContent = char.name;
    renderCharacterList();
    renderMessages();
}

function renderMessages() {
    const messagesContainer = document.getElementById('messages');
    messagesContainer.innerHTML = '';
    
    if (state.messages.length === 0) {
        messagesContainer.innerHTML = '<p class="text-muted">Start a conversation!</p>';
        return;
    }
    
    state.messages.forEach(msg => {
        const messageEl = createMessageElement(msg);
        messagesContainer.appendChild(messageEl);
    });
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createMessageElement(msg) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    
    const avatar = msg.role === 'user' ? '👤' : state.currentCharacter?.name.charAt(0).toUpperCase() || '🤖';
    
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div>${msg.content}</div>
            ${msg.role === 'assistant' ? `
                <div class="message-actions">
                    ${msg.audio_base64 ? '<button class="message-action-btn" onclick="playTTS(this)">🔊 Play</button>' : ''}
                    <button class="message-action-btn" onclick="regenerateMessage()">🔄 Regenerate</button>
                </div>
            ` : ''}
        </div>
    `;
    
    if (msg.audio_base64) {
        div.dataset.audio = msg.audio_base64;
    }
    
    return div;
}

// Send Message
const btnSend = document.getElementById('btn-send');
const userInput = document.getElementById('user-input');

btnSend.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    if (!state.currentCharacter) {
        showError('Please select a character first');
        return;
    }
    
    const message = userInput.value.trim();
    if (!message) return;
    
    // Add user message to UI
    state.messages.push({
        role: 'user',
        content: message
    });
    renderMessages();
    
    userInput.value = '';
    btnSend.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/chat/${state.currentCharacter.id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) throw new Error('Chat request failed');
        
        const data = await response.json();
        
        // Add assistant message to UI
        state.messages.push({
            role: 'assistant',
            content: data.reply || 'No response',
            audio_base64: data.audio_base64
        });
        renderMessages();
    } catch (error) {
        showError('Failed to send message: ' + error.message);
        state.messages.pop(); // Remove user message on failure
        renderMessages();
    } finally {
        btnSend.disabled = false;
    }
}

async function regenerateMessage() {
    if (state.messages.length < 2) return;
    
    // Remove last assistant message
    state.messages.pop();
    
    // Get last user message
    const lastUserMsg = state.messages[state.messages.length - 1];
    if (lastUserMsg && lastUserMsg.role === 'user') {
        // Resend
        userInput.value = lastUserMsg.content;
        state.messages.pop();
        sendMessage();
    }
}

function playTTS(button) {
    const messageEl = button.closest('.message');
    const audioBase64 = messageEl.dataset.audio;
    
    if (audioBase64) {
        const audio = new Audio(audioBase64);
        audio.play().catch(err => showError('Failed to play audio: ' + err.message));
    }
}

// Image Generation
const btnGenImage = document.getElementById('btn-gen-image');
const imageModal = document.getElementById('image-modal');
const imageForm = document.getElementById('image-form');

btnGenImage.addEventListener('click', () => {
    imageModal.classList.add('active');
});

imageForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!state.currentCharacter) {
        showError('Please select a character first');
        return;
    }
    
    const prompt = document.getElementById('image-prompt').value;
    const negative = document.getElementById('image-negative').value;
    const width = parseInt(document.getElementById('image-width').value);
    const height = parseInt(document.getElementById('image-height').value);
    
    const resultDiv = document.getElementById('image-result');
    resultDiv.innerHTML = '<div class="loading"></div><p>Generating image...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/posts/${state.currentCharacter.id}/image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, negative, width, height, steps: 20 })
        });
        
        if (!response.ok) throw new Error('Image generation failed');
        
        const data = await response.json();
        
        if (data.image_base64) {
            resultDiv.innerHTML = `<img src="${data.image_base64}" alt="Generated image">`;
            
            // Add to chat
            state.messages.push({
                role: 'assistant',
                content: `🎨 Generated image: ${prompt}`,
                image: data.image_base64
            });
            renderMessages();
        } else {
            resultDiv.innerHTML = '<p class="text-error">Image generation failed</p>';
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="text-error">Error: ${error.message}</p>`;
    }
});

// Voice Recording (Stub)
const btnRecord = document.getElementById('btn-record');
btnRecord.addEventListener('click', () => {
    showError('Voice recording not yet implemented');
});

// Settings
async function loadSettings() {
    try {
        // Load current config (you'd need a /config endpoint in backend)
        // For now, we'll use defaults
        document.getElementById('llm-mode').value = 'gguf';
        document.getElementById('llm-model-path').value = 'models/llm/model.gguf';
        document.getElementById('tts-enabled').checked = true;
        document.getElementById('image-gen-enabled').checked = false;
        
        // Check backend status
        checkBackendStatus();
    } catch (error) {
        showError('Failed to load settings: ' + error.message);
    }
}

async function checkBackendStatus() {
    const backendStatus = document.getElementById('status-backend');
    const llmStatus = document.getElementById('status-llm');
    const ttsStatus = document.getElementById('status-tts');
    
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (response.ok) {
            const data = await response.json();
            
            // Backend status
            backendStatus.classList.remove('offline');
            backendStatus.classList.add('online');
            backendStatus.textContent = '● Online';
            
            // LLM status
            llmStatus.classList.remove('online', 'offline');
            if (data.llm === 'online') {
                llmStatus.classList.add('online');
                llmStatus.textContent = '● Online';
            } else if (data.llm === 'offline') {
                llmStatus.classList.add('offline');
                llmStatus.textContent = '● Offline';
            } else {
                llmStatus.textContent = '● Unknown';
            }
            
            // TTS status
            ttsStatus.classList.remove('online', 'offline');
            if (data.tts === 'online') {
                ttsStatus.classList.add('online');
                ttsStatus.textContent = '● Online';
            } else if (data.tts === 'offline' || data.tts === 'disabled') {
                ttsStatus.textContent = `● ${data.tts}`;
            } else {
                ttsStatus.textContent = '● Unknown';
            }
        } else {
            backendStatus.classList.add('offline');
            backendStatus.textContent = '● Offline';
        }
    } catch (error) {
        backendStatus.classList.remove('online');
        backendStatus.classList.add('offline');
        backendStatus.textContent = '● Offline';
        llmStatus.textContent = '● Unknown';
        ttsStatus.textContent = '● Unknown';
    }
}

const btnSaveSettings = document.getElementById('btn-save-settings');
btnSaveSettings.addEventListener('click', () => {
    showError('Settings save not yet implemented - edit config/settings.json manually');
});

// Character Search
const charSearch = document.getElementById('char-search');
charSearch.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const cards = document.querySelectorAll('.char-card');
    
    cards.forEach(card => {
        const name = card.querySelector('.char-name').textContent.toLowerCase();
        const desc = card.querySelector('.char-description').textContent.toLowerCase();
        
        if (name.includes(query) || desc.includes(query)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCharacters();
});
