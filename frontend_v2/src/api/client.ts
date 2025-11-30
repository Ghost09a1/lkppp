import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || '';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const apiClient = {
    // Chat
    async sendMessage(charId: number, message: string, enableTTS: boolean = true) {
        console.log(`[API] Sending message to char ${charId} with enable_tts=${enableTTS}`);
        const res = await api.post(`/chat/${charId}`, { message, enable_tts: !!enableTTS });
        return res.data;
    },

    // Characters
    async getCharacters() {
        const res = await api.get('/characters');
        return res.data.characters || [];
    },

    async createCharacter(data: any) {
        const res = await api.post('/characters', data);
        return res.data;
    },

    async updateCharacter(id: number, data: any) {
        const res = await api.post(`/characters/${id}/update`, data);
        return res.data;
    },

    // STT
    async transcribeAudio(audioBlob: Blob, language?: string) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        if (language) formData.append('language', language);

        const res = await api.post('/stt', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return res.data;
    },

    // TTS
    async getTTS(text: string, characterId?: number) {
        const res = await api.post('/tts', {
            message: text,
            character_id: characterId
        });
        return res.data;
    },

    // Image Generation
    async generateImage(charId: number, prompt: string, negative = '', steps = 20) {
        const res = await api.post(`/posts/${charId}/image`, {
            prompt,
            negative,
            steps,
            width: 512,
            height: 768,
        });
        return res.data;
    },

    // Status
    async getStatus() {
        const res = await api.get('/status');
        return res.data;
    },
};
