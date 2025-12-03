import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || '';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 600000, // 600 seconds (10 min) to be absolutely safe
});

export const apiClient = {
    // Chat
    async sendMessage(charId: number, message: string, enableTTS: boolean = true, enableImage: boolean = false, forceImage: boolean = false) {
        console.log(`[API] Sending: charId=${charId}, msg="${message.substring(0, 50)}...", enableTTS=${enableTTS}, enableImage=${enableImage}, forceImage=${forceImage}`);
        const body = {
            message,
            enable_tts: !!enableTTS,
            enable_image: !!enableImage,
            force_image: !!forceImage
        };
        console.log("[API] Request body:", body);
        const res = await api.post(`/chat/${charId}`, body);
        console.log("[API] Response:", res.data);
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
