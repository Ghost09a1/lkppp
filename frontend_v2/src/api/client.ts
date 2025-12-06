import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || '';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 1800000, // 30 min (1800s) for slow CPU inference
});

export const apiClient = {
    // Chat
    // Chat
    async sendMessage(
        charId: number,
        message: string,
        enableTTS: boolean = true,
        enableImage: boolean = false,
        forceImage: boolean = false,
        onUpdate?: (partial: { text?: string; audio?: string; image?: string }) => void
    ) {
        console.log(`[API] Stream Send: charId=${charId}, msg="${message.substring(0, 50)}..."`);

        const response = await fetch(`${API_URL}/chat/${charId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                enable_tts: !!enableTTS,
                enable_image: !!enableImage,
                force_image: !!forceImage
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Helper: Clean display text (remove all <...> tokens and [GENERATE_...] tags)
        const cleanDisplayText = (text: string): string => {
            return text
                .replace(/<[^>]+>/g, '')  // Remove all <...> tags (custom tokens, audio markers, etc.)
                .replace(/\[GENERATE_[^\]]*\]/gi, '')  // Remove [GENERATE_IMAGE] etc.
                .replace(/\[EMOTE:[^\]]*\]/gi, '')  // Remove [EMOTE:...]
                .replace(/\s{2,}/g, ' ')  // Collapse multiple spaces
                .trim();
        };

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullReply = '';
        let fullAudio = '';
        let fullImage = '';

        if (!reader) throw new Error("No reader available");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Normalize newlines to \n
            buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line

            for (const line of lines) {
                if (!line.trim()) continue; // Skip empty lines (separators)

                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    if (dataStr.trim() === '[STREAM_END]') continue;

                    try {
                        const data = JSON.parse(dataStr);

                        // Handle Token (Text Stream)
                        if (data.type === 'token') {
                            fullReply += data.token;
                            // [FIX] Clean accumulated text before sending to UI
                            onUpdate?.({ text: cleanDisplayText(fullReply) });
                        }

                        // Handle Audio
                        if (data.type === 'audio') {
                            console.log("[API] Received Audio Chunk");
                            fullAudio = data.audio;
                            onUpdate?.({ audio: fullAudio });
                        }

                        // Handle Image
                        if (data.type === 'image') {
                            console.log("[API] Received Image");
                            fullImage = data.image;
                            onUpdate?.({ image: fullImage });
                        }

                    } catch (e) {
                        // Only warn if it's not a multi-line data fragment (rare in this app)
                        console.warn("Failed to parse SSE line:", dataStr.substring(0, 50));
                    }
                }
            }
        }

        // Process final buffer residue if it's a complete line
        if (buffer.trim() && buffer.startsWith('data: ')) {
            const dataStr = buffer.slice(6);
            try {
                const data = JSON.parse(dataStr);
                if (data.type === 'token') {
                    fullReply += data.token;
                    onUpdate?.({ text: cleanDisplayText(fullReply) });
                }
            } catch (e) { }
        }

        console.log("[API] Final Reply Length:", fullReply.length);

        // [FIX] Return cleaned text for final display
        return { reply: cleanDisplayText(fullReply), audio_base64: fullAudio, image_base64: fullImage };
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

    // Reference Images
    async uploadReferenceImage(charId: number, file: File) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await api.post(`/characters/${charId}/reference_images`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return res.data;
    },

    async deleteReferenceImage(charId: number, imgId: number) {
        const res = await api.delete(`/characters/${charId}/reference_images/${imgId}`);
        return res.data;
    },

    // Avatar Upload
    async uploadAvatar(charId: number, file: File) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await api.post(`/characters/${charId}/avatar`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return res.data;
    },

    // Status
    async getStatus() {
        const res = await api.get('/status');
        const data = res.data;
        // [FIX 4] Convert new backend format to frontend ServiceStatus format
        return {
            backend: data.status || 'offline',
            llm: data.llm_online ? 'online' : 'offline',
            tts: data.tts_online ? 'online' : 'offline',
            image_gen: 'disabled', // Not checked yet
        };
    },

    // Voice Training
    async uploadVoiceSample(charId: number, file: File) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await api.post(`/characters/${charId}/voice_sample`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return res.data;
    },

    async trainVoice(charId: number) {
        const res = await api.post(`/characters/${charId}/train_voice`);
        return res.data;
    },
};
