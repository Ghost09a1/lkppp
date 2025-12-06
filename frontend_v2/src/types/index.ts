// Type definitions for MyCandyLocal API
export interface Character {
    id: number;
    name: string;
    description?: string;
    personality?: string;
    backstory?: string;
    visual_style?: string;
    avatar_path?: string;
    language?: string;
    voice_style?: string;
    voice_model_path?: string;
    voice_training_status?: string;
    reference_images?: { id: number; url: string }[];
    negative_prompt?: string;
}

export interface Message {
    id: number;
    role: 'user' | 'assistant';
    content: string;
    audio_base64?: string;
    image_base64?: string;
    timestamp?: number;
}

export interface ServiceStatus {
    backend: string;
    llm: string;
    tts: string;
    image_gen: string;
}

export interface ChatResponse {
    reply: string;
    audio_base64?: string;
    transcription?: string;
}
