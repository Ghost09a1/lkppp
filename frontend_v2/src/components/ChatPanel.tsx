import { useEffect, useRef } from 'react';
import { Message } from '../types';
import { Volume2 } from 'lucide-react';

interface ChatPanelProps {
    messages: Message[];
    loading: boolean;
}

export default function ChatPanel({ messages, loading }: ChatPanelProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const playAudio = (audio_base64: string) => {
        const audio = new Audio(audio_base64);
        audio.play().catch(err => console.error('Audio play failed:', err));
    };

    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="space-y-4">
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[70%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                                    ? 'bg-candy-pink text-white'
                                    : 'bg-gray-800 text-gray-100'
                                }`}
                        >
                            <p className="whitespace-pre-wrap">{msg.content}</p>

                            {msg.image_base64 && (
                                <img
                                    src={msg.image_base64}
                                    alt="Generated"
                                    className="mt-2 rounded-lg"
                                />
                            )}

                            {msg.audio_base64 && msg.role === 'assistant' && (
                                <button
                                    onClick={() => playAudio(msg.audio_base64!)}
                                    className="mt-2 flex items-center gap-1 text-sm opacity-75 hover:opacity-100"
                                >
                                    <Volume2 size={14} />
                                    <span>Play</span>
                                </button>
                            )}
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="flex justify-start">
                        <div className="rounded-2xl bg-gray-800 px-4 py-3">
                            <div className="flex gap-1">
                                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: '0ms' }} />
                                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: '150ms' }} />
                                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>
        </div>
    );
}
