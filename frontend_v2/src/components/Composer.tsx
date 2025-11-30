import React from 'react';
import { Send, Mic, Image as ImageIcon, Book } from 'lucide-react';

interface ComposerProps {
    input: string;
    setInput: (val: string) => void;
    onSend: () => void;
    onRecordStart: () => void;
    onRecordStop: () => void;
    isRecording: boolean;
    isSending: boolean;
    onImageClick: () => void;
    onTogglePrompts: () => void;
    inputRef?: React.RefObject<HTMLTextAreaElement>;
}

export default function Composer({
    input,
    setInput,
    onSend,
    onRecordStart,
    onRecordStop,
    isRecording,
    isSending,
    onImageClick,
    onTogglePrompts,
    inputRef
}: ComposerProps) {
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSend();
        }
    };

    return (
        <div className="border-t border-gray-800 bg-candy-card p-4">
            <div className="flex gap-2 items-end">
                <button
                    onClick={onTogglePrompts}
                    className="p-3 text-gray-400 hover:text-pink-500 transition"
                    title="Prompt Library"
                >
                    <Book size={20} />
                </button>
                <div className="flex-1 bg-gray-900/50 rounded-xl border border-gray-700 focus-within:border-pink-500/50 transition-colors flex items-center">
                    <textarea
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type a message..."
                        className="w-full bg-transparent border-none outline-none text-white p-3 min-h-[44px] max-h-32 resize-none"
                        rows={1}
                        disabled={isSending}
                    />
                </div>

                <div className="flex gap-2">
                    <button
                        onMouseDown={onRecordStart}
                        onMouseUp={onRecordStop}
                        onMouseLeave={onRecordStop}
                        className={`p-3 rounded-xl transition-colors ${isRecording ? 'bg-red-500/20 text-red-500' : 'bg-gray-800 hover:bg-gray-700 text-gray-400'
                            }`}
                        title="Hold to record"
                        disabled={isSending}
                    >
                        <Mic size={20} />
                    </button>

                    <button
                        onClick={onImageClick}
                        className="p-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-400 transition-colors"
                        title="Generate Image"
                        disabled={isSending}
                    >
                        <ImageIcon size={20} />
                    </button>

                    <button
                        onClick={onSend}
                        disabled={!input.trim() || isSending}
                        className="p-3 rounded-xl bg-pink-600 hover:bg-pink-500 disabled:bg-gray-800 disabled:text-gray-600 text-white transition-colors shadow-lg shadow-pink-900/20"
                        title="Send"
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
}
