import { useState } from 'react';
import { X, Image as ImageIcon, Play, Loader2 } from 'lucide-react';

interface MediaGenModalProps {
    isOpen: boolean;
    onClose: () => void;
    type: 'image' | 'video';
    onGenerate: (prompt: string, negative?: string) => Promise<void>;
}

export function MediaGenModal({ isOpen, onClose, type, onGenerate }: MediaGenModalProps) {
    const [prompt, setPrompt] = useState("");
    const [negative, setNegative] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);

    if (!isOpen) return null;

    const handleGenerate = async () => {
        if (!prompt.trim()) return;
        setIsGenerating(true);
        try {
            await onGenerate(prompt, negative);
            onClose();
            setPrompt("");
            setNegative("");
        } catch (err) {
            console.error("Generation failed", err);
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-md space-y-4 shadow-2xl">
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        {type === 'image' ? <ImageIcon size={20} /> : <Play size={20} />}
                        Generate {type === 'image' ? 'Image' : 'Video'}
                    </h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>

                <div className="space-y-3">
                    <div>
                        <label className="text-xs text-gray-400">Prompt</label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                            rows={3}
                            placeholder={type === 'image' ? "Describe the image..." : "Describe the scene..."}
                            autoFocus
                        />
                    </div>

                    {type === 'image' && (
                        <div>
                            <label className="text-xs text-gray-400">Negative Prompt (Optional)</label>
                            <input
                                value={negative}
                                onChange={(e) => setNegative(e.target.value)}
                                className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                                placeholder="Blurry, low quality, etc."
                            />
                        </div>
                    )}
                </div>

                <div className="flex justify-end gap-2 pt-2">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating || !prompt.trim()}
                        className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-400 flex items-center gap-2"
                    >
                        {isGenerating && <Loader2 size={16} className="animate-spin" />}
                        {isGenerating ? "Generating..." : "Generate"}
                    </button>
                </div>
            </div>
        </div>
    );
}
