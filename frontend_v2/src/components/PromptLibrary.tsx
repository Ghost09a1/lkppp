import { useState } from 'react';
import { Sparkles, X } from 'lucide-react';

interface PromptLibraryProps {
    onSelectPrompt: (prompt: string) => void;
    onClose: () => void;
}

interface PromptPreset {
    id: string;
    title: string;
    prompt: string;
    category: string;
}

const defaultPresets: PromptPreset[] = [
    {
        id: '1',
        title: 'Casual Chat',
        prompt: "Let's have a casual conversation about ",
        category: 'conversation',
    },
    {
        id: '2',
        title: 'Roleplay Scenario',
        prompt: "Let's roleplay a scenario where ",
        category: 'roleplay',
    },
    {
        id: '3',
        title: 'Creative Writing',
        prompt: "Help me write a creative story about ",
        category: 'creative',
    },
    {
        id: '4',
        title: 'Ask for Advice',
        prompt: "I need your advice on ",
        category: 'help',
    },
    {
        id: '5',
        title: 'Explain Concept',
        prompt: "Can you explain ",
        category: 'educational',
    },
];

export default function PromptLibrary({ onSelectPrompt, onClose }: PromptLibraryProps) {
    const [presets] = useState<PromptPreset[]>(defaultPresets);
    const [selectedCategory, setSelectedCategory] = useState<string>('all');

    const categories = ['all', 'conversation', 'roleplay', 'creative', 'help', 'educational'];

    const filtered =
        selectedCategory === 'all'
            ? presets
            : presets.filter((p) => p.category === selectedCategory);

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-lg rounded-lg bg-candy-card p-6 shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Sparkles size={20} className="text-candy-pink" />
                        <h3 className="font-semibold">Prompt Library</h3>
                    </div>
                    <button onClick={onClose} className="rounded-lg p-2 hover:bg-gray-700">
                        <X size={20} />
                    </button>
                </div>

                {/* Category Filter */}
                <div className="mb-4 flex flex-wrap gap-2">
                    {categories.map((cat) => (
                        <button
                            key={cat}
                            onClick={() => setSelectedCategory(cat)}
                            className={`rounded-full px-3 py-1 text-xs capitalize transition-colors ${selectedCategory === cat
                                ? 'bg-candy-pink text-white'
                                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                                }`}
                        >
                            {cat}
                        </button>
                    ))}
                </div>

                {/* Preset List */}
                <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
                    {filtered.map((preset) => (
                        <button
                            key={preset.id}
                            onClick={() => {
                                onSelectPrompt(preset.prompt);
                                onClose();
                            }}
                            className="w-full rounded-lg bg-gray-800 p-3 text-left transition-colors hover:bg-gray-700"
                        >
                            <div className="mb-1 font-medium text-sm">{preset.title}</div>
                            <div className="text-xs text-gray-400 line-clamp-2">{preset.prompt}</div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
