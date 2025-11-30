import { useState } from 'react';
import { User, ImageIcon, Bug, X } from 'lucide-react';
import { Character } from '../types';

interface RightSidebarProps {
    character: Character | null;
    onClose: () => void;
}

type TabType = 'character' | 'media' | 'debug';

export default function RightSidebar({ character, onClose }: RightSidebarProps) {
    const [activeTab, setActiveTab] = useState<TabType>('character');

    const tabs: { id: TabType; label: string; icon: any }[] = [
        { id: 'character', label: 'Character', icon: User },
        { id: 'media', label: 'Media', icon: ImageIcon },
        { id: 'debug', label: 'Debug', icon: Bug },
    ];

    return (
        <div className="flex h-full w-96 flex-col border-l border-gray-800 bg-candy-card">
            {/* Tabs */}
            <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
                <div className="flex gap-2">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${activeTab === tab.id
                                    ? 'bg-candy-pink text-white'
                                    : 'text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                                }`}
                        >
                            <tab.icon size={16} />
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </div>
                <button onClick={onClose} className="rounded-lg p-1 hover:bg-gray-700 md:hidden">
                    <X size={20} />
                </button>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-4">
                {activeTab === 'character' && <CharacterTab character={character} />}
                {activeTab === 'media' && <MediaTab />}
                {activeTab === 'debug' && <DebugTab />}
            </div>
        </div>
    );
}

function CharacterTab({ character }: { character: Character | null }) {
    if (!character) {
        return (
            <div className="flex h-full items-center justify-center text-gray-500">
                <p>No character selected</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div>
                <h3 className="mb-2 text-sm font-semibold text-gray-400">Name</h3>
                <p className="text-gray-100">{character.name}</p>
            </div>

            {character.description && (
                <div>
                    <h3 className="mb-2 text-sm font-semibold text-gray-400">Description</h3>
                    <p className="text-sm text-gray-300">{character.description}</p>
                </div>
            )}

            {character.personality && (
                <div>
                    <h3 className="mb-2 text-sm font-semibold text-gray-400">Personality</h3>
                    <p className="whitespace-pre-wrap text-sm text-gray-300">{character.personality}</p>
                </div>
            )}

            {character.backstory && (
                <div>
                    <h3 className="mb-2 text-sm font-semibold text-gray-400">Backstory</h3>
                    <p className="whitespace-pre-wrap text-sm text-gray-300">{character.backstory}</p>
                </div>
            )}

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <h3 className="mb-1 text-xs font-semibold text-gray-400">Language</h3>
                    <p className="text-sm text-gray-300">{character.language || 'en'}</p>
                </div>
                <div>
                    <h3 className="mb-1 text-xs font-semibold text-gray-400">Voice</h3>
                    <p className="text-sm text-gray-300">{character.voice_style || 'default'}</p>
                </div>
            </div>
        </div>
    );
}

function MediaTab() {
    return (
        <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-400">Recent Images</h3>
            <div className="grid grid-cols-2 gap-2">
                {/* Placeholder for generated images */}
                <div className="aspect-square rounded-lg bg-gray-800 flex items-center justify-center text-gray-600 text-xs">
                    No images yet
                </div>
            </div>

            <h3 className="mt-6 text-sm font-semibold text-gray-400">Audio Clips</h3>
            <div className="text-sm text-gray-500">
                <p>TTS audio history will appear here</p>
            </div>
        </div>
    );
}

function DebugTab() {
    const isDev = import.meta.env.DEV;

    if (!isDev) {
        return (
            <div className="text-sm text-gray-500">
                <p>Debug mode only available in development</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div>
                <h3 className="mb-2 text-sm font-semibold text-gray-400">Environment</h3>
                <div className="rounded-lg bg-gray-800 p-3 font-mono text-xs">
                    <p>Mode: {import.meta.env.MODE}</p>
                    <p>API URL: {import.meta.env.VITE_API_URL || 'default'}</p>
                </div>
            </div>

            <div>
                <h3 className="mb-2 text-sm font-semibold text-gray-400">Recent Errors</h3>
                <div className="text-xs text-gray-500">
                    <p>No errors logged</p>
                </div>
            </div>
        </div>
    );
}
