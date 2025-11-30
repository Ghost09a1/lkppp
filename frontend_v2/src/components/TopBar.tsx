import { Search, Settings } from 'lucide-react';
import { Character, ServiceStatus } from '../types';

interface TopBarProps {
    character: Character | null;
    status: ServiceStatus | null;
    onSearch: () => void;
    onSettings: () => void;
    apiBase: string;
}

const StatusDot = ({ status }: { status: string }) => {
    const color = status === 'online' ? 'bg-green-500' :
        status === 'disabled' ? 'bg-gray-500' :
            'bg-red-500';
    return <div className={`h-2 w-2 rounded-full ${color}`} />;
};

export default function TopBar({ character, status, onSearch, onSettings }: TopBarProps) {
    return (
        <div className="flex items-center justify-between border-b border-gray-800 bg-candy-card px-4 md:px-6 py-3">
            {/* Left: Title & Char */}
            <div className="flex items-center gap-3">
                <h1 className="text-xl font-bold text-candy-pink hidden md:block">MyCandy</h1>
                {character && (
                    <div className="flex items-center gap-2 bg-gray-800/50 px-3 py-1 rounded-full">
                        <span className="text-sm font-medium text-gray-200">{character.name}</span>
                    </div>
                )}
            </div>

            {/* Right: Status & Actions */}
            <div className="flex items-center gap-3 md:gap-4">
                {status && (
                    <div className="hidden md:flex items-center gap-4 text-xs mr-4 border-r border-gray-700 pr-4">
                        <div className="flex items-center gap-1.5" title="LLM Status">
                            <StatusDot status={status.llm} />
                            <span className="text-gray-400">LLM</span>
                        </div>
                        <div className="flex items-center gap-1.5" title="TTS Status">
                            <StatusDot status={status.tts} />
                            <span className="text-gray-400">TTS</span>
                        </div>
                    </div>
                )}

                <button onClick={onSearch} className="p-2 text-gray-400 hover:text-white transition hover:bg-gray-800 rounded-lg">
                    <Search size={20} />
                </button>
                <button onClick={onSettings} className="p-2 text-gray-400 hover:text-white transition hover:bg-gray-800 rounded-lg">
                    <Settings size={20} />
                </button>
            </div>
        </div>
    );
}
