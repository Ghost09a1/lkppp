import { User, PlusCircle, CheckCircle, AlertTriangle } from 'lucide-react';
import { Character } from '../types';

interface CharacterSidebarProps {
    characters: Character[];
    selectedCharId: number | null;
    onSelect: (id: number) => void;
    onEdit: (char: Character) => void;
    onCreate: () => void;
    apiBase: string;
}

export default function CharacterSidebar({
    characters,
    selectedCharId,
    onSelect,
    onEdit,
    onCreate,
    apiBase
}: CharacterSidebarProps) {

    const avatarUrl = (char: Character) => {
        if (!char.avatar_path) return "";
        return `${apiBase}/avatars/${char.avatar_path}`;
    };

    const statusTone = (status: string) => {
        switch (status) {
            case "done": return "text-green-400 border-green-500/60 bg-green-500/10";
            case "running": return "text-pink-300 border-pink-500/60 bg-pink-500/10";
            case "queued": return "text-amber-300 border-amber-500/60 bg-amber-500/10";
            case "failed": return "text-red-300 border-red-500/60 bg-red-500/10";
            default: return "text-gray-300 border-gray-600 bg-gray-700/30";
        }
    };

    return (
        <div className="w-80 bg-candy-card border-r border-gray-800 p-4 hidden md:flex flex-col h-full">
            <h1 className="text-2xl font-bold text-pink-500 mb-4 flex items-center gap-2">
                MyCandyLocal
            </h1>

            <button
                onClick={onCreate}
                className="mb-4 w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-pink-600/80 hover:bg-pink-500 transition text-sm font-medium shadow"
            >
                <PlusCircle size={18} />
                Create character
            </button>

            <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                <h2 className="text-gray-400 text-xs uppercase font-semibold mb-2">Voices</h2>
                {characters.map((char) => (
                    <div
                        key={char.id}
                        className={`w-full p-3 rounded-xl transition-all cursor-pointer ${selectedCharId === char.id
                            ? "bg-pink-600/20 border border-pink-500/50"
                            : "hover:bg-gray-800 border border-transparent"
                            }`}
                        onClick={() => onSelect(char.id)}
                    >
                        <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden shrink-0">
                                    {char.avatar_path ? (
                                        <img
                                            src={avatarUrl(char)}
                                            alt={char.name}
                                            className="w-full h-full object-cover"
                                            onError={(e) => {
                                                e.currentTarget.style.display = 'none';
                                                e.currentTarget.parentElement?.classList.add('fallback-user-icon');
                                            }}
                                        />
                                    ) : (
                                        <User size={20} />
                                    )}
                                </div>
                                <div className="text-left overflow-hidden">
                                    <div className="font-medium truncate">{char.name}</div>
                                    <div className="text-xs text-gray-400 truncate">
                                        {char.voice_model_path ? "RVC ready" : "No model yet"}
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onEdit(char);
                                }}
                                className="text-xs text-gray-300 border border-gray-700 px-2 py-1 rounded-lg hover:border-pink-500 shrink-0"
                            >
                                Edit
                            </button>
                        </div>
                        <div className="mt-2 flex items-center gap-2 text-xs">
                            <span
                                className={`px-2 py-1 rounded-full border ${statusTone(
                                    char.voice_training_status || ""
                                )}`}
                            >
                                {char.voice_training_status || "no status"}
                            </span>
                            {char.voice_training_status === "done" && <CheckCircle size={14} className="text-green-400" />}
                            {char.voice_training_status === "failed" && (
                                <AlertTriangle size={14} className="text-red-400" />
                            )}
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-auto pt-4 border-t border-gray-800">
                <div className="text-xs text-gray-500 truncate">Backend: {apiBase}</div>
            </div>
        </div>
    );
}
