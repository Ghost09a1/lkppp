import { useState } from 'react';
import { Search, X } from 'lucide-react';
import { Message } from '../types';

interface MessageSearchProps {
    messages: Message[];
    onSelectMessage: (messageId: number) => void;
    onClose: () => void;
}

export default function MessageSearch({ messages, onSelectMessage, onClose }: MessageSearchProps) {
    const [query, setQuery] = useState('');

    const filtered = messages.filter((msg) =>
        msg.content.toLowerCase().includes(query.toLowerCase())
    );

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-2xl rounded-lg bg-candy-card p-6 shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Search size={20} className="text-candy-pink" />
                        <h2 className="text-xl font-bold">Search Messages</h2>
                    </div>
                    <button onClick={onClose} className="rounded-lg p-2 hover:bg-gray-700">
                        <X size={20} />
                    </button>
                </div>

                {/* Search Input */}
                <div className="relative mb-4">
                    <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search in conversation..."
                        className="input-field pl-10"
                        autoFocus
                    />
                </div>

                {/* Results */}
                <div className="max-h-96 space-y-2 overflow-y-auto">
                    {query.length < 2 ? (
                        <p className="text-center text-sm text-gray-500 py-8">
                            Type at least 2 characters to search
                        </p>
                    ) : filtered.length === 0 ? (
                        <p className="text-center text-sm text-gray-500 py-8">No messages found</p>
                    ) : (
                        filtered.map((msg) => (
                            <button
                                key={msg.id}
                                onClick={() => {
                                    onSelectMessage(msg.id);
                                    onClose();
                                }}
                                className="w-full rounded-lg bg-gray-800 p-3 text-left hover:bg-gray-700"
                            >
                                <div className="mb-1 flex items-center gap-2">
                                    <span
                                        className={`text-xs font-medium ${msg.role === 'user' ? 'text-candy-pink' : 'text-blue-400'
                                            }`}
                                    >
                                        {msg.role === 'user' ? 'You' : 'Assistant'}
                                    </span>
                                    {msg.timestamp && (
                                        <span className="text-xs text-gray-500">
                                            {new Date(msg.timestamp).toLocaleString()}
                                        </span>
                                    )}
                                </div>
                                <p className="text-sm text-gray-300 line-clamp-2">{msg.content}</p>
                            </button>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
