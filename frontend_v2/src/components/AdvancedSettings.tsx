import { useState, useEffect } from 'react';
import { Settings, X } from 'lucide-react';

interface AdvancedSettingsProps {
    onClose: () => void;
}

interface LLMSettings {
    temperature: number;
    topP: number;
    maxTokens: number;
    frequencyPenalty: number;
    presencePenalty: number;
}

const SETTINGS_KEY = 'mycandy_llm_settings';

const defaultSettings: LLMSettings = {
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 2048,
    frequencyPenalty: 0,
    presencePenalty: 0
};

export default function AdvancedSettings({ onClose }: AdvancedSettingsProps) {
    const [temperature, setTemperature] = useState(defaultSettings.temperature);
    const [topP, setTopP] = useState(defaultSettings.topP);
    const [maxTokens, setMaxTokens] = useState(defaultSettings.maxTokens);
    const [frequencyPenalty, setFrequencyPenalty] = useState(defaultSettings.frequencyPenalty);
    const [presencePenalty, setPresencePenalty] = useState(defaultSettings.presencePenalty);

    // Load settings from localStorage on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem(SETTINGS_KEY);
            if (saved) {
                const settings = JSON.parse(saved) as LLMSettings;
                setTemperature(settings.temperature);
                setTopP(settings.topP);
                setMaxTokens(settings.maxTokens);
                setFrequencyPenalty(settings.frequencyPenalty);
                setPresencePenalty(settings.presencePenalty);
            }
        } catch (err) {
            console.error('Failed to load settings:', err);
        }
    }, []);

    const handleSave = () => {
        try {
            const settings: LLMSettings = {
                temperature,
                topP,
                maxTokens,
                frequencyPenalty,
                presencePenalty
            };
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
            console.log('Settings saved:', settings);
            onClose();
        } catch (err) {
            console.error('Failed to save settings:', err);
            alert('Failed to save settings');
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-lg rounded-lg bg-candy-card p-6 shadow-2xl">
                <div className="mb-6 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Settings size={20} className="text-candy-pink" />
                        <h2 className="text-xl font-bold">Advanced Settings</h2>
                    </div>
                    <button onClick={onClose} className="rounded-lg p-2 hover:bg-gray-700">
                        <X size={20} />
                    </button>
                </div>

                <div className="space-y-6">
                    {/* Temperature */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <label className="text-sm font-medium">Temperature</label>
                            <span className="text-sm text-gray-400">{temperature.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.05"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <p className="mt-1 text-xs text-gray-500">
                            Higher values make output more random, lower values more focused
                        </p>
                    </div>

                    {/* Top P */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <label className="text-sm font-medium">Top P</label>
                            <span className="text-sm text-gray-400">{topP.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={topP}
                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <p className="mt-1 text-xs text-gray-500">
                            Nucleus sampling - alternative to temperature
                        </p>
                    </div>

                    {/* Max Tokens */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <label className="text-sm font-medium">Max Tokens</label>
                            <span className="text-sm text-gray-400">{maxTokens}</span>
                        </div>
                        <input
                            type="range"
                            min="256"
                            max="4096"
                            step="128"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                            className="w-full"
                        />
                        <p className="mt-1 text-xs text-gray-500">
                            Maximum length of the response
                        </p>
                    </div>

                    {/* Frequency Penalty */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <label className="text-sm font-medium">Frequency Penalty</label>
                            <span className="text-sm text-gray-400">{frequencyPenalty.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={frequencyPenalty}
                            onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <p className="mt-1 text-xs text-gray-500">
                            Reduces repetition of frequent tokens
                        </p>
                    </div>

                    {/* Presence Penalty */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <label className="text-sm font-medium">Presence Penalty</label>
                            <span className="text-sm text-gray-400">{presencePenalty.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={presencePenalty}
                            onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <p className="mt-1 text-xs text-gray-500">
                            Encourages talking about new topics
                        </p>
                    </div>

                    {/* Actions */}
                    <div className="flex justify-end gap-2 border-t border-gray-800 pt-4">
                        <button onClick={onClose} className="btn-secondary">
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="btn-primary"
                        >
                            Save Settings
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
