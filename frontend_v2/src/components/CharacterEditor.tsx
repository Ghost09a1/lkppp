import { useState } from 'react';
import { X, Mic, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { Character } from '../types';
import { apiClient } from '../api/client';

interface CharacterEditorProps {
    character: Character | null;
    onClose: () => void;
    onSave: () => void;
}

export default function CharacterEditor({ character, onClose, onSave }: CharacterEditorProps) {
    const [formData, setFormData] = useState<Partial<Character>>(
        character || {
            name: '',
            description: '',
            personality: '',
            backstory: '',
            language: 'en',
            voice_style: 'breathy-female',
        }
    );
    const [saving, setSaving] = useState(false);
    const [refImages, setRefImages] = useState<{ id: number; url: string }[]>(
        character?.reference_images || []
    );
    // Voice Training State
    const [voiceFile, setVoiceFile] = useState<File | null>(null);
    const [trainingStatus, setTrainingStatus] = useState(character?.voice_training_status || '');
    const [isTraining, setIsTraining] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setSaving(true);

        try {
            if (character?.id) {
                await apiClient.updateCharacter(character.id, formData);
            } else {
                await apiClient.createCharacter(formData);
            }
            onSave();
            onClose();
        } catch (err) {
            console.error('Save failed:', err);
            alert('Failed to save character');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-2xl rounded-lg bg-candy-card p-6 shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                    <h2 className="text-xl font-bold">
                        {character ? 'Edit Character' : 'New Character'}
                    </h2>
                    <button onClick={onClose} className="rounded-lg p-2 hover:bg-gray-700">
                        <X size={20} />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Avatar Upload Section */}
                    {character?.id && (
                        <div className="flex items-center gap-4 pb-4 border-b border-gray-700">
                            <div className="relative group">
                                <div className="w-16 h-16 rounded-full bg-gray-700 overflow-hidden">
                                    {character.avatar_path ? (
                                        <img
                                            src={`${import.meta.env.VITE_API_URL || ''}/avatars/${character.avatar_path}`}
                                            alt="Avatar"
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center text-gray-500 text-2xl">
                                            {(character.name || '?')[0].toUpperCase()}
                                        </div>
                                    )}
                                </div>
                                <label className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full opacity-0 group-hover:opacity-100 cursor-pointer transition-opacity">
                                    <span className="text-white text-xs">Change</span>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        className="hidden"
                                        onChange={async (e) => {
                                            const file = e.target.files?.[0];
                                            if (file && character.id) {
                                                try {
                                                    await apiClient.uploadAvatar(character.id, file);
                                                    onSave(); // Trigger refresh
                                                } catch (err) {
                                                    alert('Avatar upload failed');
                                                }
                                            }
                                        }}
                                    />
                                </label>
                            </div>
                            <div className="text-sm text-gray-400">
                                <p>Click avatar to change</p>
                                <p className="text-xs">(PNG/JPEG, 100x100px)</p>
                            </div>
                        </div>
                    )}

                    <div>
                        <label className="mb-1 block text-sm font-medium">Name *</label>
                        <input
                            type="text"
                            value={formData.name || ''}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            className="input-field"
                            required
                        />
                    </div>

                    <div>
                        <label className="mb-1 block text-sm font-medium">Description</label>
                        <textarea
                            value={formData.description || ''}
                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                            className="input-field"
                            rows={3}
                        />
                    </div>

                    <div>
                        <label className="mb-1 block text-sm font-medium">Personality</label>
                        <textarea
                            value={formData.personality || ''}
                            onChange={(e) => setFormData({ ...formData, personality: e.target.value })}
                            className="input-field"
                            rows={4}
                            placeholder="Describe the character's personality traits..."
                        />
                    </div>

                    <div>
                        <label className="mb-1 block text-sm font-medium">Backstory</label>
                        <textarea
                            value={formData.backstory || ''}
                            onChange={(e) => setFormData({ ...formData, backstory: e.target.value })}
                            className="input-field"
                            rows={4}
                            placeholder="Character's background and history..."
                        />
                    </div>

                    <div className="border-t border-gray-700 pt-4">
                        <label className="mb-2 block text-sm font-medium text-candy-pink">Image Generation Settings</label>

                        <div className="mb-4">
                            <label className="mb-1 block text-sm font-medium text-gray-400">Positive Image Prompt (Visual Style)</label>
                            <textarea
                                value={formData.visual_style || ''}
                                onChange={(e) => setFormData({ ...formData, visual_style: e.target.value })}
                                className="input-field text-xs font-mono"
                                rows={3}
                                placeholder="e.g. 1girl, red hair, blue eyes, cinematic lighting..."
                            />
                        </div>

                        <div>
                            <label className="mb-1 block text-sm font-medium text-gray-400">Negative Image Prompt</label>
                            <textarea
                                value={formData.negative_prompt || ''}
                                onChange={(e) => setFormData({ ...formData, negative_prompt: e.target.value })}
                                className="input-field text-xs font-mono"
                                rows={3}
                                placeholder="e.g. bad anatomy, text, watermark, low quality..."
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="mb-1 block text-sm font-medium">Language</label>
                            <select
                                value={formData.language || 'en'}
                                onChange={(e) => setFormData({ ...formData, language: e.target.value })}
                                className="input-field"
                            >
                                <option value="en">English</option>
                                <option value="de">German</option>
                                <option value="es">Spanish</option>
                                <option value="fr">French</option>
                            </select>
                        </div>

                        <div>
                            <label className="mb-1 block text-sm font-medium">Voice Style</label>
                            <select
                                value={formData.voice_style || 'breathy-female'}
                                onChange={(e) => setFormData({ ...formData, voice_style: e.target.value })}
                                className="input-field"
                            >
                                <option value="breathy-female">Breathy Female</option>
                                <option value="neutral-female">Neutral Female</option>
                                <option value="neutral-male">Neutral Male</option>
                            </select>
                        </div>
                    </div>

                    {/* Voice Training Section */}
                    {character?.id && (
                        <div className="border-t border-gray-700 pt-4">
                            <label className="mb-2 block text-sm font-medium text-candy-pink flex items-center gap-2">
                                <Mic size={16} />
                                Voice Training (RVC)
                            </label>

                            <div className="space-y-3">
                                {/* Current Status */}
                                <div className="flex items-center gap-2 text-sm">
                                    <span className="text-gray-400">Status:</span>
                                    {trainingStatus === 'done' && (
                                        <span className="flex items-center gap-1 text-green-400">
                                            <CheckCircle size={14} /> Ready
                                        </span>
                                    )}
                                    {trainingStatus === 'running' && (
                                        <span className="flex items-center gap-1 text-pink-400 animate-pulse">
                                            <Loader2 size={14} className="animate-spin" /> Training...
                                        </span>
                                    )}
                                    {trainingStatus === 'queued' && (
                                        <span className="flex items-center gap-1 text-amber-400">
                                            <Loader2 size={14} /> Queued
                                        </span>
                                    )}
                                    {trainingStatus === 'failed' && (
                                        <span className="flex items-center gap-1 text-red-400">
                                            <AlertCircle size={14} /> Failed
                                        </span>
                                    )}
                                    {!trainingStatus && (
                                        <span className="text-gray-500">No model yet</span>
                                    )}
                                </div>

                                {/* Upload Voice Sample */}
                                <div className="flex items-center gap-2">
                                    <label className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg border-2 border-dashed border-gray-600 hover:border-candy-pink hover:bg-white/5 cursor-pointer transition-colors">
                                        <Mic size={16} />
                                        <span className="text-sm">
                                            {voiceFile ? voiceFile.name : 'Upload Voice Sample (MP3/WAV)'}
                                        </span>
                                        <input
                                            type="file"
                                            accept="audio/*,.mp3,.wav,.m4a"
                                            className="hidden"
                                            onChange={async (e) => {
                                                const file = e.target.files?.[0];
                                                if (file) {
                                                    setVoiceFile(file);
                                                    try {
                                                        await apiClient.uploadVoiceSample(character.id, file);
                                                        alert('Voice sample uploaded!');
                                                    } catch (err) {
                                                        alert('Upload failed');
                                                    }
                                                }
                                            }}
                                        />
                                    </label>
                                </div>

                                {/* Train Button */}
                                <button
                                    type="button"
                                    disabled={isTraining || trainingStatus === 'running'}
                                    onClick={async () => {
                                        setIsTraining(true);
                                        setTrainingStatus('queued');
                                        try {
                                            await apiClient.trainVoice(character.id);
                                            setTrainingStatus('running');
                                            // Poll for status (simplified - in production use WebSocket)
                                            alert('Training started! Check status in a few minutes.');
                                        } catch (err) {
                                            setTrainingStatus('failed');
                                            alert('Training failed to start');
                                        } finally {
                                            setIsTraining(false);
                                        }
                                    }}
                                    className="w-full py-2 px-4 rounded-lg bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 disabled:from-gray-700 disabled:to-gray-700 disabled:text-gray-500 font-medium transition-all flex items-center justify-center gap-2"
                                >
                                    {isTraining ? (
                                        <>
                                            <Loader2 size={16} className="animate-spin" />
                                            Starting...
                                        </>
                                    ) : (
                                        <>
                                            <Mic size={16} />
                                            Train Voice Model
                                        </>
                                    )}
                                </button>

                                <p className="text-xs text-gray-500">
                                    Upload 5-10 minutes of clean voice audio. Training takes ~10-30 minutes.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Reference Images Section */}
                    {character?.id && (
                        <div className="border-t border-gray-700 pt-4">
                            <label className="mb-2 block text-sm font-medium">Reference Images (Max 5)</label>
                            <div className="mb-4 grid grid-cols-5 gap-2">
                                {refImages.map((img) => (
                                    <div key={img.id} className="relative group aspect-[2/3] overflow-hidden rounded-lg bg-black/20">
                                        <img src={`${import.meta.env.VITE_API_URL || ''}${img.url}`} alt="Ref" className="h-full w-full object-cover" />
                                        <button
                                            type="button"
                                            onClick={async () => {
                                                if (confirm('Delete this reference image?')) {
                                                    try {
                                                        await apiClient.deleteReferenceImage(character.id, img.id);
                                                        setRefImages(prev => prev.filter(p => p.id !== img.id));
                                                        onSave(); // Trigger parent refresh too
                                                    } catch (err) {
                                                        alert('Delete failed');
                                                    }
                                                }
                                            }}
                                            className="absolute right-1 top-1 rounded-full bg-red-500/80 p-1 text-white opacity-0 transition-opacity group-hover:opacity-100"
                                        >
                                            <X size={12} />
                                        </button>
                                    </div>
                                ))}
                                {refImages.length < 5 && (
                                    <label className="flex aspect-[2/3] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-600 hover:border-candy-pink hover:bg-white/5">
                                        <span className="text-2xl">+</span>
                                        <span className="text-xs">Upload</span>
                                        <input
                                            type="file"
                                            accept="image/*"
                                            className="hidden"
                                            onChange={async (e) => {
                                                const file = e.target.files?.[0];
                                                if (file) {
                                                    try {
                                                        const res = await apiClient.uploadReferenceImage(character.id, file);
                                                        if (res.ok && res.url && res.id) {
                                                            setRefImages(prev => [...prev, { id: res.id, url: res.url }]);
                                                            onSave(); // Trigger parent refresh too
                                                        }
                                                    } catch (err) {
                                                        alert('Upload failed');
                                                    }
                                                }
                                            }}
                                        />
                                    </label>
                                )}
                            </div>
                            <p className="text-xs text-gray-400">These images will be used by ComfyUI (IPAdapter) to keep character consistency.</p>
                        </div>
                    )}

                    <div className="flex justify-end gap-2 pt-4">
                        <button type="button" onClick={onClose} className="btn-secondary">
                            Cancel
                        </button>
                        <button type="submit" disabled={saving} className="btn-primary">
                            {saving ? 'Saving...' : 'Save'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
