import { useState } from 'react';
import { X } from 'lucide-react';
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
