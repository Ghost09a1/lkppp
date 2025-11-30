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
