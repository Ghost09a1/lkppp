import { useEffect } from 'react';

type KeyboardHandler = (event: KeyboardEvent) => void;

interface KeyboardShortcut {
    key: string;
    ctrl?: boolean;
    shift?: boolean;
    alt?: boolean;
    handler: () => void;
    description?: string;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], enabled = true) {
    useEffect(() => {
        if (!enabled) return;

        const handleKeyDown: KeyboardHandler = (event) => {
            for (const shortcut of shortcuts) {
                const ctrlMatch = shortcut.ctrl === undefined || shortcut.ctrl === event.ctrlKey;
                const shiftMatch = shortcut.shift === undefined || shortcut.shift === event.shiftKey;
                const altMatch = shortcut.alt === undefined || shortcut.alt === event.altKey;
                const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

                if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
                    event.preventDefault();
                    shortcut.handler();
                    break;
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [shortcuts, enabled]);
}
