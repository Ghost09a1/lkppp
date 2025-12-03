import { useState, useEffect, useRef, useMemo } from 'react';
import { apiClient } from './api/client';
import { Character, Message, ServiceStatus } from './types';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

// Components
import TopBar from './components/TopBar';
import ChatPanel from './components/ChatPanel';
import Composer from './components/Composer';
import RightSidebar from './components/RightSidebar';
import CharacterSidebar from './components/CharacterSidebar';
import CharacterEditor from './components/CharacterEditor';
import { MediaGenModal } from './components/MediaGenModal';
import AdvancedSettings from './components/AdvancedSettings';
import PromptLibrary from './components/PromptLibrary';
import MessageSearch from './components/MessageSearch';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  // State
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharId, setSelectedCharId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showRightSidebar, setShowRightSidebar] = useState(false);
  const [status, setStatus] = useState<ServiceStatus | null>(null);

  // Modals
  const [showCharEditor, setShowCharEditor] = useState(false);
  const [editingChar, setEditingChar] = useState<Character | null>(null);
  const [showMediaModal, setShowMediaModal] = useState(false);
  const [mediaType] = useState<'image' | 'video'>('image');
  const [showSettings, setShowSettings] = useState(false);
  const [showPrompts, setShowPrompts] = useState(false);
  const [showSearch, setShowSearch] = useState(false);

  // TTS Settings
  const [autoTTS, setAutoTTS] = useState(() => {
    const saved = localStorage.getItem('mycandy_auto_tts');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [autoImage, setAutoImage] = useState(() => {
    const saved = localStorage.getItem('mycandy_auto_image');
    return saved !== null ? JSON.parse(saved) : false;
  });
  const [currentAudioId, setCurrentAudioId] = useState<number | null>(null);

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Derived
  const selectedChar = useMemo(() =>
    characters.find(c => c.id === selectedCharId) || null,
    [characters, selectedCharId]
  );

  // Initial Load
  useEffect(() => {
    loadCharacters();
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Reset messages on char switch
  useEffect(() => {
    if (selectedCharId) {
      setMessages([{
        id: Date.now(),
        role: 'assistant',
        content: `Ready to chat with ${selectedChar?.name || 'character'}.`
      }]);
    }
  }, [selectedCharId]);

  const loadCharacters = async () => {
    try {
      const list = await apiClient.getCharacters();
      setCharacters(list);
      if (list.length > 0 && !selectedCharId) {
        setSelectedCharId(list[0].id);
      }
    } catch (err) {
      console.error("Failed to load characters", err);
    }
  };

  const checkStatus = async () => {
    try {
      const s = await apiClient.getStatus();
      setStatus(s);
    } catch (err) {
      console.warn("Status check failed", err);
    }
  };

  // Helper to close all modals
  const closeAllModals = () => {
    setShowCharEditor(false);
    setShowMediaModal(false);
    setShowSettings(false);
    setShowPrompts(false);
    setShowSearch(false);
  };

  // Chat Logic
  const handleSend = async (textOverride?: string | any) => {
    // [HOTFIX] Ensure textOverride is a string (ignore event objects from onClick)
    const actualText = typeof textOverride === 'string' ? textOverride : undefined;
    const textToUse = actualText || input;

    if (!textToUse.trim() || !selectedCharId) return;

    const text = textToUse.trim();
    if (!actualText) {
      setInput("");
    }
    setIsSending(true);

    // Optimistic User Message
    const userMsg: Message = {
      id: Date.now(),
      role: 'user',
      content: text
    };
    setMessages(prev => [...prev, userMsg]);

    try {
      const res = await apiClient.sendMessage(selectedCharId, text, autoTTS, autoImage);

      // AI Message
      const aiMsg: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: res.reply || "(No response)",
        audio_base64: res.audio_base64,
        image_base64: res.image_base64
      };
      setMessages(prev => [...prev, aiMsg]);

      // TTS (auto-play if enabled)
      if (res.audio_base64 && autoTTS) {
        playAudio(res.audio_base64, aiMsg.id);
      }

    } catch (err) {
      console.error("Send failed", err);
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: "Error: Could not send message."
      }]);
    } finally {
      setIsSending(false);
    }
  };

  const playAudio = (b64: string, messageId?: number) => {
    console.log("Attempting to play audio", { messageId, length: b64?.length });
    // Stop any currently playing audio
    stopAudio();

    // Set current playing ID
    if (messageId) {
      setCurrentAudioId(messageId);
    }

    // [HOTFIX] Clean base64 string to prevent static/errors
    const cleanB64 = b64.trim().replace(/[\r\n]/g, '');
    const url = cleanB64.startsWith('data:') ? cleanB64 : `data:audio/wav;base64,${cleanB64}`;
    const audio = new Audio(url);
    audioRef.current = audio;

    audio.onended = () => {
      setCurrentAudioId(null);
      audioRef.current = null;
    };

    audio.onerror = (e) => {
      console.error("Audio play failed", e);
      setCurrentAudioId(null);
      audioRef.current = null;
    };

    audio.play().catch(e => {
      console.error("Audio play failed", e);
      setCurrentAudioId(null);
    });
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setCurrentAudioId(null);
  };

  const toggleAutoTTS = () => {
    const newValue = !autoTTS;
    setAutoTTS(newValue);
    localStorage.setItem('mycandy_auto_tts', JSON.stringify(newValue));
  };

  const toggleAutoImage = () => {
    const newValue = !autoImage;
    setAutoImage(newValue);
    localStorage.setItem('mycandy_auto_image', JSON.stringify(newValue));
  };

  // STT Logic
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach(t => t.stop());

        try {
          const res = await apiClient.transcribeAudio(blob, selectedChar?.language);
          console.log("STT Result:", res.text);
          console.log("[STT] Raw response:", res);

          const transcript = (res.text || "").trim();
          console.log("[STT] Transcript (JSON):", JSON.stringify(transcript), "length:", transcript.length);

          // [HOTFIX] Filter BEFORE creating message - only block empty or single dot
          if (!transcript || transcript === "." || !selectedCharId) {
            console.log("[STT] Ignored transcript:", JSON.stringify(transcript));
            return;
          }

          // Now create user message with valid transcript
          const userMsg: Message = {
            id: Date.now(),
            role: 'user',
            content: transcript
          };
          console.log("[STT] Creating message:", userMsg);
          setMessages(prev => [...prev, userMsg]);
          setIsSending(true);

          try {
            const llmRes = await apiClient.sendMessage(selectedCharId, transcript, autoTTS, autoImage);
            const aiMsg: Message = {
              id: Date.now() + 1,
              role: 'assistant',
              content: llmRes.reply || "(No response)",
              audio_base64: llmRes.audio_base64,
              image_base64: llmRes.image_base64
            };
            setMessages(prev => [...prev, aiMsg]);

            if (llmRes.audio_base64 && autoTTS) {
              playAudio(llmRes.audio_base64, aiMsg.id);
            }
          } catch (err) {
            console.error("LLM response failed", err);
            setMessages(prev => [...prev, {
              id: Date.now(),
              role: 'assistant',
              content: "Error: Could not get response."
            }]);
          } finally {
            setIsSending(false);
          }
        } catch (err) {
          console.error("STT failed", err);
          // [HOTFIX] No alert for STT failures (silence/empty is normal)
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Mic access failed", err);
      alert("Microphone access denied");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Media Logic
  // Media Logic
  const handleGenerateMedia = async (prompt: string, negative?: string) => {
    if (!selectedCharId) return;

    // Optimistic message
    const tempId = Date.now();
    setMessages(prev => [...prev, {
      id: tempId,
      role: 'assistant',
      content: `Generating ${mediaType}...`
    }]);

    try {
      let res;
      if (mediaType === 'image') {
        res = await apiClient.generateImage(selectedCharId, prompt, negative);
      } else {
        throw new Error("Video generation not fully implemented in client");
      }

      // Update message with image
      setMessages(prev => prev.map(m => {
        if (m.id === tempId) {
          return {
            ...m,
            content: res.prompt || "Generated Image",
            image_base64: res.image_base64 ? (res.image_base64.startsWith('data:') ? res.image_base64 : `data:image/png;base64,${res.image_base64}`) : undefined
          };
        }
        return m;
      }));

    } catch (err) {
      console.error("Generation failed", err);
      setMessages(prev => prev.map(m => {
        if (m.id === tempId) {
          return { ...m, content: "Generation failed." };
        }
        return m;
      }));
    }
  };

  // Keyboard Shortcuts
  useKeyboardShortcuts([
    { key: 'Escape', handler: closeAllModals, description: 'Close all modals' },
    { key: 'k', ctrl: true, handler: () => { closeAllModals(); setShowSearch(true); }, description: 'Search messages' },
    { key: '/', ctrl: true, handler: () => { closeAllModals(); setShowPrompts(true); }, description: 'Open prompt library' },
    { key: 'Enter', ctrl: true, handler: handleSend, description: 'Send message' }
  ], true);

  return (
    <div className="flex h-screen bg-candy-dark text-white font-sans overflow-hidden">
      {/* Left Sidebar (Characters) */}
      <CharacterSidebar
        characters={characters}
        selectedCharId={selectedCharId}
        onSelect={setSelectedCharId}
        onEdit={(char) => { setEditingChar(char); closeAllModals(); setShowCharEditor(true); }}
        onCreate={() => { setEditingChar(null); closeAllModals(); setShowCharEditor(true); }}
        apiBase={API_URL}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 bg-gray-900/50 relative">
        <TopBar
          character={selectedChar}
          status={status}
          onSearch={() => { closeAllModals(); setShowSearch(true); }}
          onSettings={() => { closeAllModals(); setShowSettings(true); }}
          apiBase={API_URL}
        />

        <ChatPanel
          messages={messages}
          loading={isSending}
          currentAudioId={currentAudioId}
          onStopAudio={stopAudio}
          onPlayAudio={playAudio}
        />

        <Composer
          input={input}
          setInput={setInput}
          onSend={handleSend}
          onRecordStart={startRecording}
          onRecordStop={stopRecording}
          isRecording={isRecording}
          isSending={isSending}
          onImageClick={() => { closeAllModals(); setShowMediaModal(true); }}
          onTogglePrompts={() => { closeAllModals(); setShowPrompts(prev => !prev); }}
          autoTTS={autoTTS}
          onToggleTTS={toggleAutoTTS}
          autoImage={autoImage}
          onToggleAutoImage={toggleAutoImage}
          inputRef={inputRef}
        />
      </div>

      {/* Right Sidebar (Info/Media) */}
      <div className={`fixed inset-y-0 right-0 z-40 w-80 transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0 ${showRightSidebar ? 'translate-x-0' : 'translate-x-full'}`}>
        <RightSidebar
          character={selectedChar}
          onClose={() => setShowRightSidebar(false)}
        />
      </div>

      {/* Modals */}
      {showCharEditor && (
        <CharacterEditor
          character={editingChar}
          onClose={() => setShowCharEditor(false)}
          onSave={loadCharacters}
        />
      )}

      <MediaGenModal
        isOpen={showMediaModal}
        onClose={() => setShowMediaModal(false)}
        type={mediaType}
        onGenerate={handleGenerateMedia}
      />

      {showSettings && (
        <AdvancedSettings
          onClose={() => setShowSettings(false)}
        />
      )}

      {showPrompts && (
        <PromptLibrary
          onClose={() => setShowPrompts(false)}
          onSelectPrompt={(text) => {
            setInput(prev => prev + (prev ? ' ' : '') + text);
            inputRef.current?.focus();
            setShowPrompts(false);
          }}
        />
      )}

      {showSearch && (
        <MessageSearch
          onClose={() => setShowSearch(false)}
          messages={messages}
          onSelectMessage={(_id) => {
            setShowSearch(false);
          }}
        />
      )}
    </div>
  );
}
