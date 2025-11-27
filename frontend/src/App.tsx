import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { Send, Mic, Play, Volume2, User, Settings } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Message {
  id: number;
  sender: "user" | "ai";
  text: string;
  audioUrl?: string;
  duration?: number;
}

interface Character {
  id: string;
  name: string;
  image: string;
}

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, sender: "ai", text: "Hey! Ich bin bereit. Womit starten wir?" },
  ]);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedChar, setSelectedChar] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    async function fetchChars() {
      try {
        const res = await axios.get(`${API_URL}/api/characters`);
        setCharacters(res.data);
        if (res.data.length > 0) setSelectedChar(res.data[0].id);
      } catch (err) {
        console.error("Backend nicht erreichbar?", err);
      }
    }
    fetchChars();
  }, []);

  const sendMessage = async (audioFile?: File) => {
    if (!input.trim() && !audioFile) return;
    if (isProcessing) return;

    const userMsg: Message = {
      id: Date.now(),
      sender: "user",
      text: input || (audioFile ? "(Audio gesendet)" : ""),
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!audioFile) setInput("");
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("text", userMsg.text);
      formData.append("character_id", selectedChar || "char_1");
      if (audioFile) {
        formData.append("audio", audioFile, audioFile.name);
      }

      const res = await axios.post(`${API_URL}/api/chat`, formData);

      const aiMsg: Message = {
        id: Date.now() + 1,
        sender: "ai",
        text: res.data.text,
        audioUrl: res.data.audio_url ? `${API_URL}${res.data.audio_url}` : undefined,
      };

      setMessages((prev) => [...prev, aiMsg]);

      if (aiMsg.audioUrl) {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }
        const audio = new Audio(aiMsg.audioUrl);
        audioRef.current = audio;
        audio.onloadedmetadata = () => {
          setMessages((prev) =>
            prev.map((m) => (m.id === aiMsg.id ? { ...m, duration: audio.duration } : m))
          );
        };
        audio.play().catch((err) => console.warn("Autoplay failed", err));
      }
    } catch (error) {
      console.error("Fehler im Chat:", error);
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), sender: "ai", text: "Fehler bei der Verbindung zum Backend." },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen bg-candy-dark text-white font-sans overflow-hidden">
      <div className="w-80 bg-candy-card border-r border-gray-800 p-4 hidden md:flex flex-col">
        <h1 className="text-2xl font-bold text-pink-500 mb-6 flex items-center gap-2">
          <Volume2 /> MyCandyLocal
        </h1>

        <div className="flex-1 overflow-y-auto space-y-2">
          <h2 className="text-gray-400 text-xs uppercase font-semibold mb-2">Verfügbare Stimmen</h2>
          {characters.map((char) => (
            <button
              key={char.id}
              onClick={() => setSelectedChar(char.id)}
              className={`w-full flex items-center p-3 rounded-xl transition-all ${
                selectedChar === char.id
                  ? "bg-pink-600/20 border border-pink-500/50"
                  : "hover:bg-gray-800"
              }`}
            >
              <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center mr-3 overflow-hidden">
                <User size={20} />
              </div>
              <div className="text-left">
                <div className="font-medium">{char.name}</div>
                <div className="text-xs text-gray-400">RVC v2</div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-auto pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-500">Backend: {API_URL}</div>
        </div>
      </div>

      <div className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-800 to-candy-dark">
        <div className="p-4 border-b border-gray-800/50 flex justify-between items-center md:hidden">
          <span className="font-bold">Chat</span>
          <Settings size={20} className="text-gray-400" />
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] md:max-w-[60%] p-4 rounded-2xl shadow-lg backdrop-blur-sm ${
                  msg.sender === "user"
                    ? "bg-pink-600 text-white rounded-br-sm"
                    : "bg-gray-800/80 border border-gray-700 text-gray-100 rounded-bl-sm"
                }`}
              >
                <p className="leading-relaxed">{msg.text}</p>
                {msg.audioUrl && (
                  <div className="mt-3 flex items-center gap-2 bg-black/20 p-2 rounded-lg">
                    <button
                      onClick={() => {
                        if (audioRef.current) {
                          audioRef.current.pause();
                          audioRef.current = null;
                        }
                        const audio = new Audio(msg.audioUrl!);
                        audioRef.current = audio;
                        audio.play().catch((err) => console.warn("play failed", err));
                      }}
                      className="p-2 bg-pink-500 rounded-full hover:scale-105 transition"
                    >
                      <Play size={12} fill="currentColor" />
                    </button>
                    <div className="h-1 flex-1 bg-gray-600 rounded-full overflow-hidden">
                      <div className="h-full w-full bg-pink-400 animate-pulse" />
                    </div>
                    <span className="text-xs text-gray-400">
                      {msg.duration ? msg.duration.toFixed(1) + "s" : "audio"}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-gray-800/50 p-4 rounded-2xl rounded-bl-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 md:p-6 pb-8">
          <div className="max-w-4xl mx-auto relative flex items-center bg-gray-800/90 backdrop-blur border border-gray-700 rounded-full p-2 pr-2 pl-6 shadow-2xl">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Schreib eine Nachricht..."
              className="flex-1 bg-transparent border-none outline-none text-white placeholder-gray-500 h-10"
              disabled={isProcessing}
            />
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="p-2 text-gray-400 hover:text-pink-500 transition"
                onClick={() => fileInputRef.current?.click()}
              >
                <Mic size={20} />
              </button>
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isProcessing}
                className="p-3 bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-full transition-all shadow-lg hover:shadow-pink-500/25"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
          <div className="text-center mt-2 text-xs text-gray-600">Powered by RVC Local & FastAPI</div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              sendMessage(file);
              e.target.value = "";
            }
          }}
        />
      </div>
    </div>
  );
}

export default App;
