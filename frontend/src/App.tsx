import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  Send,
  Mic,
  Play,
  Volume2,
  VolumeX,
  User,
  Settings,
  PlusCircle,
  Image as ImageIcon,
  CheckCircle,
  AlertTriangle,
  Loader2,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const audioMarkersRegex = /<\|audio_start\|>|<\|audio_end\|>/gi;
const customTokenRegex = /<custom_token_\d+>/gi;

const cleanReplyText = (raw: string | undefined | null) => {
  if (!raw) return "";
  return raw.replace(audioMarkersRegex, "").replace(customTokenRegex, "").replace(/\s{2,}/g, " ").trim();
};

type TrainingStatus = "" | "queued" | "running" | "done" | "failed";

interface Message {
  id: number;
  sender: "user" | "ai";
  text: string;
  audioUrl?: string;
  duration?: number;
  imageBase64?: string;
}

interface Character {
  id: number;
  name: string;
  description?: string;
  visual_style?: string;
  appearance_notes?: string;
  personality?: string;
  backstory?: string;
  relationship_type?: string;
  dos?: string;
  donts?: string;
  voice_style?: string;
  voice_pitch_shift?: number;
  voice_speed?: number;
  voice_ref_path?: string;
  voice_youtube_url?: string;
  voice_model_path?: string;
  voice_training_status?: TrainingStatus;
  voice_error?: string;
  language?: string;
}

interface CharacterFormState {
  id?: number;
  name: string;
  description: string;
  visual_style: string;
  appearance_notes: string;
  personality: string;
  backstory: string;
  relationship_type: string;
  dos: string;
  donts: string;
  voice_style: string;
  voice_youtube_url: string;
  language: string;
}

const emptyForm: CharacterFormState = {
  name: "",
  description: "",
  visual_style: "",
  appearance_notes: "",
  personality: "",
  backstory: "",
  relationship_type: "",
  dos: "",
  donts: "",
  voice_style: "",
  voice_youtube_url: "",
  language: "en",
};

function statusTone(status: TrainingStatus) {
  switch (status) {
    case "done":
      return "text-green-400 border-green-500/60 bg-green-500/10";
    case "running":
      return "text-pink-300 border-pink-500/60 bg-pink-500/10";
    case "queued":
      return "text-amber-300 border-amber-500/60 bg-amber-500/10";
    case "failed":
      return "text-red-300 border-red-500/60 bg-red-500/10";
    default:
      return "text-gray-300 border-gray-600 bg-gray-700/30";
  }
}

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, sender: "ai", text: "Hey! I am ready. What should we explore?" },
  ]);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharId, setSelectedCharId] = useState<number | null>(null);
  const [autoTts, setAutoTts] = useState(true);
  const [isSending, setIsSending] = useState(false);

  const [showEditor, setShowEditor] = useState(false);
  const [form, setForm] = useState<CharacterFormState>(emptyForm);
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [isSavingChar, setIsSavingChar] = useState(false);

  const [showImageModal, setShowImageModal] = useState(false);
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageNegative, setImageNegative] = useState("");
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);

  const [trainStatus, setTrainStatus] = useState<TrainingStatus>("");
  const [trainProgress, setTrainProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const trainTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const selectedCharacter = useMemo(
    () => characters.find((c) => c.id === selectedCharId) || null,
    [characters, selectedCharId]
  );

  const fetchCharacters = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/characters`);
      const list: Character[] = res.data?.characters || [];
      setCharacters(list);
      if (list.length > 0) {
        setSelectedCharId((prev) => prev ?? list[0].id);
      }
    } catch (err) {
      console.error("Backend not reachable?", err);
    }
  }, []);

  const fetchCharacter = useCallback(
    async (id: number) => {
      try {
        const res = await axios.get(`${API_URL}/characters/${id}`);
        const updated: Character = res.data;
        setCharacters((prev) =>
          prev.map((c) => (c.id === id ? { ...c, ...updated } : c))
        );
        setTrainStatus(updated.voice_training_status || "");
        if (updated.voice_training_status === "done") {
          setTrainProgress(100);
        }
        return updated;
      } catch (err) {
        console.error("Failed to fetch character", err);
        return null;
      }
    },
    []
  );

  useEffect(() => {
    fetchCharacters();
  }, [fetchCharacters]);

  useEffect(() => {
    return () => {
      if (trainTimer.current) clearInterval(trainTimer.current);
    };
  }, []);

  const handleTrainingPoll = useCallback(
    (id: number) => {
      if (trainTimer.current) clearInterval(trainTimer.current);
      trainTimer.current = setInterval(async () => {
        const updated = await fetchCharacter(id);
        const status = updated?.voice_training_status || "";
        if (status === "running") {
          setTrainProgress((p) => Math.min(95, p + Math.random() * 7 + 3));
        } else if (status === "done") {
          setTrainProgress(100);
          setIsTraining(false);
          if (trainTimer.current) clearInterval(trainTimer.current);
        } else if (status === "failed") {
          setTrainProgress(0);
          setIsTraining(false);
          if (trainTimer.current) clearInterval(trainTimer.current);
        }
      }, 1500);
    },
    [fetchCharacter]
  );

  const openEditor = (char?: Character) => {
    if (char) {
      setForm({
        id: char.id,
        name: char.name || "",
        description: char.description || "",
        visual_style: char.visual_style || "",
        appearance_notes: char.appearance_notes || "",
        personality: char.personality || "",
        backstory: char.backstory || "",
        relationship_type: char.relationship_type || "",
        dos: char.dos || "",
        donts: char.donts || "",
        voice_style: char.voice_style || "",
        voice_youtube_url: char.voice_youtube_url || "",
        language: char.language || "en",
      });
      setTrainStatus(char.voice_training_status || "");
      setTrainProgress(char.voice_training_status === "done" ? 100 : 0);
    } else {
      setForm(emptyForm);
      setTrainStatus("");
      setTrainProgress(0);
    }
    setVoiceFile(null);
    setShowEditor(true);
  };

  const saveCharacter = async () => {
    if (!form.name.trim()) {
      alert("Please enter a name.");
      return;
    }
    setIsSavingChar(true);
    try {
      let id = form.id;
      const payload = { ...form };
      if (form.id) {
        await axios.post(`${API_URL}/characters/${form.id}/update`, payload);
      } else {
        const res = await axios.post(`${API_URL}/characters`, payload);
        id = res.data?.id;
      }
      if (id && voiceFile) {
        const fd = new FormData();
        fd.append("file", voiceFile);
        await axios.post(`${API_URL}/characters/${id}/voice_dataset`, fd);
      }
      await fetchCharacters();
      if (id) setSelectedCharId(id);
      setShowEditor(false);
      setVoiceFile(null);
    } catch (err: any) {
      console.error("Failed to save character:", err);
      alert(err?.response?.data?.detail || "Could not save character.");
    } finally {
      setIsSavingChar(false);
    }
  };

  const startTraining = async () => {
    const id = form.id || selectedCharId;
    if (!id) {
      alert("Save the character first, then start training.");
      return;
    }
    setIsTraining(true);
    setTrainStatus("queued");
    setTrainProgress(5);
    try {
      await axios.post(`${API_URL}/characters/${id}/train_voice`);
      setTrainStatus("running");
      handleTrainingPoll(id);
    } catch (err: any) {
      console.error("Training failed to start", err);
      setTrainStatus("failed");
      setIsTraining(false);
    }
  };

  const generateImage = async () => {
    if (!imagePrompt.trim()) return;
    setIsGeneratingImage(true);
    try {
      const res = await axios.post(`${API_URL}/generate_image`, {
        prompt: imagePrompt,
        negative: imageNegative,
        steps: 24,
        width: 640,
        height: 832,
      });
      const img = res.data?.images_base64?.[0] || res.data?.image_base64;
      if (img) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now(),
            sender: "ai",
            text: "Generated image",
            imageBase64: img.startsWith("data:") ? img : `data:image/png;base64,${img}`,
          },
        ]);
      }
      setShowImageModal(false);
      setImagePrompt("");
      setImageNegative("");
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: "ai",
          text: err?.response?.data?.detail || "Image generation failed.",
        },
      ]);
    } finally {
      setIsGeneratingImage(false);
    }
  };

  const sendMessage = async (audioFile?: File) => {
    if ((!input.trim() && !audioFile) || !selectedCharId) return;
    if (isSending) return;

    const userMsg: Message = {
      id: Date.now(),
      sender: "user",
      text: input || (audioFile ? "(Audio uploaded)" : ""),
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!audioFile) setInput("");
    setIsSending(true);

    try {
      const res = await axios.post(`${API_URL}/chat/${selectedCharId}`, {
        message: userMsg.text,
      });
      const reply = res.data?.reply || "LLM did not respond.";
      const displayReply = cleanReplyText(reply);
      const aiMsg: Message = {
        id: Date.now() + 1,
        sender: "ai",
        text: displayReply || reply,
      };
      setMessages((prev) => [...prev, aiMsg]);

      if (autoTts && reply) {
        try {
          const tts = await axios.post(`${API_URL}/tts`, {
            message: reply,
            character_id: selectedCharId,
          });
          if (tts.data?.audio_base64) {
            const url = `data:audio/wav;base64,${tts.data.audio_base64}`;
            const audio = new Audio(url);
            audioRef.current = audio;
            audio.onloadedmetadata = () => {
              setMessages((prev) =>
                prev.map((m) => (m.id === aiMsg.id ? { ...m, duration: audio.duration } : m))
              );
            };
            if (autoTts) {
              audio.play().catch((err) => console.warn("Autoplay failed", err));
            }
            setMessages((prev) =>
              prev.map((m) => (m.id === aiMsg.id ? { ...m, audioUrl: url } : m))
            );
          }
        } catch (err) {
          console.error("TTS failed", err);
        }
      }
    } catch (error) {
      console.error("Chat failed:", error);
      setMessages((prev) => [
        ...prev,
        { id: Date.now(), sender: "ai", text: "Backend error. Check local model." },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="flex h-screen bg-candy-dark text-white font-sans overflow-hidden">
      <div className="w-80 bg-candy-card border-r border-gray-800 p-4 hidden md:flex flex-col">
        <h1 className="text-2xl font-bold text-pink-500 mb-4 flex items-center gap-2">
          <Volume2 /> MyCandyLocal
        </h1>

        <button
          onClick={() => openEditor()}
          className="mb-4 w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-pink-600/80 hover:bg-pink-500 transition text-sm font-medium shadow"
        >
          <PlusCircle size={18} />
          Create character
        </button>

        <div className="flex-1 overflow-y-auto space-y-2">
          <h2 className="text-gray-400 text-xs uppercase font-semibold mb-2">Voices</h2>
          {characters.map((char) => (
            <div
              key={char.id}
              className={`w-full p-3 rounded-xl transition-all cursor-pointer ${
                selectedCharId === char.id
                  ? "bg-pink-600/20 border border-pink-500/50"
                  : "hover:bg-gray-800 border border-transparent"
              }`}
              onClick={() => setSelectedCharId(char.id)}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden">
                    <User size={20} />
                  </div>
                  <div className="text-left">
                    <div className="font-medium">{char.name}</div>
                    <div className="text-xs text-gray-400">
                      {char.voice_model_path ? "RVC ready" : "No model yet"}
                    </div>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openEditor(char);
                  }}
                  className="text-xs text-gray-300 border border-gray-700 px-2 py-1 rounded-lg hover:border-pink-500"
                >
                  Edit
                </button>
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs">
                <span
                  className={`px-2 py-1 rounded-full border ${statusTone(
                    (char.voice_training_status as TrainingStatus) || ""
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
          <div className="text-xs text-gray-500">Backend: {API_URL}</div>
        </div>
      </div>

      <div className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-800 to-candy-dark">
        <div className="p-4 border-b border-gray-800/50 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <span className="font-bold">Chat</span>
            {selectedCharacter && (
              <span className="text-xs text-gray-400 bg-gray-800/70 px-2 py-1 rounded-full">
                Voice: {selectedCharacter.name}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoTts((v) => !v)}
              className="flex items-center gap-2 text-xs px-3 py-2 rounded-full border border-gray-700 hover:border-pink-500 transition bg-gray-800/60"
            >
              {autoTts ? <Volume2 size={16} /> : <VolumeX size={16} />}
              {autoTts ? "Voice on" : "Voice off"}
            </button>
            <button
              onClick={() => setShowImageModal(true)}
              className="flex items-center gap-2 text-xs px-3 py-2 rounded-full bg-pink-600/80 hover:bg-pink-500 transition"
            >
              <ImageIcon size={16} />
              Generate image
            </button>
            <Settings size={20} className="text-gray-400 hidden md:block" />
          </div>
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
                <p className="leading-relaxed whitespace-pre-line">{msg.text}</p>
                {msg.imageBase64 && (
                  <div className="mt-3 overflow-hidden rounded-xl border border-gray-700 bg-black/30">
                    <img src={msg.imageBase64} alt="Generated" className="w-full h-auto" />
                  </div>
                )}
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
          {isSending && (
            <div className="flex justify-start">
              <div className="bg-gray-800/50 p-4 rounded-2xl rounded-bl-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <div
                    className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
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
              placeholder="Type a line to start the scene..."
              className="flex-1 bg-transparent border-none outline-none text-white placeholder-gray-500 h-10"
              disabled={isSending}
            />
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="p-2 text-gray-400 hover:text-pink-500 transition"
                onClick={() => setShowImageModal(true)}
              >
                <ImageIcon size={20} />
              </button>
              <button
                type="button"
                className="p-2 text-gray-400 hover:text-pink-500 transition"
                onClick={() => fileInputRef.current?.click()}
              >
                <Mic size={20} />
              </button>
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isSending || !selectedCharId}
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

      {showEditor && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-3xl space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs uppercase text-gray-400 tracking-widest">Character setup</div>
                <h3 className="text-lg font-semibold">Persona + Voice training</h3>
              </div>
              <button onClick={() => setShowEditor(false)} className="text-gray-400 hover:text-white text-xl leading-none">
                &times;
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-400">Name</label>
                  <input
                    value={form.name}
                    onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                    className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                    placeholder="Velvet Storm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400">Persona summary (LLM prompt)</label>
                  <textarea
                    value={form.description}
                    onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
                    className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                    rows={3}
                    placeholder="Tone, flirting style, kinks to lean into, safety notes."
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400">Backstory & hooks</label>
                  <textarea
                    value={form.backstory}
                    onChange={(e) => setForm((f) => ({ ...f, backstory: e.target.value }))}
                    className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                    rows={3}
                    placeholder="Setting, story hooks, default scenes."
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400">Visual style prompt</label>
                  <textarea
                    value={form.visual_style}
                    onChange={(e) => setForm((f) => ({ ...f, visual_style: e.target.value }))}
                    className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                    rows={2}
                    placeholder="For image gens: lighting, outfit, vibe."
                  />
                </div>
              </div>

              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-400">Personality & tone</label>
                    <textarea
                      value={form.personality}
                      onChange={(e) => setForm((f) => ({ ...f, personality: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                      rows={2}
                      placeholder="Playful, bratty, dominant, etc."
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-400">Relationship type</label>
                    <input
                      value={form.relationship_type}
                      onChange={(e) => setForm((f) => ({ ...f, relationship_type: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                      placeholder="Companion, domme, sub, girlfriend..."
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-400">Do's / turn-ons</label>
                    <textarea
                      value={form.dos}
                      onChange={(e) => setForm((f) => ({ ...f, dos: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                      rows={2}
                      placeholder="Praise, rough talk, etc."
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-400">Hard limits</label>
                    <textarea
                      value={form.donts}
                      onChange={(e) => setForm((f) => ({ ...f, donts: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                      rows={2}
                      placeholder="No gore, no noncon, etc."
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs text-gray-400">Voice notes</label>
                  <input
                    value={form.voice_style}
                    onChange={(e) => setForm((f) => ({ ...f, voice_style: e.target.value }))}
                    className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                    placeholder="Breathy alto, teasing, soft edges..."
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-400">Voice sample (mp3/wav/zip)</label>
                    <div className="border border-dashed border-gray-700 rounded-lg p-3 bg-gray-800/50">
                      <input
                        type="file"
                        accept=".mp3,.wav,.zip,audio/*"
                        onChange={(e) => setVoiceFile(e.target.files?.[0] || null)}
                        className="w-full text-sm text-gray-300"
                      />
                      <div className="text-[11px] text-gray-500 mt-1">15-60s clean speech, no music.</div>
                      {voiceFile && <div className="text-xs text-gray-300 mt-1">{voiceFile.name}</div>}
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400">YouTube voice ref (optional)</label>
                    <input
                      value={form.voice_youtube_url}
                      onChange={(e) => setForm((f) => ({ ...f, voice_youtube_url: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                      placeholder="https://youtu.be/..."
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center border border-gray-800 rounded-xl p-4 bg-gray-800/40">
              <div>
                <div className="text-xs uppercase text-gray-400">Voice training</div>
                <div className="flex items-center gap-2 mt-2">
                  <span className={`px-3 py-1 rounded-full border ${statusTone(trainStatus)}`}>
                    {trainStatus || "not queued"}
                  </span>
                  {trainStatus === "done" && <CheckCircle size={16} className="text-green-400" />}
                  {trainStatus === "failed" && (
                    <span className="text-red-300 text-xs flex items-center gap-1">
                      <AlertTriangle size={14} /> Check training log
                    </span>
                  )}
                </div>
                <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-pink-500 to-pink-300 transition-all"
                    style={{ width: `${trainProgress}%` }}
                  />
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {trainStatus === "done"
                    ? "Trained model ready for TTS."
                    : "Queued jobs update automatically once a worker starts."}
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <button
                  onClick={startTraining}
                  disabled={isTraining}
                  className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-400 flex items-center gap-2"
                >
                  {isTraining && <Loader2 size={16} className="animate-spin" />}
                  Start training
                </button>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowEditor(false)}
                className="px-4 py-2 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={saveCharacter}
                disabled={isSavingChar}
                className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-400"
              >
                {isSavingChar ? "Saving..." : form.id ? "Update" : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}

      {showImageModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-md space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Generate image</h3>
              <button onClick={() => setShowImageModal(false)} className="text-gray-400 hover:text-white">
                &times;
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400">Prompt</label>
                <textarea
                  value={imagePrompt}
                  onChange={(e) => setImagePrompt(e.target.value)}
                  className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                  rows={3}
                  placeholder="What should be rendered?"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400">Negative prompt (optional)</label>
                <input
                  value={imageNegative}
                  onChange={(e) => setImageNegative(e.target.value)}
                  className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500"
                  placeholder="What to avoid?"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowImageModal(false)}
                className="px-4 py-2 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={generateImage}
                disabled={isGeneratingImage}
                className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-400"
              >
                {isGeneratingImage ? "Generating..." : "Generate"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
