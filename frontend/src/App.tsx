import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  Send,
  Mic,
  Play,
  Square,
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
  return raw
    .replace(audioMarkersRegex, "")
    .replace(customTokenRegex, "")
    .replace(/\s{2,}/g, " ")
    .trim();
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
  avatar_path?: string;
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

const baseConversation: Message[] = [
  { id: 1, sender: "ai", text: "Hey! I am ready. What should we explore?" },
];

const avatarUrl = (char: Character | null) => {
  if (!char?.avatar_path) return "";
  const base = API_URL.replace(/\/$/, "");
  return `${base}/avatars/${char.avatar_path}`;
};

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>(baseConversation);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharId, setSelectedCharId] = useState<number | null>(null);
  const [autoTts, setAutoTts] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [recordSeconds, setRecordSeconds] = useState(0);

  const [showEditor, setShowEditor] = useState(false);
  const [form, setForm] = useState<CharacterFormState>(emptyForm);
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [isSavingChar, setIsSavingChar] = useState(false);

  const [showImageModal, setShowImageModal] = useState(false);
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageNegative, setImageNegative] = useState("");
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);

  const [trainStatus, setTrainStatus] = useState<TrainingStatus>("");
  const [trainProgress, setTrainProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const trainTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  const recordStartRef = useRef<number | null>(null);
  const recordTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [micLevel, setMicLevel] = useState(0);
  const [micStatus, setMicStatus] = useState("");
  const micLevelTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const selectedCharacter = useMemo(
    () => characters.find((c) => c.id === selectedCharId) || null,
    [characters, selectedCharId]
  );

  // Reset conversation when switching characters so each has a fresh chat
  useEffect(() => {
    if (selectedCharId !== null) {
      setMessages(baseConversation);
      setInput("");
    }
  }, [selectedCharId]);

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
      if (recordTimerRef.current) clearInterval(recordTimerRef.current);
      if (micLevelTimerRef.current) clearInterval(micLevelTimerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
      }
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
    setAvatarFile(null);
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
      if (id && avatarFile) {
        const fd = new FormData();
        fd.append("file", avatarFile);
        await axios.post(`${API_URL}/characters/${id}/avatar`, fd);
      }
      await fetchCharacters();
      if (id) {
        setSelectedCharId(id);
        setMessages(baseConversation);
      }
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
    if (isSending || isAiSpeaking) return;

    const userId = Date.now();
    const outgoingText = input.trim() || (audioFile ? "Voice message..." : "");
    setIsSending(true);
    setInput("");

    // Show the user message immediately; update text later if transcript arrives
    setMessages((prev) => [
      ...prev,
      {
        id: userId,
        sender: "user",
        text: outgoingText || "(audio not recognized)",
      },
    ]);

    try {
      let res;
      if (audioFile) {
        const fd = new FormData();
        // send empty text so backend transcribes the audio; UI shows placeholder
        fd.append("message", "");
        fd.append("audio", audioFile);
        res = await axios.post(`${API_URL}/chat/${selectedCharId}`, fd);
      } else {
        res = await axios.post(`${API_URL}/chat/${selectedCharId}`, {
          message: outgoingText,
        });
      }
      const usedText =
        (res.data?.transcription as string | undefined)?.trim() ??
        (res.data?.user_text as string | undefined)?.trim() ??
        outgoingText;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === userId ? { ...m, text: usedText || "(audio not recognized)" } : m
        )
      );

      const reply = res.data?.reply || "LLM did not respond.";
      const ttsText = res.data?.reply_tts || reply;
      const displayReply = cleanReplyText(reply);
      const aiMsg: Message = {
        id: Date.now() + 1,
        sender: "ai",
        text: displayReply || reply,
      };
      setMessages((prev) => [...prev, aiMsg]);

      if (autoTts && reply) {
        // Prefer inline audio from chat response; fallback to separate /tts call
        const inlineB64 = res.data?.audio_base64 as string | undefined;
        const playFromBase64 = (b64: string) => {
          const url = `data:audio/wav;base64,${b64}`;
          const audio = new Audio(url);
          audioRef.current = audio;
          setPlayingId(aiMsg.id);
          setIsAiSpeaking(true);
          audio.onloadedmetadata = () => {
            setMessages((prev) =>
              prev.map((m) => (m.id === aiMsg.id ? { ...m, duration: audio.duration } : m))
            );
          };
          audio.onended = () => {
            setPlayingId(null);
            setIsAiSpeaking(false);
          };
          audio.onpause = () => {
            if (audio.ended) {
              setPlayingId(null);
              setIsAiSpeaking(false);
            }
          };
          if (autoTts) {
            audio.play().catch((err) => console.warn("Autoplay failed", err));
          }
          setMessages((prev) => prev.map((m) => (m.id === aiMsg.id ? { ...m, audioUrl: url } : m)));
        };

        if (inlineB64) {
          playFromBase64(inlineB64);
        } else {
          try {
            const tts = await axios.post(`${API_URL}/tts`, {
              message: ttsText,
              character_id: selectedCharId,
            });
            if (tts.data?.audio_base64) {
              playFromBase64(tts.data.audio_base64);
            }
          } catch (err) {
            console.error("TTS failed", err);
          }
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

  const stopRecording = () => {
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    if (micLevelTimerRef.current) {
      clearInterval(micLevelTimerRef.current);
      micLevelTimerRef.current = null;
    }
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    }
    mediaRecorderRef.current = null;
    recordStartRef.current = null;
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
      analyserRef.current = null;
    }
    setIsRecording(false);
    setMicStatus("");
  };

  const startRecording = async () => {
    if (isRecording) return;
    setRecordingError("");
    setMicStatus("Opening mic…");
    try {
      // Always pick system default mic; if it's virtual/blacklisted, fall back to first non-blacklisted mic
      const blacklist = ["voidol", "vb-audio", "virtual", "cable"];
      const baseConstraints: MediaStreamConstraints = {
        audio: {
          deviceId: "default",
          channelCount: 1,
          sampleRate: 16000,
          noiseSuppression: true,
          echoCancellation: true,
          autoGainControl: true,
        },
      };
      let stream = await navigator.mediaDevices.getUserMedia(baseConstraints);
      let activeTrack = stream.getAudioTracks()[0];
      let label = (activeTrack?.label || "").toLowerCase();
      if (activeTrack?.label) {
        setMicStatus(`Using: ${activeTrack.label}`);
      } else {
        setMicStatus("Using: system default");
      }
      const labelIsBlacklisted = label && blacklist.some((b) => label.includes(b));
      if (labelIsBlacklisted) {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const inputs = devices.filter((d) => d.kind === "audioinput");
          const best =
            inputs.find((d) => !blacklist.some((b) => (d.label || "").toLowerCase().includes(b))) ||
            inputs[0];
          if (best && best.deviceId) {
            stream.getTracks().forEach((t) => t.stop());
            stream = await navigator.mediaDevices.getUserMedia({
              audio: {
                deviceId: { exact: best.deviceId },
                channelCount: 1,
                sampleRate: 16000,
                noiseSuppression: true,
                echoCancellation: true,
                autoGainControl: true,
              },
            });
            activeTrack = stream.getAudioTracks()[0];
            setMicStatus(`Using: ${activeTrack?.label || "fallback mic"}`);
          }
        } catch (err) {
          console.warn("Fallback mic selection failed; using default", err);
        }
      }
      const recorder = new MediaRecorder(stream);
      recordChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (micLevelTimerRef.current) {
          clearInterval(micLevelTimerRef.current);
          micLevelTimerRef.current = null;
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close().catch(() => {});
          audioCtxRef.current = null;
          analyserRef.current = null;
        }
        setMicStatus("");
        const blob = new Blob(recordChunksRef.current, { type: "audio/webm" });
        recordChunksRef.current = [];
        if (blob.size < 500) {
          setRecordingError("Recording too short.");
          return;
        }
        const file = new File([blob], `mic_${Date.now()}.webm`, { type: blob.type });
        sendMessage(file);
      };
      recorder.start();
      // VU meter setup
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      micLevelTimerRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const v = dataArray[i] - 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / dataArray.length) / 128; // 0..1
        setMicLevel(Math.min(1, rms * 2));
      }, 100);

      mediaRecorderRef.current = recorder;
      recordStartRef.current = Date.now();
      setRecordSeconds(0);
      setIsRecording(true);
      recordTimerRef.current = setInterval(() => {
        if (recordStartRef.current) {
          const diff = Date.now() - recordStartRef.current;
          setRecordSeconds(Math.max(0, diff / 1000));
        }
      }, 200);
    } catch (err: any) {
      setRecordingError("Mic Zugriff fehlgeschlagen.");
      setMicStatus("Mic failed");
      setIsRecording(false);
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
                      {char.avatar_path ? (
                        <img src={avatarUrl(char)} alt={char.name} className="w-full h-full object-cover" />
                      ) : (
                        <User size={20} />
                      )}
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
                          audioRef.current.currentTime = 0;
                        }
                        const audio = new Audio(msg.audioUrl!);
                        audioRef.current = audio;
                        setPlayingId(msg.id);
                        setIsAiSpeaking(true);
                        audio.onended = () => {
                          setPlayingId(null);
                          setIsAiSpeaking(false);
                        };
                        audio.onpause = () => {
                          if (audio.ended) {
                            setPlayingId(null);
                            setIsAiSpeaking(false);
                          }
                        };
                        audio.play().catch((err) => console.warn("play failed", err));
                      }}
                      className="p-2 bg-pink-500 rounded-full hover:scale-105 transition"
                    >
                      <Play size={12} fill="currentColor" />
                    </button>
                    <button
                      onClick={() => {
                        if (audioRef.current) {
                          audioRef.current.pause();
                          audioRef.current.currentTime = 0;
                        }
                        setPlayingId(null);
                      }}
                      className="p-2 bg-gray-600 rounded-full hover:scale-105 transition disabled:opacity-50"
                      disabled={playingId !== msg.id}
                      title="Stop"
                    >
                      <Square size={12} />
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
          <div className="max-w-4xl mx-auto space-y-2">
            <div className="flex flex-col md:flex-row md:items-center md:gap-3 text-xs text-gray-500 px-1">
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Mic:</span>
                <span className="text-gray-200">System default</span>
              </div>
              <div className="flex items-center gap-2 mt-2 md:mt-0">
                <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-pink-400 transition-all"
                    style={{ width: `${Math.min(100, Math.round(micLevel * 100))}%` }}
                  />
                </div>
                <span className="text-gray-400">
                  {micStatus || (micLevel > 0.05 ? "Signal" : "No signal")}
                </span>
                {recordingError && <span className="text-red-300">• {recordingError}</span>}
              </div>
            </div>

            <div className="relative flex items-center bg-gray-800/90 backdrop-blur border border-gray-700 rounded-full p-2 pr-2 pl-6 shadow-2xl">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Type a line to start the scene..."
                className="flex-1 bg-transparent border-none outline-none text-white placeholder-gray-500 h-10"
                disabled={isSending || isAiSpeaking}
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
                  className={`p-2 rounded-full transition ${
                    isRecording
                      ? "bg-red-600 text-white shadow-lg shadow-red-600/40"
                      : "text-gray-400 hover:text-pink-500 hover:bg-gray-700/60"
                  }`}
                  disabled={isSending || isAiSpeaking}
                  onMouseDown={() => !isSending && !isAiSpeaking && startRecording()}
                  onMouseUp={stopRecording}
                  onMouseLeave={() => isRecording && stopRecording()}
                  onTouchStart={(e) => {
                    e.preventDefault();
                    if (!isSending && !isAiSpeaking) startRecording();
                  }}
                  onTouchEnd={(e) => {
                    e.preventDefault();
                    stopRecording();
                  }}
                  onTouchCancel={(e) => {
                    e.preventDefault();
                    stopRecording();
                  }}
                  title="Hold to record voice"
                >
                  <Mic size={20} />
                </button>
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || isSending || isAiSpeaking || !selectedCharId}
                  className="p-3 bg-pink-600 hover:bg-pink-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-full transition-all shadow-lg hover:shadow-pink-500/25"
                >
                  <Send size={18} />
                </button>
              </div>
            </div>

            <div className="flex items-center gap-3 text-xs text-gray-500 px-1">
              {isRecording ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <span className="text-red-200">
                    Recording... {recordSeconds.toFixed(1)}s (release to send)
                  </span>
                </>
              ) : (
                <span className="text-gray-500">Hold mic to speak; release to send.</span>
              )}
            </div>
            <div className="text-center mt-2 text-xs text-gray-600">Powered by RVC Local & FastAPI</div>
          </div>
        </div>

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
                  <div>
                    <label className="text-xs text-gray-400">Language (LLM & STT)</label>
                    <select
                      value={form.language}
                      onChange={(e) => setForm((f) => ({ ...f, language: e.target.value }))}
                      className="w-full mt-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 outline-none focus:border-pink-500 text-sm"
                    >
                      <option value="en">English</option>
                      <option value="de">Deutsch</option>
                      <option value="fr">Français</option>
                      <option value="es">Español</option>
                      <option value="it">Italiano</option>
                    </select>
                    <div className="text-[11px] text-gray-500 mt-1">
                      Steuert Antwortsprache des LLM und bevorzugte STT-Sprache.
                    </div>
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
                    <label className="text-xs text-gray-400">Avatar (100x100)</label>
                    <div className="border border-dashed border-gray-700 rounded-lg p-3 bg-gray-800/50">
                      <input
                        type="file"
                        accept=".png,.jpg,.jpeg,image/*"
                        onChange={(e) => setAvatarFile(e.target.files?.[0] || null)}
                        className="w-full text-sm text-gray-300"
                      />
                      <div className="text-[11px] text-gray-500 mt-1">Square image, will be resized to 100x100.</div>
                      {avatarFile && <div className="text-xs text-gray-300 mt-1">{avatarFile.name}</div>}
                    </div>
                  </div>
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




