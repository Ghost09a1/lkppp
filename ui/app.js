// ui/app.js

// Basic helper: currently no prefix adjustment needed, kept for possible future changes
const api = (path) => `${path.startsWith("http") ? path : path}`;

let characters = [];
let currentId = null;
let autoTtsEnabled = true;
const audioTokenRegex = /<custom_token_\d+>/;
const audioMarkersRegex = /<\|audio_start\|>|<\|audio_end\|>/g;
const customTokenRegex = /<custom_token_\d+>/g;
const emoteRegex = /\*[^*]+\*/g;
let audioCache = [];
let audioCounter = 0;

const elements = {
    list: document.getElementById("character-list"),
    name: document.getElementById("character-name"),
    desc: document.getElementById("character-desc"),
    chat: document.getElementById("chat-window"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("chat-input"),
    modal: document.getElementById("modal"),
    characterForm: document.getElementById("character-form"),
    newCharacterBtn: document.getElementById("new-character-btn"),
    audioBtn: document.getElementById("audio-btn"),
    cancelModal: document.getElementById("cancel-modal"),
    imageBtn: document.getElementById("image-btn"),
    voiceSampleInput: null,
    voiceDatasetInput: document.getElementById("voice-dataset"),
    trainVoiceBtn: document.getElementById("train-voice-btn"),
    trainStatus: document.getElementById("train-status"),
    trainOk: document.getElementById("train-ok"),
};

// --- Generic Fetch-Helper ----------------------------------------------------

async function fetchJSON(url, options = {}) {
    const res = await fetch(api(url), {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

// --- Character-Handling ------------------------------------------------------

function renderCharacters() {
    elements.list.innerHTML = "";
    characters.forEach((c) => {
        const card = document.createElement("div");
        card.className = `character-card ${c.id === currentId ? "active" : ""}`;
        const header = document.createElement("header");
        const title = document.createElement("h3");
        title.textContent = c.name;
        const editBtn = document.createElement("button");
        editBtn.type = "button";
        editBtn.className = "edit-btn";
        editBtn.textContent = "Edit";
        editBtn.onclick = (e) => {
            e.stopPropagation();
            openCharacterModal(c);
        };
        header.appendChild(title);
        header.appendChild(editBtn);

        const desc = document.createElement("p");
        desc.textContent = c.description || "No description";

        card.appendChild(header);
        card.appendChild(desc);
        card.onclick = () => selectCharacter(c.id);
        elements.list.appendChild(card);
    });
}

async function loadCharacters() {
    try {
        const data = await fetchJSON("/characters");
        characters = data.characters || [];
        renderCharacters();
        if (characters.length && !currentId) {
            await selectCharacter(characters[0].id);
        }
    } catch (e) {
        console.error("Failed to load characters:", e);
    }
}

async function selectCharacter(id) {
    currentId = id;
    const c = characters.find((x) => x.id === id);
    elements.name.textContent = c?.name || "Character";
    elements.desc.textContent = c?.description || "";
    renderCharacters();
    elements.chat.innerHTML = "";
}

// --- Chat-UI Helpers ---------------------------------------------------------

function pushBubble(role, text) {
    const div = document.createElement("div");
    div.className = `bubble ${role}`;
    div.textContent = text;
    elements.chat.appendChild(div);
    elements.chat.scrollTop = elements.chat.scrollHeight;
    return div;
}

function displayTextFromRaw(raw) {
    if (!raw) return "";
    return raw.replace(audioMarkersRegex, "").replace(customTokenRegex, "").trim();
}

function setAssistantContent(bubble, displayText, rawText) {
    bubble.innerHTML = "";
    const span = document.createElement("span");
    span.textContent = displayText;
    bubble.appendChild(span);

    if (rawText) {
        const btn = document.createElement("button");
        btn.className = "play-btn";
        btn.type = "button";
        btn.textContent = "▶";
        btn.title = "Play reply audio";
        btn.dataset.raw = rawText;
        btn.onclick = async () => {
            const audioId = btn.dataset.audioId;
            const cached = getCachedAudio(audioId);
            if (cached) {
                const audio = new Audio(cached);
                audio.play();
                return;
            }
            await playTtsAndWait(rawText, btn);
        };
        bubble.appendChild(btn);
    }
}

function cacheAudio(audioBase64) {
    const id = `aud-${Date.now()}-${audioCounter++}`;
    audioCache.push({ id, url: `data:audio/wav;base64,${audioBase64}` });
    if (audioCache.length > 10) {
        audioCache.shift();
    }
    return id;
}

function getCachedAudio(id) {
    if (!id) return null;
    const found = audioCache.find((x) => x.id === id);
    return found ? found.url : null;
}

async function playTtsAndWait(text, btnOverride) {
    if (!text) return;
    const btn = btnOverride || elements.audioBtn;
    const previous = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Audio...";
    try {
        const res = await fetchJSON("/tts", {
            method: "POST",
            body: JSON.stringify({ message: text, character_id: currentId }),
        });
        if (!res.audio_base64) {
            throw new Error(res.error || "TTS failed");
        }
        const audioId = cacheAudio(res.audio_base64);
        const audio = new Audio(`data:audio/wav;base64,${res.audio_base64}`);
        await audio.play();
        btn.dataset.audioId = audioId;
        return audioId;
    } catch (err) {
        console.error("TTS error:", err);
    } finally {
        btn.disabled = false;
        btn.textContent = previous;
    }
}

function updateAudioButton() {
    elements.audioBtn.classList.toggle("active", autoTtsEnabled);
    elements.audioBtn.classList.toggle("off", !autoTtsEnabled);
    elements.audioBtn.textContent = autoTtsEnabled ? "Audio On" : "Audio Off";
}

// --- Chat: simpler /chat endpoint, no streaming -----------------------------

async function sendMessage(message) {
    if (!currentId) {
        alert("Create or select a character first.");
        return;
    }

    // User bubble
    pushBubble("user", message);
    elements.input.value = "";

    // Placeholder for assistant
    const assistantBubble = pushBubble("assistant", "Is answering...");

    try {
        const res = await fetchJSON(`/chat/${currentId}`, {
            method: "POST",
            body: JSON.stringify({ message }),
        });

        const replyText = res.reply || "No response from LLM.";
        const displayText = displayTextFromRaw(replyText) || replyText;
        let audioId = null;
        if (autoTtsEnabled) {
            try {
                audioId = await playTtsAndWait(replyText);
            } catch (err) {
                console.error("TTS error:", err);
            }
        }
        setAssistantContent(assistantBubble, displayText, replyText);
        if (audioId) {
            const btn = assistantBubble.querySelector(".play-btn");
            if (btn) btn.dataset.audioId = audioId;
        }
    } catch (e) {
        console.error("Error calling /chat:", e);
        assistantBubble.textContent = "Error contacting backend.";
    }
}

// --- Event-Listener ----------------------------------------------------------

elements.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = elements.input.value.trim();
    if (text) sendMessage(text);
});

elements.newCharacterBtn.onclick = () => {
    elements.characterForm.reset();
    delete elements.characterForm.dataset.editId;
    elements.modal.classList.remove("hidden");
};
function openCharacterModal(character) {
    if (!character) return;
    elements.modal.classList.remove("hidden");
    const form = elements.characterForm;
    form.elements["name"].value = character.name || "";
    form.elements["description"].value = character.description || "";
    form.elements["visual_style"].value = character.visual_style || "";
    form.elements["appearance_notes"].value = character.appearance_notes || "";
    form.elements["personality"].value = character.personality || "";
    form.elements["backstory"].value = character.backstory || "";
    form.elements["relationship_type"].value = character.relationship_type || "";
    form.elements["dos"].value = character.dos || "";
    form.elements["donts"].value = character.donts || "";
    form.elements["voice_style"].value = character.voice_style || "";
    form.elements["voice_youtube_url"].value = character.voice_youtube_url || "";
    if (form.elements["language"]) {
        form.elements["language"].value = character.language || "en";
    }
    if (form.elements["voice_model_path"]) {
        form.elements["voice_model_path"].value = character.voice_model_path || "";
    }
    if (elements.trainStatus) {
        elements.trainStatus.textContent = character.voice_training_status || "";
        elements.trainOk.style.display = character.voice_training_status === "done" ? "inline" : "none";
    }
    form.dataset.editId = character.id;
}
function openCharacterModal(character) {
    if (!character) return;
    elements.modal.classList.remove("hidden");
    const form = elements.characterForm;
    form.elements["name"].value = character.name || "";
    form.elements["description"].value = character.description || "";
    form.elements["visual_style"].value = character.visual_style || "";
    form.elements["appearance_notes"].value = character.appearance_notes || "";
    form.elements["personality"].value = character.personality || "";
    form.elements["backstory"].value = character.backstory || "";
    form.elements["relationship_type"].value = character.relationship_type || "";
    form.elements["dos"].value = character.dos || "";
    form.elements["donts"].value = character.donts || "";
    form.dataset.editId = character.id;
}

elements.audioBtn.onclick = () => {
    autoTtsEnabled = !autoTtsEnabled;
    updateAudioButton();
};

elements.cancelModal.onclick = () => {
    elements.modal.classList.add("hidden");
};

async function uploadVoiceDataset(charId) {
    const file = elements.voiceDatasetInput?.files?.[0];
    if (!file || !charId) return;
    const fd = new FormData();
    fd.append("file", file);
    await fetch(api(`/characters/${charId}/voice_dataset`), {
        method: "POST",
        body: fd,
    });
}

function setTrainStatus(text, ok) {
    if (elements.trainStatus) elements.trainStatus.textContent = text || "";
    if (elements.trainOk) elements.trainOk.style.display = ok ? "inline" : "none";
}

async function triggerTrainVoice(charId) {
    if (!charId) return;
    try {
        setTrainStatus("queued", false);
        const res = await fetchJSON(`/characters/${charId}/train_voice`, { method: "POST" });
        if (!res.ok && res.status !== "queued") {
            setTrainStatus("failed", false);
            return;
        }
        // best-effort refresh status after a short delay
        setTimeout(loadCharacters, 2000);
    } catch (err) {
        console.error("Train voice failed:", err);
        setTrainStatus("failed", false);
    }
}

elements.trainVoiceBtn?.addEventListener("click", async () => {
    const editId = elements.characterForm.dataset.editId || currentId;
    if (!editId) {
        alert("Save the character first, then train.");
        return;
    }
    try {
        await uploadVoiceDataset(editId);
    } catch (err) {
        console.error("Dataset upload failed:", err);
    }
    await triggerTrainVoice(editId);
});

elements.characterForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(elements.characterForm);
    const payload = Object.fromEntries(formData.entries());
    const editId = elements.characterForm.dataset.editId;
    delete elements.characterForm.dataset.editId;

    try {
        let charId = editId;
        if (editId) {
            await fetchJSON(`/characters/${editId}/update`, {
                method: "POST",
                body: JSON.stringify(payload),
            });
        } else {
            const res = await fetchJSON("/characters", {
                method: "POST",
                body: JSON.stringify(payload),
            });
            charId = res.id;
        }
        // Optional: upload dataset for training
        try {
            await uploadVoiceDataset(charId);
        } catch (err) {
            console.error("Dataset upload failed:", err);
        }
        const file = elements.characterForm.elements["voice_sample"]?.files?.[0];
        if (file && charId) {
            const upload = new FormData();
            upload.append("file", file);
            await fetch(api(`/characters/${charId}/voice_sample`), {
                method: "POST",
                body: upload,
            });
        }
        const yt = payload.voice_youtube_url;
        if (yt && charId) {
            try {
                await fetchJSON(`/characters/${charId}/voice_sample_url`, {
                    method: "POST",
                    body: JSON.stringify({ url: yt }),
                });
            } catch (err) {
                console.error("YouTube download failed:", err);
            }
        }
        elements.modal.classList.add("hidden");
        elements.characterForm.reset();
        await loadCharacters();
    } catch (err) {
        console.error("Error creating character:", err);
        alert("Failed to create character.");
    }
});

// --- Image-Button -> /generate_image ----------------------------------------

elements.imageBtn.onclick = async () => {
    if (!currentId) {
        alert("Select a character first.");
        return;
    }

    const promptText = prompt(
        "Enter image prompt",
        `Portrait of ${elements.name.textContent} in vivid detail, NSFW`
    );
    if (!promptText) return;

    try {
        const res = await fetchJSON("/generate_image", {
            method: "POST",
            body: JSON.stringify({
                prompt: promptText,
                steps: 20,
                width: 640,
                height: 832,
            }),
        });

        if (!res.ok) {
            alert(res.error || "Image generation failed");
            return;
        }

        const img = document.createElement("img");
        img.src = `data:image/png;base64,${res.images_base64[0]}`;
        img.style.maxWidth = "220px";

        pushBubble("assistant", "Generated image:");
        elements.chat.appendChild(img);
        elements.chat.scrollTop = elements.chat.scrollHeight;
    } catch (err) {
        console.error("Error generating image:", err);
        alert("Image generation failed.");
    }
};

// --- Init --------------------------------------------------------------------

loadCharacters();
updateAudioButton();
