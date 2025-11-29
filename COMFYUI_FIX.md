# ComfyUI Path-Konflikt beheben

## Problem
ComfyUI findet seine Python-Installation nicht richtig, weil Pfade durcheinander sind.

## Lösung

### Option 1: Neues Startskript nutzen (Empfohlen)
Ich habe ein neues `start_comfyui.bat` Skript erstellt, das mehrere Startmethoden probiert.

**Verwendung:**
1. Doppelklick auf `start_comfyui.bat` im Hauptverzeichnis
2. Das Skript probiert automatisch:
   - RUN_Launcher.bat (falls vorhanden)
   - launcher.py mit standalone Python
   - main.py mit standalone Python
   - main.py mit System-Python

### Option 2: ComfyUI manuell starten
1. Öffne PowerShell im ComfyUI-Ordner:
   ```powershell
   cd C:\Users\Ghost\Desktop\MyCandyLocal\ComfyUI
   ```

2. Starte mit einer dieser Methoden:
   
   **A) Mit RUN_Launcher.bat:**
   ```powershell
   .\RUN_Launcher.bat
   ```
   
   **B) Mit standalone Python:**
   ```powershell
   .\python_standalone\python.exe main.py --port 8002
   ```
   
   **C) Mit System-Python:**
   ```powershell
   python main.py --port 8002
   ```

### Option 3: start_all.ps1 verwenden (jetzt verbessert)
Die `start_all.ps1` wurde aktualisiert und probiert jetzt automatisch verschiedene Startmethoden.

Einfach neu starten:
```powershell
.\start_all.ps1
```

## Häufige Probleme

### "python_standalone\python.exe nicht gefunden"
- ComfyUI's standalone Python fehlt
- **Lösung**: ComfyUI neu installieren oder System-Python verwenden

### "main.py nicht gefunden"
- ComfyUI Installation unvollständig
- **Lösung**: ComfyUI komplett neu klonen/installieren

### Port-Konflikt
- Ein anderer Dienst nutzt Port 8002
- **Lösung**: Port in settings.json ändern:
  ```json
  "comfy_host": "http://127.0.0.1:8003"
  ```
  Und dann ComfyUI mit `--port 8003` starten

## Testen
Nach dem Start sollte ComfyUI erreichbar sein unter:
```
http://127.0.0.1:8002
```

Öffne die URL im Browser - wenn die ComfyUI-Oberfläche lädt, funktioniert es!
