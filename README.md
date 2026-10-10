# Healthcare AI Clinical Documentation Assistant (Healthcare ASR)

A lightweight, reliable, offline-first desktop application designed to transcribe clinical consultations, estimate conversational speaker roles, extract structured medical entities (symptoms, vital signs, medications, allergies, diagnoses, and treatment plans) with explicit negation and uncertainty preservation, compile formatted SOAP notes, and persist structured consultation records to disk.

---

## ⚠️ Clinical Safety Disclaimer

> **IMPORTANT CLINICAL NOTICE**  
> This software is an experimental AI documentation assistant developed strictly for research, demonstration, and preliminary drafting.  
> - **It does NOT provide medical advice, diagnosis, or treatment.**  
> - **All generated notes are classified as `draft_pending_review`.**  
> - **A licensed healthcare professional must independently verify, edit, and validate all extracted clinical facts, vital signs, and treatment plans before entering them into official medical records.**  
> - **All development and testing use synthetic patient conversations only.**

---

## 1. Project Overview & Architecture

The application is structured into modular layers designed for simplicity, transparency, and local Windows CPU execution without paid APIs or cloud dependencies:

```
HEALTHCARE_ASR/
├── app.py                     # Desktop Graphical User Interface (Tkinter/ttk)
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Comprehensive documentation and beginner guide
├── PROJECT_STATUS.md          # Implementation verification and test report
├── .gitignore                 # Version control exclusions for privacy and caches
├── .gitattributes             # Line ending normalization rules
├── src/
│   ├── models/
│   │   └── patient_record.py  # Typed dataclasses for clinical records & JSON persistence
│   ├── audio/
│   │   ├── recorder.py        # Sounddevice 16 kHz mono microphone capture
│   │   └── asr.py             # Faster-Whisper CPU int8 transcription engine
│   └── processing/
│       ├── speaker_processor.py # Rule-based speaker role estimator (Doctor/Patient/Unknown)
│       ├── medical_processor.py # Deterministic clinical entity extractor with negation
│       └── soap_note.py         # Structured SOAP note formatter and missing-data detector
├── recordings/                # Local audio files (synthetic fixtures & temporary WAVs)
│   └── synthetic_consultation_test.wav
├── output/                    # Persisted consultation records in structured JSON
└── tests/                     # Automated pytest suite (49 unit and integration tests)
    ├── test_audio_recorder.py
    ├── test_asr.py
    ├── test_speaker_processor.py
    ├── test_medical_processor.py
    ├── test_soap_note.py
    ├── test_patient_record.py
    ├── test_json_export.py
    ├── test_gui.py
    └── test_end_to_end_integration.py
```

### Module Responsibilities:
1. **Audio Capture (`src/audio/recorder.py`):** Records mono audio at 16,000 Hz using `sounddevice` into a thread-safe queue. Validates audio length and non-emptiness before writing to a WAV file.
2. **Speech Recognition (`src/audio/asr.py`):** Uses `faster-whisper` (`tiny.en` model, CPU inference, `int8` quantization) to transcribe speech into text segments with timestamps without external API calls.
3. **Speaker Estimation (`src/processing/speaker_processor.py`):** Contextually tags transcript segments as Doctor, Patient, or Unknown using conversational patterns.
4. **Clinical Processing (`src/processing/medical_processor.py`):** Deterministic extraction using regex and clinical dictionaries. Detects explicit negation (*"denies chest pain"*), preserves uncertainty (*"possible asthma"*), and parses vital signs, medications, and allergies.
5. **SOAP Note Formatting (`src/processing/soap_note.py`):** Organizes extracted information into Subjective (S), Objective (O), Assessment (A), and Plan (P) sections with explicit callouts for unmeasured data.
6. **Data Persistence (`src/models/patient_record.py`):** Saves consultation records atomically using temporary files and timestamped unique filenames in `output/`.
7. **Desktop Interface (`app.py`):** Multi-threaded Tkinter GUI ensuring the UI never freezes during audio recording or model transcription.

---

## 2. System Requirements & Prerequisites

- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Supported Python Version:** Python 3.12 (Python 3.11 or 3.12 recommended; avoids Python 3.14 native C-extension incompatibility)
- **Disk Space:** ~500 MB (includes virtual environment and cached Whisper `tiny.en` model)
- **RAM:** Minimum 4 GB RAM (runs comfortably on CPU)
- **Audio:** Microphone (optional; synthetic WAV file import is built-in for testing)

---

## 3. Step-by-Step Installation for Beginners

All commands should be executed from PowerShell.

### Step 1: Navigate to the Project Directory
```powershell
Set-Location "C:\Users\Aziza\HEALTHCARE_ASR"
```

### Step 2: Verify the Virtual Environment
A local virtual environment named `.venv` is configured in the project root:
```powershell
& "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" --version
```
*Expected output: `Python 3.12.15`*

### Step 3: Install Required Dependencies
If you ever need to reinstall dependencies into the virtual environment:
```powershell
& "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

---

## 4. Running the Application

Launch the desktop user interface by running:
```powershell
& "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" app.py
```

### How to Use the Interface:
1. **Live Microphone Recording:**
   - Click **🎙️ Start Recording** to begin capturing consultation audio.
   - Speak clearly into your microphone.
   - Click **⏹️ Stop & Transcribe** when the encounter ends. The application will validate the recording, load the Whisper model in the background, and transcribe the audio.
2. **Instant Synthetic Demo (No Microphone Needed):**
   - Click **✨ Run Synthetic Demo** to instantly transcribe and analyze a pre-recorded synthetic consultation audio fixture.
3. **Importing WAV Files:**
   - Click **📁 Import WAV File** to select any existing standard 16-bit PCM WAV audio file.
4. **Inspecting the Results:**
   - **Tab 1 (Transcript & Speakers):** View recognized text broken down by utterance timestamps and estimated Doctor/Patient roles.
   - **Tab 2 (Extracted Clinical Data):** View detected symptoms, vital signs, medications, allergies, diagnoses, and unclassified phrases.
   - **Tab 3 (SOAP Clinical Note):** Review the draft SOAP note with clinical warnings and missing information callouts.
   - **Tab 4 (Structured JSON Record):** Inspect the raw schema v1.0 JSON representation.
5. **Exporting Data:**
   - Click **💾 Export JSON Record** to save the structured record to the `output/` directory.
6. **Starting a New Patient Encounter:**
   - Click **🔄 New / Clear** to reset the interface.

---

## 5. Running the Automated Test Suite

To run all 49 automated unit, integration, and GUI tests:
```powershell
& "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" -m pytest -v
```

To run a specific test file:
```powershell
& "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" -m pytest -v tests/test_medical_processor.py
```

---

## 6. Example Synthetic Consultation & Output

### Input Conversation (Synthetic):
> **Doctor:** "Good morning, Mrs. Davis. What brings you in today?"  
> **Patient:** "Hello doctor, I have had a bad sore throat and fever for two days."  
> **Doctor:** "I see. Are you experiencing any chest pain or difficulty breathing?"  
> **Patient:** "No chest pain, but swallowing hurts."

### Generated SOAP Note (Draft):
```
CLINICAL SAFETY NOTICE: This documentation is AI-generated draft text for licensed clinician review only.

S — SUBJECTIVE:
Reported Symptoms: sore throat (2 days), fever (2 days).
Pertinent Negatives: Denies chest pain.
Allergies: Not explicitly documented.

O — OBJECTIVE:
Vital Signs:
- Blood Pressure: Not recorded
- Heart Rate: Not recorded
- Body Temperature: Not recorded
- Oxygen Saturation (SpO2): Not recorded
- Blood Glucose: Not recorded

A — ASSESSMENT:
Assessment not documented in consultation transcript. Requires clinician clinical evaluation.

P — PLAN:
Plan not documented.

MISSING OR UNCERTAIN INFORMATION REQUIRING ATTENTION:
• Blood pressure was not explicitly documented in transcript
• Heart rate was not explicitly documented in transcript
• Body temperature was not explicitly documented in transcript
• Oxygen saturation (SpO2) was not explicitly documented in transcript
```

---

## 7. Troubleshooting Common Windows Issues

### 1. PowerShell Execution Policy Restriction
If PowerShell reports that running scripts is disabled:
- **Do not lower system security globally.** Instead, invoke the python binary directly using the ampersand operator:
  ```powershell
  & "C:\Users\Aziza\HEALTHCARE_ASR\.venv\Scripts\python.exe" app.py
  ```

### 2. Microphone Access Denied in Windows 11
If `AudioDeviceError` appears:
- Open **Windows Settings** → **Privacy & Security** → **Microphone**.
- Ensure **Microphone access** is turned **On**.
- Ensure **Let desktop apps access your microphone** is turned **On**.

### 3. ASR Model Download or Disk Space
- The Faster-Whisper `tiny.en` model is ~75 MB and downloads automatically on first run to the user cache (`~/.cache/huggingface/hub`).
- Inference runs entirely on CPU using `int8` quantization.

---

## 8. Backup and Recovery Instructions

To create a safe manual backup of your project code without version control:
```powershell
Compress-Archive -Path "C:\Users\Aziza\HEALTHCARE_ASR\src", "C:\Users\Aziza\HEALTHCARE_ASR\tests", "C:\Users\Aziza\HEALTHCARE_ASR\app.py", "C:\Users\Aziza\HEALTHCARE_ASR\requirements.txt", "C:\Users\Aziza\HEALTHCARE_ASR\README.md" -DestinationPath "C:\Users\Aziza\HEALTHCARE_ASR_backup.zip" -Force
```
Store this zip file on an external drive or trusted secure storage.
