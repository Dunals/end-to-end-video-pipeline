# Automated AI Content generator

An end-to-end autonomous pipeline that generates short-form video content from a single topic prompt. This system orchestrates multiple AI models to handle scripting, voice synthesis, visual generation, animation, and post-production editing without human intervention.

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.x-yellow.svg)

## 🏗️ Architecture Overview

The pipeline executes the following stages sequentially and in parallel:

1.  **Script Generation:** Custom wrapper for LLM APIs to generate viral scripts.
2.  **Audio Synthesis:** TTS integration for realistic voiceovers.
3.  **Visual Asset Generation:** Multi-threaded fetching of realistic scenes via internal API endpoints.
4.  **Transcription:** OpenAI Whisper for word-level timestamp generation.
5.  **Post-Production:** `MoviePy` based engine to stitch video, sync audio, and burn in dynamic subtitles.

## 🔓 Reverse Engineering Core

This project demonstrates advanced network engineering capabilities by interacting directly with undocumented internal APIs, rather than relying on standard public SDKs.

* **Traffic Analysis:** Utilized browser network inspection to deconstruct complex HTTP requests, headers, and payload structures.
* **API Simulation:** Replicates legitimate browser fingerprints (User-Agents, Referers) to interact with `NoteGPT` (LLM), `FlatAI` (Image Gen), and `TTS` services programmatically.
* **Token Management:** Implements dynamic generation of nonces, UUIDs, and session tokens to maintain authenticated states without manual intervention.

## 🚀 Key Technical Features

* **Concurrency:** Implements `concurrent.futures` to render multiple video scenes in parallel, significantly reducing generation time.
* **Resilient Networking:**
    * **Proxy Rotation:** Hybrid system using Paid Proxies for sensitive APIs and Free Proxies for high-bandwidth tasks.
    * **Retry Logic:** Automatic handling of timeouts and server rejections.
* **Multimedia Processing:** Automated video editing, resizing, cropping (9:16), and subtitle synchronization.

## 🛠 Dependencies & Setup

### Prerequisites
* Python 3.10+
* [ImageMagick](https://imagemagick.org/) (Required for Text overlays)
* [FFmpeg](https://ffmpeg.org/) (Required for video processing)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/automated-ai-content-engine.git](https://github.com/yourusername/automated-ai-content-engine.git)
    cd automated-ai-content-engine
    ```

2.  Install python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    * Update `IMAGEMAGICK_BINARY` path in `main.py` if not on Windows default.
    * Add Proxy credentials (optional but recommended for stability).

## 🏃 Usage

Run the script and enter a topic:

```bash
python main.py
