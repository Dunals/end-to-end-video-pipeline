import requests
import json
import uuid
import random
import string
import time
import re
import os
import whisper
import numpy as np
import concurrent.futures
import shutil
from PIL import Image
from moviepy.editor import *
from moviepy.config import change_settings

# --- SETTINGS ---
IMAGEMAGICK_BINARY = r"c:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
OUTPUT_FOLDER = "Final_Video_Output"

# --- PAID PROXY SETTINGS (LightningProxies for NoteGPT, FlatAI, TTS) ---
PROXY_HOST = ""
PROXY_PORT = ""
PROXY_USER = ""
PROXY_PASS = ""

# --- FREE PROXY SETTINGS (For Image-to-Video Only) ---
SPACE_URL = "https://zerogpu-aoti-wan2-2-fp8da-aoti-faster.hf.space"
FREE_PROXY_SOURCES = [
    "https://advanced.name/freeproxy/6946567254114"
]
THREAD_COUNT = 300 
PROXY_CHECK_TIMEOUT = 5

if os.path.exists(IMAGEMAGICK_BINARY):
    change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})
else:
    print(f"⚠️ ImageMagick නෑ. Path එක බලන්න.")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ==============================================================================
# 1. PROXY MANAGERS
# ==============================================================================

# 🔥 Global Pool to store working proxies immediately
VERIFIED_POOL = []

# --- A. Paid Proxy ---
def get_paid_proxy():
    session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    proxy_url = f"http://{PROXY_USER}-session-{session_id}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
    return {"http": proxy_url, "https": proxy_url}

# --- B. Free Proxy Logic ---
def get_all_free_proxies():
    print("🌍 Download Free Proxy List...")
    all_proxies = set()
    for url in FREE_PROXY_SOURCES:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                lines = r.text.splitlines()
                for line in lines:
                    if line.strip() and ":" in line:
                        all_proxies.add(line.strip())
        except: continue
    
    proxy_list = list(all_proxies)
    random.shuffle(proxy_list) 
    print(f"✅ Loaded {len(proxy_list)} Free Proxies.")
    return proxy_list

def check_free_proxy(proxy_ip):
    proxy_dict = {"http": f"http://{proxy_ip}", "https": f"http://{proxy_ip}"}
    try:
        start_time = time.time()
        r = requests.get(SPACE_URL, proxies=proxy_dict, timeout=8)
        latency = (time.time() - start_time) * 1000 
        
        if r.status_code == 200 and latency < 9000:
            return proxy_dict, latency
    except:
        return None

def get_working_free_proxy(proxy_list):
    if len(VERIFIED_POOL) > 0:
        print(f"⚡ FAST TRACK: Using verified proxy from Pool!")
        return VERIFIED_POOL.pop(0) 

    print(f"🔍 Scanning for NEW Proxy...")
    random.shuffle(proxy_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        batch_proxies = proxy_list[:500] 
        future_to_proxy = {executor.submit(check_free_proxy, p): p for p in batch_proxies}
        
        for future in concurrent.futures.as_completed(future_to_proxy):
            result = future.result()
            if result:
                proxy, latency = result
                print(f"🟢 Found NEW Proxy: {proxy['http']} ({latency:.0f}ms)")
                executor.shutdown(wait=False)
                return proxy
    
    print("❌ No STABLE proxy found. Retrying...")
    return None

# ==============================================================================
# 2. CONTENT GENERATION (Text, Voice, Images)
# ==============================================================================

def generate_ai_script(topic):
    print(f"\n🧠 Generating Script: {topic}...")
    url = "https://notegpt.io/api/v2/chat/stream"
    prompt = f"Write a mind-blowing, viral, and interesting fact about '{topic}'. Keep it short (under 50 words). Make it sound amazing. No emojis. no () brackets, Do not write 'Here is a fact'."
    
    headers = { "authority": "notegpt.io", "content-type": "application/json", "user-agent": "Mozilla/5.0" }
    payload = { "conversation_id": str(uuid.uuid4()), "message": prompt, "model": "gemini-3-pro-preview", "tone": "default" }

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, proxies=get_paid_proxy(), timeout=30)
            full_text = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').replace("data: ", "")
                    try:
                        if '"done": true' in decoded: continue
                        data = json.loads(decoded)
                        if "text" in data: full_text += data["text"]
                    except: pass
            
            if full_text and len(full_text.strip()) > 5:
                cleaned = re.sub(r'[\*\#\_`\[\]]', '', full_text).strip()
                cleaned = cleaned.replace("Did you know that", "").replace("Did you know", "").strip()
                if cleaned: cleaned = cleaned[0].lower() + cleaned[1:]
                final_script = f"Did you know that {cleaned}"
                print(f"✅ Script: {final_script}")
                return final_script
            time.sleep(2)
        except: time.sleep(2)

def generate_full_voice(text, voice="fable"):
    print(f"🎤 Generating Voice...")
    url = "https://ttsmp3.com/makemp3_ai.php"
    headers = { "Referer": "https://ttsmp3.com/ai", "User-Agent": "Mozilla/5.0" }
    payload = { "msg": text, "lang": voice, "source": "ttsmp3", "speed": "1.00" }

    while True:
        try:
            r = requests.post(url, data=payload, headers=headers, proxies=get_paid_proxy(), timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("success") == 1:
                    audio_r = requests.get(data.get("URL"))
                    filename = f"{OUTPUT_FOLDER}/full_voice.mp3"
                    with open(filename, "wb") as f: f.write(audio_r.content)
                    print("✅ Voice Saved.")
                    return filename
            time.sleep(2)
        except: time.sleep(2)

# 🔥 NEW: Single Image Generator for Threading
def generate_single_image(index, prompt):
    # Style Selection
    styles = [
        f"Cinematic wide photograph shot of {prompt}, shot on 35mm film, Kodak Portra 400, depth of field, soft sunlight, realistic atmosphere, 4k",
        f"Cinematic photograph of {prompt}, motion blur, gopro style, authentic colors, high resolution photography",
        f"Dramatic photograph of {prompt}, studio lighting, rim light, volumetric fog, highly detailed, shot on Sony A7R IV"
    ]
    current_prompt = styles[index % len(styles)]
    current_prompt += ", no cartoon, no 3d render, no illustration, no drawing"
    
    HOME_URL = "https://flatai.org/"
    API_URL = "https://flatai.org/wp-admin/admin-ajax.php"
    
    print(f"   🎨 Starting Image {index+1}...")

    while True:
        try:
            s = requests.Session()
            # 1. API Call uses PAID PROXY (To avoid bans)
            s.proxies.update(get_paid_proxy())
            
            r = s.get(f"{HOME_URL}?t={int(time.time())}", timeout=30)
            match = re.search(r'"ai_generate_image_nonce"\s*:\s*"([a-zA-Z0-9]{10})"', r.text)
            if match:
                payload = { "action": "ai_generate_image", "nonce": match.group(1), "prompt": current_prompt, "aspect_ratio": "9:16", "style_model": "realistic" }
                api_r = s.post(API_URL, data=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                data = api_r.json()
                if data.get("success"):
                    # 2. Download uses LOCAL INTERNET (No Proxy - Faster & Saves Data)
                    # We use a fresh requests.get() without session proxies
                    img_url = data["data"]["images"][0]
                    img_r = requests.get(img_url, timeout=30) 
                    
                    fname = f"{OUTPUT_FOLDER}/src_image_{index}.jpg"
                    with open(fname, "wb") as f: f.write(img_r.content)
                    
                    if os.path.getsize(fname) > 1024:
                        print(f"   ✅ Image {index+1} Downloaded.")
                        return (index, fname) # Return index to keep order
            time.sleep(2)
        except: time.sleep(2)

def generate_images_parallel(prompt, count=4):
    print(f"\n🎨 Generating {count} Images (Parallel Mode)...")
    image_paths = [None] * count
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(count):
            futures.append(executor.submit(generate_single_image, i, prompt))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, fname = future.result()
                image_paths[idx] = fname
            except Exception as e:
                print(f"   ❌ Image Gen Failed: {e}")
                
    return [p for p in image_paths if p is not None]

# ==============================================================================
# 3. IMAGE TO VIDEO (Uses FREE Proxies)
# ==============================================================================

def convert_image_to_video_hf(image_path, free_proxy_list):
    print(f"🎬 Converting {os.path.basename(image_path)}...")
    output_vid_name = image_path.replace(".jpg", "_ai_vid.mp4")
    
    while True:
        good_proxy = get_working_free_proxy(free_proxy_list)
        
        if not good_proxy:
            time.sleep(2)
            continue

        try:
            session = requests.Session()
            session.proxies.update(good_proxy)
            
            # --- A. Upload ---
            with open(image_path, 'rb') as f:
                files = {'files': f}
                upload_res = session.post(f"{SPACE_URL}/gradio_api/upload", files=files, timeout=60)
            
            if upload_res.status_code != 200:
                raise Exception(f"Upload Fail: {upload_res.status_code}")
                
            uploaded_path = upload_res.json()[0]
            
            # --- B. Queue ---
            session_hash = str(uuid.uuid4())[:10]
            payload = {
                "data": [
                    {"path": uploaded_path, "url": f"{SPACE_URL}/file={uploaded_path}", 
                     "orig_name": "img.jpg", "size": 0, "mime_type": "image/jpeg", "meta": {"_type": "gradio.FileData"}},
                    "make this image come alive, cinematic motion, smooth animation", 
                    6, "low quality, error", 3.5, 1, 1, 42, True
                ],
                "fn_index": 0, "session_hash": session_hash
            }
            session.post(f"{SPACE_URL}/gradio_api/queue/join", json=payload, timeout=60)
            
            # --- C. Listen ---
            stream_url = f"{SPACE_URL}/gradio_api/queue/data?session_hash={session_hash}"
            res = session.get(stream_url, stream=True, timeout=300)
            
            video_url = None
            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8').replace('data: ', '')
                    try:
                        data = json.loads(decoded)
                        if data.get('msg') == 'process_completed':
                            vid_data = data.get('output', {}).get('data', [])[0]
                            video_url = vid_data.get('url') or f"{SPACE_URL}/file={vid_data['path']}"
                            break
                    except: pass
            
            if video_url:
                r = session.get(video_url, stream=True, timeout=60)
                if r.status_code == 200:
                    with open(output_vid_name, 'wb') as f:
                        f.write(r.content)
                    print(f"✅ Video Created: {output_vid_name}")
                    VERIFIED_POOL.append(good_proxy) 
                    return output_vid_name
            
            raise Exception("No Video URL")

        except Exception as e:
            print(f"   ⚠️ Proxy Failed ({str(e)[:30]}). Discarding...")
            continue

# ==============================================================================
# 4. FINAL RENDERING
# ==============================================================================

def get_transcription(audio_path):
    print("👂 Whisper AI is listening...")
    model = whisper.load_model("base") 
    result = model.transcribe(audio_path, word_timestamps=True)
    word_segments = []
    for segment in result['segments']:
        for word in segment['words']:
            word_segments.append({
                "word": word['word'].strip().upper(),
                "start": word['start'],
                "end": word['end']
            })
    return word_segments

def create_full_ai_video(audio_path, video_paths, word_segments):
    print("\n🎬 Stitching Final AI Video...")
    try:
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration + 0.5
        
        clips = []
        for vid_path in video_paths:
            try:
                clip = VideoFileClip(vid_path).without_audio().resize(height=1920)
                if clip.w > 1080:
                    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920)
                else:
                    clip = clip.resize(width=1080)
                    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920)
                clips.append(clip)
            except Exception as e:
                print(f"⚠️ Error loading clip {vid_path}: {e}")

        if not clips:
            print("❌ No valid clips found!")
            return

        final_video_base = concatenate_videoclips(clips, method="compose")
        if final_video_base.duration < total_duration:
            final_video_base = final_video_base.loop(duration=total_duration)
        else:
            final_video_base = final_video_base.subclip(0, total_duration)

        text_clips = []
        for item in word_segments:
            txt = item["word"]
            start = item["start"]
            end = item["end"]
            txt_clip = TextClip(txt, fontsize=110, color='yellow', font='Arial-Bold', stroke_color='black', stroke_width=5, method='caption', align='center', size=(1000, None))
            txt_clip = txt_clip.set_position('center').set_start(start).set_duration(end - start)
            text_clips.append(txt_clip)

        final = CompositeVideoClip([final_video_base] + text_clips)
        final = final.set_audio(audio_clip)
        
        output = f"{OUTPUT_FOLDER}/FINAL_FULL_AI_VIDEO_{uuid.uuid4().hex[:5]}.mp4"
        
        # 🔥 UPDATED: Removed verbose=False so you can see the progress bar
        final.write_videofile(output, fps=24, codec="libx264", audio_codec="libmp3lame", preset="ultrafast", threads=8)
        print(f"\n🎉 DONE! Full AI Video: {output}")

    except Exception as e:
        print(f"❌ Render Error: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    start_global = time.time() # ⏱️ Timer Starts
    
    topic = input("Enter Topic: ")
    
    # 1. Script & Audio
    script = generate_ai_script(topic)
    if script:
        audio = generate_full_voice(script)
        if audio:
            # 2. Determine videos
            audio_len = AudioFileClip(audio).duration
            CALC_DURATION = 3.0 
            needed_clips = int(audio_len / CALC_DURATION) + 1
            print(f"ℹ️ Audio Duration: {audio_len}s | Mode: {CALC_DURATION}s calc | Need {needed_clips} clips.")

            # 3. Generate Images (PARALLEL NOW)
            images = generate_images_parallel(topic, count=needed_clips)
            
            # 4. Convert Images to Video (PARALLEL)
            free_proxies = get_all_free_proxies() 
            ai_videos = [None] * len(images)
            
            if images:
                print(f"\n🚀 Converting images to AI Videos (Parallel: 4 Threads)...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_index = {
                        executor.submit(convert_image_to_video_hf, img, free_proxies): i 
                        for i, img in enumerate(images)
                    }
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            vid_path = future.result()
                            if vid_path:
                                ai_videos[idx] = vid_path
                                print(f"   ✅ Clip {idx+1} Ready.")
                        except Exception as e:
                            print(f"   ❌ Clip {idx+1} Failed: {e}")

                ai_videos = [v for v in ai_videos if v is not None]

                # 5. Render
                if ai_videos:
                    words = get_transcription(audio)
                    create_full_ai_video(audio, ai_videos, words)
                else:
                    print("❌ Video conversion failed completely.")

    # ⏱️ Timer Ends
    end_global = time.time()
    elapsed = end_global - start_global
    print(f"\n⏱️ Total Process Time: {int(elapsed // 60)} mins {int(elapsed % 60)} secs")
