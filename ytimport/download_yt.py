import yt_dlp
import os
import subprocess
import sys
from tqdm import tqdm

# FFmpeg-Installation prüfen
try:
    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
except (subprocess.SubprocessError, FileNotFoundError):
    print("FFmpeg ist nicht installiert. Bitte installiere es über:")
    if sys.platform.startswith('linux'):
        print("sudo apt install ffmpeg")
    elif sys.platform == 'darwin':
        print("brew install ffmpeg")
    elif sys.platform == 'win32':
        print("https://ffmpeg.org/download.html herunterladen oder choco install ffmpeg")
    sys.exit(1)

# Unterordner erstellen
if not os.path.exists('raw_videos'):
    os.makedirs('raw_videos')

# YouTube-URLs aus Datei lesen
with open('youtube_links.txt', 'r') as file:
    urls = [line.strip() for line in file if line.strip()]

# Anzahl der URLs anzeigen
total_videos = len(urls)
print(f"Insgesamt {total_videos} Videos zum Herunterladen gefunden.")

# Funktion zur Video-ID-Extraktion
def get_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url and 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return None

# Zähler für übersprungene und heruntergeladene Videos
skipped = 0
downloaded = 0

# Videos herunterladen mit Fortschrittsanzeige
for i, url in enumerate(tqdm(urls, desc="Gesamtfortschritt", unit="video")):
    try:
        # Video-ID extrahieren
        video_id = get_video_id(url)
        if not video_id:
            print(f"Konnte keine Video-ID für {url} extrahieren. Überspringe.")
            skipped += 1
            continue
        
        # Zuerst Titel abrufen
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', f'video_{video_id}')
            
        # Prüfen, ob bereits heruntergeladen
        vorhandene_dateien = [f for f in os.listdir('raw_videos') 
                             if os.path.splitext(f)[0] == title]
        if vorhandene_dateien:
            print(f"Video '{title}' existiert bereits als {vorhandene_dateien[0]}. Überspringe.")
            skipped += 1
            continue
        
        # Download-Optionen mit eigenem Fortschritt für aktuelles Video
        download_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            'outtmpl': f'raw_videos/%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'quiet': False,
            'progress_hooks': [lambda d: print(f"\rVideo-Fortschritt: {d['_percent_str'] if '_percent_str' in d else '...'}", end="") 
                             if d['status'] == 'downloading' else None]
        }
        
        print(f"\nStarte Download ({i+1}/{total_videos}): '{title}'")
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
            print(f"\nVideo '{title}' erfolgreich heruntergeladen.")
            downloaded += 1
            
    except Exception as e:
        print(f"\nFehler beim Verarbeiten von {url}: {e}")
        
    # Status nach jedem Video anzeigen
    remaining = total_videos - (i + 1)
    print(f"Status: {downloaded} heruntergeladen, {skipped} übersprungen, {remaining} verbleibend")

print(f"\nDownload abgeschlossen: {downloaded} Videos heruntergeladen, {skipped} übersprungen.")