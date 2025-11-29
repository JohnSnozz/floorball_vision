import os
import random
import subprocess
from tqdm import tqdm

# Ordner definieren
input_folder = "raw_videos"
output_folder = "screenshot_raw"

# Ausgabeordner erstellen, falls nicht vorhanden
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Alle mp4-Dateien im Eingabeordner finden
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    base_name = os.path.splitext(video_file)[0]
    
    # Videolänge ermitteln (in Sekunden)
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
    
    # 100 zufällige Zeitpunkte auswählen
    timestamps = sorted([random.uniform(0, duration) for _ in range(100)])
    
    # Für jeden Zeitpunkt ein Screenshot erstellen
    for i, timestamp in enumerate(tqdm(timestamps, desc=f"Verarbeite {video_file}")):
        output_path = os.path.join(output_folder, f"{base_name}_frame_{i+1:03d}.jpg")
        
        # FFmpeg-Befehl zum Extrahieren eines Frames
        cmd = [
            'ffmpeg', '-ss', str(timestamp), '-i', video_path, 
            '-vf', 'scale=1920:1080', '-vframes', '1', 
            '-q:v', '2', output_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Fehler bei {output_path}: {e}")

print("Alle Screenshots wurden erstellt.")