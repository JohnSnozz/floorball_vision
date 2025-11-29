import os
import random
import shutil
from pathlib import Path

# Definiere Pfade
source_folder = "screenshot_raw"
target_folder = "screenshot_batches"

# Stelle sicher, dass der Zielordner existiert
Path(target_folder).mkdir(exist_ok=True)

# Liste alle JPG-Dateien
all_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
#print(f"Gefunden: {len(all_files)} JPG-Dateien")

# Mische die Dateien zufällig
random.shuffle(all_files)

# Berechne die Batch-Größe (etwa gleich für alle Batches)
batch_size = len(all_files) // 20
remainder = len(all_files) % 20

# Erstelle 20 Batches
for batch_num in range(20):
    # Bestimme Batch-Größe (verteile die Reste auf die ersten Batches)
    current_batch_size = batch_size + (1 if batch_num < remainder else 0)
    
    # Berechne Start- und Endindizes für diesen Batch
    start_idx = batch_num * batch_size + min(batch_num, remainder)
    end_idx = start_idx + current_batch_size
    
    # Erstelle Batch-Ordner
    batch_folder = os.path.join(target_folder, f"batch_{batch_num+1}")
    Path(batch_folder).mkdir(exist_ok=True)
    
    # Kopiere Dateien in den Batch-Ordner
    batch_files = all_files[start_idx:end_idx]
    for file in batch_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(batch_folder, file)
        shutil.copy2(source_path, target_path)
    
    print(f"Batch {batch_num+1} erstellt mit {len(batch_files)} Dateien")

print("Fertig! Alle Batches wurden erstellt.")