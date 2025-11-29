import os
import shutil
import string

def reorganize_photos():
    # Hauptordner
    main_folder = "screenshot_batches"
    
    # Alle Batch-Unterordner durchlaufen
    for i in range(1, 21):
        batch_folder = f"batch_{i}"
        batch_path = os.path.join(main_folder, batch_folder)
        
        # Prüfen, ob der Batch-Ordner existiert
        if not os.path.exists(batch_path):
            continue
        
        # Alle Dateien im Batch-Ordner sammeln
        files = [f for f in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, f))]
        
        # Nur weitermachen, wenn Dateien vorhanden
        if not files:
            continue
            
        # Anzahl der benötigten Unterordner berechnen (100 Dateien pro Ordner)
        num_subfolders = (len(files) + 99) // 100  # Aufrunden
        
        # Unterordner mit Buchstaben erstellen
        for j in range(num_subfolders):
            subfolder_name = string.ascii_lowercase[j]  # a, b, c, ...
            subfolder_path = os.path.join(batch_path, subfolder_name)
            
            # Unterordner erstellen, falls nicht vorhanden
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            
            # Dateien für diesen Unterordner berechnen
            start_idx = j * 100
            end_idx = min((j + 1) * 100, len(files))
            current_files = files[start_idx:end_idx]
            
            # Dateien in den Unterordner verschieben
            for file in current_files:
                src = os.path.join(batch_path, file)
                dst = os.path.join(subfolder_path, file)
                shutil.move(src, dst)
                
        print(f"Batch {batch_folder} wurde in {num_subfolders} Unterordner aufgeteilt")

if __name__ == "__main__":
    reorganize_photos()