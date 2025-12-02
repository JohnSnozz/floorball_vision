/**
 * Floorball Vision - Main Application
 *
 * Frontend-Logik für Video-Upload und -Verwaltung.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elemente
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadPercent = document.getElementById('upload-percent');
    const uploadBar = document.getElementById('upload-bar');

    const youtubeForm = document.getElementById('youtube-form');
    const youtubeUrl = document.getElementById('youtube-url');
    const youtubeProgress = document.getElementById('youtube-progress');
    const youtubePercent = document.getElementById('youtube-percent');
    const youtubeBar = document.getElementById('youtube-bar');

    const videoList = document.getElementById('video-list');
    const videoTemplate = document.getElementById('video-template');

    // Videos laden
    loadVideos();

    // File Upload: Klick
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());

        // Drag & Drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('border-blue-500', 'bg-blue-50');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
    }

    // File Input Change
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileUpload(fileInput.files[0]);
            }
        });
    }

    // YouTube Form
    if (youtubeForm) {
        youtubeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = youtubeUrl.value.trim();
            if (!url) return;

            try {
                youtubeProgress.classList.remove('hidden');
                youtubePercent.textContent = 'Starte...';
                youtubeBar.style.width = '0%';

                const video = await API.videos.downloadYoutube(url);
                youtubeUrl.value = '';

                // Status Polling starten
                pollVideoStatus(video.id, (status) => {
                    if (status === 'downloading') {
                        youtubePercent.textContent = 'Downloading...';
                        youtubeBar.style.width = '50%';
                    } else if (status === 'processing') {
                        youtubePercent.textContent = 'Verarbeite...';
                        youtubeBar.style.width = '75%';
                    }
                }, () => {
                    youtubeProgress.classList.add('hidden');
                    loadVideos();
                });

            } catch (error) {
                alert('Fehler: ' + error.message);
                youtubeProgress.classList.add('hidden');
            }
        });
    }

    /**
     * Datei hochladen
     */
    async function handleFileUpload(file) {
        // Validierung
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
            alert('Dateityp nicht erlaubt. Erlaubt: MP4, AVI, MOV, MKV, WebM');
            return;
        }

        try {
            uploadProgress.classList.remove('hidden');

            const video = await API.videos.upload(file, (percent) => {
                uploadPercent.textContent = `${percent}%`;
                uploadBar.style.width = `${percent}%`;
            });

            // Status Polling für Verarbeitung
            pollVideoStatus(video.id, (status) => {
                if (status === 'processing') {
                    uploadPercent.textContent = 'Verarbeite...';
                }
            }, () => {
                uploadProgress.classList.add('hidden');
                uploadBar.style.width = '0%';
                loadVideos();
            });

        } catch (error) {
            alert('Upload fehlgeschlagen: ' + error.message);
            uploadProgress.classList.add('hidden');
        }

        fileInput.value = '';
    }

    /**
     * Video Status pollen
     */
    function pollVideoStatus(videoId, onUpdate, onComplete) {
        const interval = setInterval(async () => {
            try {
                const data = await API.videos.getStatus(videoId);

                if (onUpdate) onUpdate(data.status);

                if (data.status === 'ready' || data.status === 'error') {
                    clearInterval(interval);
                    if (onComplete) onComplete(data);
                }
            } catch (error) {
                clearInterval(interval);
                if (onComplete) onComplete({ status: 'error', error: error.message });
            }
        }, 2000);
    }

    /**
     * Videos laden und anzeigen
     */
    async function loadVideos() {
        try {
            const videos = await API.videos.list();

            if (!videoList) return;

            if (videos.length === 0) {
                videoList.innerHTML = `
                    <div class="p-6 text-center text-gray-500">
                        Noch keine Videos vorhanden. Lade ein Video hoch oder starte einen YouTube Download.
                    </div>
                `;
                return;
            }

            videoList.innerHTML = '';

            videos.forEach(video => {
                const item = createVideoItem(video);
                videoList.appendChild(item);
            });

        } catch (error) {
            console.error('Fehler beim Laden der Videos:', error);
            if (videoList) {
                videoList.innerHTML = `
                    <div class="p-6 text-center text-red-500">
                        Fehler beim Laden der Videos: ${error.message}
                    </div>
                `;
            }
        }
    }

    /**
     * Video-Element erstellen
     */
    function createVideoItem(video) {
        const template = videoTemplate.content.cloneNode(true);
        const item = template.querySelector('.video-item');

        // Daten setzen
        item.dataset.id = video.id;

        const title = item.querySelector('.video-title');
        title.textContent = video.original_filename || video.filename || 'Unbekannt';

        const meta = item.querySelector('.video-meta');
        const duration = video.duration ? formatDuration(video.duration) : '-';
        const resolution = video.width && video.height ? `${video.width}x${video.height}` : '-';
        meta.textContent = `${duration} | ${resolution}`;

        // Status Badge
        const badge = item.querySelector('.status-badge');
        const statusConfig = {
            'ready': { text: 'Bereit', class: 'bg-green-100 text-green-800' },
            'processing': { text: 'Verarbeitet...', class: 'bg-yellow-100 text-yellow-800' },
            'downloading': { text: 'Download...', class: 'bg-blue-100 text-blue-800' },
            'error': { text: 'Fehler', class: 'bg-red-100 text-red-800' },
            'uploaded': { text: 'Hochgeladen', class: 'bg-gray-100 text-gray-800' },
        };
        const status = statusConfig[video.status] || statusConfig['uploaded'];
        badge.textContent = status.text;
        badge.className = `status-badge inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${status.class}`;

        // Thumbnail (Platzhalter)
        const thumb = item.querySelector('.video-thumb');
        thumb.src = '/static/img/video-placeholder.svg';
        thumb.alt = video.original_filename || 'Video';

        // Preview Button (nur wenn ready)
        const previewBtn = item.querySelector('.btn-preview');
        if (video.status === 'ready') {
            previewBtn.classList.remove('hidden');
            previewBtn.addEventListener('click', () => {
                openVideoPreview(video);
            });
        }

        // Löschen Button
        const deleteBtn = item.querySelector('.btn-delete');
        deleteBtn.addEventListener('click', async () => {
            if (!confirm('Video wirklich löschen?')) return;

            try {
                await API.videos.delete(video.id);
                loadVideos();
            } catch (error) {
                alert('Fehler beim Löschen: ' + error.message);
            }
        });

        return item;
    }

    /**
     * Dauer formatieren
     */
    function formatDuration(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);

        if (h > 0) {
            return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    /**
     * Video Preview Modal öffnen
     */
    function openVideoPreview(video) {
        const modal = document.getElementById('preview-modal');
        const player = document.getElementById('preview-player');
        const source = document.getElementById('preview-source');
        const title = document.getElementById('preview-title');
        const meta = document.getElementById('preview-meta');

        if (!modal || !player || !source) return;

        // Video-Daten setzen
        title.textContent = video.original_filename || video.filename || 'Video Preview';
        source.src = `/api/videos/${video.id}/stream`;
        player.load();

        // Metadaten
        const duration = video.duration ? formatDuration(video.duration) : '-';
        const resolution = video.width && video.height ? `${video.width}x${video.height}` : '-';
        meta.textContent = `${duration} | ${resolution}`;

        // Modal anzeigen
        modal.classList.remove('hidden');

        // Automatisch abspielen
        player.play().catch(() => {});
    }

    /**
     * Video Preview Modal schliessen
     */
    function closeVideoPreview() {
        const modal = document.getElementById('preview-modal');
        const player = document.getElementById('preview-player');

        if (!modal || !player) return;

        player.pause();
        player.currentTime = 0;
        modal.classList.add('hidden');
    }

    // Preview Modal Event Listeners
    const previewModal = document.getElementById('preview-modal');
    const previewClose = document.getElementById('preview-close');
    const previewFullscreen = document.getElementById('preview-fullscreen');
    const previewPlayer = document.getElementById('preview-player');

    if (previewClose) {
        previewClose.addEventListener('click', closeVideoPreview);
    }

    if (previewModal) {
        // Schliessen bei Klick auf Hintergrund
        previewModal.addEventListener('click', (e) => {
            if (e.target === previewModal) {
                closeVideoPreview();
            }
        });
    }

    if (previewFullscreen && previewPlayer) {
        previewFullscreen.addEventListener('click', () => {
            if (previewPlayer.requestFullscreen) {
                previewPlayer.requestFullscreen();
            } else if (previewPlayer.webkitRequestFullscreen) {
                previewPlayer.webkitRequestFullscreen();
            }
        });
    }

    // ESC-Taste schliesst Modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeVideoPreview();
        }
    });

    // Global verfügbar machen
    window.loadVideos = loadVideos;
    window.openVideoPreview = openVideoPreview;
    window.closeVideoPreview = closeVideoPreview;
});
