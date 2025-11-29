/**
 * Floorball Vision - API Client
 *
 * JavaScript API Client für die Flask Backend API.
 */

const API = {
    baseUrl: '/api',

    /**
     * Generische Fetch-Funktion
     */
    async fetch(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const response = await fetch(url, { ...defaultOptions, ...options });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unbekannter Fehler' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return response.json();
    },

    /**
     * Video API
     */
    videos: {
        /**
         * Alle Videos abrufen
         */
        async list() {
            return API.fetch('/videos');
        },

        /**
         * Video Details abrufen
         */
        async get(id) {
            return API.fetch(`/videos/${id}`);
        },

        /**
         * Video Status abrufen
         */
        async getStatus(id) {
            return API.fetch(`/videos/${id}/status`);
        },

        /**
         * Video-Datei hochladen
         */
        async upload(file, onProgress) {
            const formData = new FormData();
            formData.append('file', file);

            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        try {
                            const error = JSON.parse(xhr.responseText);
                            reject(new Error(error.error || `HTTP ${xhr.status}`));
                        } catch {
                            reject(new Error(`HTTP ${xhr.status}`));
                        }
                    }
                });

                xhr.addEventListener('error', () => reject(new Error('Netzwerkfehler')));

                xhr.open('POST', `${API.baseUrl}/videos/upload`);
                xhr.send(formData);
            });
        },

        /**
         * YouTube Video herunterladen
         */
        async downloadYoutube(url) {
            return API.fetch('/videos/youtube', {
                method: 'POST',
                body: JSON.stringify({ url }),
            });
        },

        /**
         * Video löschen
         */
        async delete(id) {
            return API.fetch(`/videos/${id}`, {
                method: 'DELETE',
            });
        },
    },
};

// Global verfügbar machen
window.API = API;
