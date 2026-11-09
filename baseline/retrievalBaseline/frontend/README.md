# Retrieval Frontend

This is the **frontend** of the Retrieval project. It provides a lightweight, browser-based interface for querying the backend service, visualizing search results, and interacting with image/text-based retrieval workflows.

---

## ✨ Features

* **Search UI**: Enter text queries or upload images to query the backend.
* **WebSocket Support (optional)**: Live query/response updates (can be toggled).
* **Result Visualization**: Grid/list view of retrieved frames, grouped by video.
* **Export Tools**: Collect, preview, and export query results.
* **Health Badge**: Automatically checks backend status.

---

## 📂 Project Structure

```
frontend/
├── index.html                # Entry point
├── login.html                # Optional login page
├── src/
│   ├── scripts/              # Core JavaScript (query logic, sockets, UI updates)
│   ├── styles/               # CSS files
│   ├── Img/                  # Icons, assets
│   └── websocket/            # Multi-query WebSocket handlers
└── service-worker.js         # Basic PWA support
```

---

## ⚙️ Configuration

The frontend reads its backend target from **`src/scripts/config.js`**:

```js
window.BACKEND_BASE = 'http://localhost:8000';   // REST API base
window.WS_URL = (window.BACKEND_BASE).replace(/^http/, 'ws') + '/ws';
window.USE_WS = false; // Disable legacy sockets by default
```

To change backend target (for deployment):

```html
<script>
  window.BACKEND_BASE = 'https://your-domain.com';
</script>
```

---

## 🌐 Hosting with Nginx

Nginx is a fast and reliable way to serve the static frontend.

### 1. Install Nginx

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install nginx -y
```

### 2. Place frontend files

Copy the built frontend into a directory Nginx can serve:

```bash
sudo mkdir -p /var/www/retrieval-frontend
sudo cp -r frontend/* /var/www/retrieval-frontend/
```

### 3. Configure Nginx

Edit a site config (e.g., `/etc/nginx/sites-available/retrieval`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/retrieval-frontend;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    # (Optional) Proxy API requests to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/retrieval /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Reference Docs

* [Nginx Beginner’s Guide](https://nginx.org/en/docs/beginners_guide.html)
* [Nginx Official Documentation](https://nginx.org/en/docs/)

---

## 🚀 Quick Start

1. Run your backend on `http://localhost:8000`.
2. Serve this frontend via Nginx or any static server.
3. Open `http://your-domain.com` in the browser.
4. Start querying and visualizing results!