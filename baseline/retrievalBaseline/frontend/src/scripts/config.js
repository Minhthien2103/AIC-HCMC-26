// ==============================================================================
// Global Configuration for Baseline Retrieval UI
// ==============================================================================

window.CONFIG = {
  // Backend REST API (Local RTX 4060 GPU Server)
  BACKEND_BASE: "http://localhost:8006",

  // Backend WebSocket endpoint
  BACKEND_WS: "ws://localhost:8006",

  // Remote Media Server (via SSH Tunnel on port 8007)
  MEDIA_BASE: "http://localhost:8007",

  // DRES Evaluation Server
  // LAN Server: "http://192.168.20.150:8080" or "http://192.168.20.150"
  // UIT Contest server: "https://aiclub.uit.edu.vn/dres"
  DRES_BASE: "https://aiclub.uit.edu.vn/dres",
  DRES_EVALUATION_ID: "bcb482c6-2b9e-4908-877a-e5ccd7c04640",
  DRES_SESSION_ID: "q-ekxzxUEkkxlpMwfkdcCvqt4CONUS6P",

  // Default image format extension: '.jpg' (AIC dataset standard) or '.webp'
  IMAGE_EXT: ".jpg"
};

// Global shortcuts for legacy script compatibility
window.BACKEND_BASE = window.CONFIG.BACKEND_BASE;
window.BACKEND_WS = window.CONFIG.BACKEND_WS;
window.MEDIA_BASE = window.CONFIG.MEDIA_BASE;
window.DRES_BASE = window.CONFIG.DRES_BASE;

console.log("[CONFIG] System configuration loaded:", window.CONFIG);
