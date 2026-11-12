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
  // Official online server: "https://eventretrieval.one"
  // Contest onsite server:  "http://192.168.20.164:5000"
  DRES_BASE: "https://eventretrieval.one",

  // Default image format extension: '.jpg' (AIC dataset standard) or '.webp'
  IMAGE_EXT: ".jpg"
};

// Global shortcuts for legacy script compatibility
window.BACKEND_BASE = window.CONFIG.BACKEND_BASE;
window.BACKEND_WS = window.CONFIG.BACKEND_WS;
window.MEDIA_BASE = window.CONFIG.MEDIA_BASE;
window.DRES_BASE = window.CONFIG.DRES_BASE;

console.log("[CONFIG] System configuration loaded:", window.CONFIG);
