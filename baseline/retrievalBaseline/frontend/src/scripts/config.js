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
  DRES_EVALUATION_ID: "675a3430-abaa-4d8c-b650-c8db1246d429",
  DRES_SESSION_ID: "AlUHzp2QiqjhGxRY5pWP9kDbdVH9PKvY",

  // Default image format extension: '.jpg' (AIC dataset standard) or '.webp'
  IMAGE_EXT: ".jpg"
};

// Global shortcuts for legacy script compatibility
window.BACKEND_BASE = window.CONFIG.BACKEND_BASE;
window.BACKEND_WS = window.CONFIG.BACKEND_WS;
window.MEDIA_BASE = window.CONFIG.MEDIA_BASE;
window.DRES_BASE = window.CONFIG.DRES_BASE;

// Replace only the known stale contest settings; preserve other saved sessions.
try {
  if (localStorage.getItem('evaluationID') === "bcb482c6-2b9e-4908-877a-e5ccd7c04640") {
    localStorage.setItem('evaluationID', window.CONFIG.DRES_EVALUATION_ID);
  }
  if (localStorage.getItem('contestSessionID') === "q-ekxzxUEkkxlpMwfkdcCvqt4CONUS6P") {
    localStorage.setItem('contestSessionID', window.CONFIG.DRES_SESSION_ID);
  }
} catch (error) {
  console.warn('[CONFIG] Browser storage unavailable');
}
