// MP4 preview: one player and one set of listeners for the modal.
class VideoPlayer {
    constructor(videoElementId, videoSrc, initialTime = 0) {
        this.video = document.getElementById(videoElementId);
        this.initialTime = 0;
        this.pendingSeek = null;
        this.video.addEventListener('loadedmetadata', () => {
            if (document.getElementById('Details').style.display === 'none') return;
            if (this.pendingSeek !== null) {
                this.seek(this.pendingSeek);
                this.pendingSeek = null;
            }
            this.play();
        });
        this.video.addEventListener('error', () => this.setStatus(
            'Không tải được video. Kiểm tra đường dẫn, kết nối media server hoặc định dạng video.'));
        this.video.addEventListener('waiting', () => this.setStatus('Đang tải video…'));
        this.video.addEventListener('playing', () => this.setStatus(''));
        document.getElementById('rewindBtn')?.addEventListener('click', () => this.skip(-5));
        document.getElementById('forwardBtn')?.addEventListener('click', () => this.skip(5));
        document.getElementById('initialTimeBtn')?.addEventListener('click', () => this.seek(this.initialTime));
        this.load(videoSrc, initialTime);
    }

    setStatus(text) {
        const status = document.getElementById('video-status');
        if (status) status.textContent = text;
    }

    load(src, time = 0) {
        this.video.pause();
        this.initialTime = Number.isFinite(Number(time)) ? Math.max(0, Number(time)) : 0;
        this.pendingSeek = this.initialTime;
        this.videoSrc = src;
        this.setStatus('Đang tải video…');
        const link = document.getElementById('video-direct-link');
        if (link) link.href = src;
        this.video.src = src;
        this.video.load();
    }

    play() {
        this.video.play().catch(error => {
            if (error.name === 'AbortError') return;
            this.setStatus(error.name === 'NotAllowedError'
                ? 'Bấm nút Play để phát video.'
                : 'Không phát được video. Thử mở video trực tiếp bằng liên kết bên dưới.');
        });
    }

    pause() { this.video.pause(); }
    seek(time) {
        if (!Number.isFinite(Number(time))) return;
        if (this.video.readyState < 1) {
            this.pendingSeek = Math.max(0, Number(time));
            return;
        }
        const upper = Number.isFinite(this.video.duration) ? this.video.duration : Infinity;
        this.video.currentTime = Math.min(upper, Math.max(0, Number(time)));
    }
    skip(seconds) { this.seek(this.video.currentTime + seconds); }
}

async function getVideo(videoName) {
    const mediaBase = (window.CONFIG && (window.CONFIG.VIDEO_BASE || window.CONFIG.MEDIA_BASE))
        || window.MEDIA_BASE || 'http://localhost:8007';
    const name = String(videoName).replace(/\.mp4$/i, '');
    return `${mediaBase.replace(/\/$/, '')}/videos/${encodeURIComponent(name)}.mp4`;
}

async function openVideoPreview(videoName, time) {
    const videoElement = document.getElementById('vid_details');
    document.getElementById('Details').style.display = 'block';
    const src = await getVideo(videoName);
    if (!videoElement.playerInstance) {
        videoElement.playerInstance = new VideoPlayer('vid_details', src, time);
    } else {
        videoElement.playerInstance.load(src, time);
    }
}

async function showVideo(img) {
    const entity = img.resultEntity || data?.kq?.[Number(img.id) - 1]?.entity;
    if (!entity) return;
    await openVideoPreview(entity.video, entity.time);
}

// Release the media resource so a hidden preview cannot keep buffering over VPN.
function closeVideoPreview() {
    document.getElementById('Details').style.display = 'none';
    const player = document.getElementById('vid_details');
    player.pause();
    if (player.playerInstance) player.playerInstance.pendingSeek = null;
    player.removeAttribute('src');
    player.load();
}

// Close video details when Escape key is pressed
document.addEventListener("keydown", event => {
    if (event.key === 'Escape') {
        event.preventDefault();
        const detailsDiv = document.getElementById('Details');
        if (detailsDiv.style.display === 'block') {
            closeVideoPreview();
        }
    }
});
  

// Close modal on click of close button or outside of modal
const modal= document.getElementById("Details")
const video= document.getElementById("vid_details")

document.getElementsByClassName("close")[0]?.addEventListener("click", () => {
    closeVideoPreview();
});


window.addEventListener("click", function(event){
    if (event.target == modal) {
        closeVideoPreview();
    }
});


// Move video
const draggableBar = document.querySelector('.draggable-bar');
const detailsBg = document.querySelector('.details_bg');
const detailsContainer = document.querySelector('#Details');

let isDragging = false;
let startX, startY, startLeft, startTop;

draggableBar.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startLeft = detailsBg.offsetLeft;
    startTop = detailsBg.offsetTop;
    e.preventDefault();
});

document.addEventListener('mouseup', () => {
    isDragging = false;
});

document.addEventListener('mousemove', (e) => {
    if (isDragging) {
        const deltaX = e.clientX - startX;
        const deltaY = e.clientY - startY;

        let newLeft = startLeft + deltaX;
        let newTop = startTop + deltaY;

        // Lấy kích thước của màn hình
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;

        // Giới hạn newLeft để không di chuyển ra khỏi màn hình bên phải
        newLeft = Math.min(newLeft, screenWidth - detailsBg.offsetWidth);

        // Giới hạn newTop để không di chuyển ra khỏi màn hình phía trên và dưới
        newTop = Math.max(44, Math.min(newTop, screenHeight - detailsBg.offsetHeight));

        detailsBg.style.left = `${newLeft}px`;
        detailsBg.style.top = `${newTop}px`;
    }
});
