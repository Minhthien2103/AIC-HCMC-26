//------------------------ Show video ------------------------//


// VideoPlayer class to handle video playback and HLS streaming
class VideoPlayer {
    constructor(videoElementId, videoSrc) {
        this.video = document.getElementById(videoElementId);
        this.videoSrc = videoSrc;
        this.hls = null;
        this.initPlayer();

        this.initialTime = 0;
        this.addCustomControls();
    }
  
    // Initialize the video player depending on browser support
    initPlayer() {
        if (Hls.isSupported()) {
            this.initHlsPlayer();
        } else if (this.video.canPlayType('application/vnd.apple.mpegurl')) {
            this.initNativePlayer();
        } else {
            console.error('HLS is not supported in this browser.');
        }
        this.addEventListeners();
    }
  
    // Initialize HLS.js player
    initHlsPlayer() {
        this.hls = new Hls({ 
            enableWorker: true,
            lowLatencyMode: true
        });
        this.hls.loadSource(this.videoSrc);
        this.hls.attachMedia(this.video);
        this.hls.on(Hls.Events.MANIFEST_PARSED, () => {
            this.video.play().catch(e => console.error('Auto-play was prevented:', e));
        });
    }
  
    // Initialize native player for HLS streams if supported
    initNativePlayer() {
        this.video.src = this.videoSrc;
    }
  
    // Add event listeners to handle video playback events
    addEventListeners() {
        this.video.addEventListener('loadedmetadata', () => this.onLoadedMetadata());
        this.video.addEventListener('error', (e) => this.onError(e));
        this.video.addEventListener('waiting', () => this.onWaiting());
        this.video.addEventListener('canplay', () => this.onCanPlay());
        this.video.addEventListener('click', () => this.togglePlayPause());
    }
  
    onLoadedMetadata() {
        // console.log('Video metadata loaded');
    }
  
    onError(e) {
        // console.error('Video error:', e);
    }
  
    onWaiting() {
        // console.log('Video is buffering');
    }
  
    onCanPlay() {
        // console.log('Video can play');
    }

    togglePlayPause() {
        // console.log('Video pause');
    }
  
    play() {
        this.video.play().catch(e => console.error('Play failed:', e));
    }
  
    pause() {
        this.video.pause();
    }
  
    seek(time) {
        if (isNaN(time)) return;
        this.video.currentTime = time;
    }
  
    setQuality(level) {
        if (this.hls) {
            this.hls.currentLevel = level;
        }
    }
  
    // Clean up resources and remove event listeners
    destroy() {
        if (this.hls) {
            this.hls.destroy();
        }
        this.video.removeEventListener('loadedmetadata', this.onLoadedMetadata);
        this.video.removeEventListener('error', this.onError);
        this.video.removeEventListener('waiting', this.onWaiting);
        this.video.removeEventListener('canplay', this.onCanPlay);
        this.video.removeEventListener('click', this.togglePlayPause);
    }
    
    //----------------------------------
    addCustomControls() {
        document.getElementById('rewindBtn').addEventListener('click', () => this.skip(-10));
        document.getElementById('forwardBtn').addEventListener('click', () => this.skip(10));
        document.getElementById('initialTimeBtn').addEventListener('click', () => this.goToInitialTime());
    }
    
    skip(seconds) {
        this.video.currentTime += seconds;
    }
    
    goToInitialTime() {
        this.video.currentTime = this.initialTime;
    }
  }


// Generate the video URL based on video name
async function getVideo(videoName) {
    const mediaBase = (window.CONFIG && window.CONFIG.MEDIA_BASE) || window.MEDIA_BASE || "http://192.168.28.112:8007";
    return `${mediaBase}/videos/${videoName}.mp4`;
}



// Show video details
async function showVideo(img) {
    const detailsDiv = document.getElementById('Details')
    const videoElement = document.getElementById('vid_details');
    let player = null;

    // Show the details div
    detailsDiv.style.display = 'block';
  
    // Get video URL based on image data
    const videoSrc= await getVideo(data.kq[img.id-1].entity.video)

    console.log("videoName: " + data.kq[img.id-1].entity.video);
    console.log("img.id: " + img.id);
    console.log("videoSrc: " + videoSrc);

    // Initialize the VideoPlayer if it hasn't been already
    if (!videoElement.playerInstance) {
        videoElement.playerInstance = new VideoPlayer('vid_details', videoSrc);
        player = videoElement.playerInstance;
    } else {
        player = videoElement.playerInstance;
        player.videoSrc = videoSrc;
        player.initPlayer(); // Re-initialize the player with the new source
    }
  
    // Set the video current time and play the video
    player.video.currentTime = data.kq[img.id - 1].entity.time;
    player.play();

    console.log("currentTime 1: " + data.kq[img.id - 1].entity.time);

    // Ensure frame navigation still works
    const divVideoFrames = document.getElementById('video-frames');
    if (divVideoFrames.style.display === 'flex') {
        setupNavigationButtons();
        document.addEventListener('keydown', handleKeyPress);
    }
  
     
    // Set up event listeners for video control buttons
    document.getElementById('playBtn')?.addEventListener('', () => player.play());
    document.getElementById('pauseBtn')?.addEventListener('click', () => player.pause());
    document.getElementById('seekBtn')?.addEventListener('click', () => player.seek(3)); // Seek to 10 seconds
    document.getElementById('qualitySelect')?.addEventListener('change', (e) => player.setQuality(parseInt(e.target.value)));


    // Set the video current time and play the video
    const initialTime = data.kq[img.id - 1].entity.time;
    player.video.currentTime = initialTime;
    player.initialTime = initialTime;  // Store the initial time
    player.play();
}
  

// Close video details when Escape key is pressed
document.addEventListener("keydown", event => {
    if (event.key === 'Escape') {
        event.preventDefault();
        const detailsDiv = document.getElementById('Details');
        if (detailsDiv.style.display === 'block') {
            detailsDiv.style.display = 'none';
            document.getElementById('vid_details')?.pause(); // Pause video if open
        }
    }
});
  

// Close modal on click of close button or outside of modal
const modal= document.getElementById("Details")
const video= document.getElementById("vid_details")

document.getElementsByClassName("close")[0]?.addEventListener("click", () => {
    modal.style.display = "none";
    video.pause();
});


window.addEventListener("click", function(event){
    if (event.target == modal) {
        modal.style.display = "none";
        video.pause();
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