//------------------------ Show videoframe ------------------------//

// Global variable to store the frame list
let globalFrameList = [];
let globalSecondList =[];

// Display video frames and set up the UI
async function showVideoFrames(imgDiv) {
  // Get the container for video frames
  const divVideoFrames = document.getElementById('video-frames');
  const leftPanel = document.querySelector('.left-panel');
  const rightPanel = document.querySelector('.right-panel');

  // Show the video frames
  divVideoFrames.style.display = 'flex';

  // Adjust panel heights
  leftPanel.style.height = 'calc(100vh - 50px - 15vh)';
  rightPanel.style.height = 'calc(100vh - 60px - 15vh)';

  // Add necessary child elements
  divVideoFrames.innerHTML = `
    <div class="frames-container"></div>
    <div class="navigation-buttons">
      <div class="nav-row">
        <button class="nav-button" id="prev-button">←</button>
        <button class="nav-button" id="next-button">→</button>
      </div>
      <div class="nav-row">
        <button class="nav-button" id="show-preview-button">↑</button>
        <button class="nav-button" id="hide-preview-button">↓</button>
      </div>
      <button class="close-button">×</button>
    </div>
  `;
  
  const framesContainer = divVideoFrames.querySelector('.frames-container');

  const exportArea = document.getElementById('export-area');
  exportArea.addEventListener('dragover', allowDrop);
  exportArea.addEventListener('drop', drop);

  
  // Get the image from the clicked div
  const img = imgDiv.querySelector('img');
  if (img) {
    const imgPath = img.src;
    const infoDiv = imgDiv.querySelector('.infor');
    const imageInfo = infoDiv.textContent;
    
    const lastSlashIndex = imgPath.lastIndexOf('/');
    const directory = imgPath.substring(0, lastSlashIndex + 1);
    const originalFrameName = imgPath.substring(lastSlashIndex + 1).split('.')[0];
    const currentFrame = originalFrameName.includes('_') ? originalFrameName.split('_')[1] : originalFrameName;

    framesContainer.dataset.currentFrame = currentFrame;
    framesContainer.dataset.directory = directory;

    // Load the CSV file and populate the globalFrameList
    await loadFrameListFromCSV(directory, imageInfo);
    
    await updateFrames(framesContainer, directory, currentFrame, imageInfo);
  }

  // Add event listener for keyboard navigation
  document.addEventListener('keydown', handleKeyPress);

  // Add event listener to close the video-frames div when Escape key is pressed
  document.addEventListener('keydown', escapeHandler);
  
  setupNavigationButtons();
}

function setupNavigationButtons() {
  document.getElementById('prev-button').addEventListener('click', () => navigateFrames(-1));
  document.getElementById('next-button').addEventListener('click', () => navigateFrames(1));
  document.getElementById('show-preview-button').addEventListener('click', showCurrentFramePreview);
  document.getElementById('hide-preview-button').addEventListener('click', hideCurrentFramePreview);
  document.querySelector('.close-button').addEventListener('click', closeVideoFrames);
}

async function loadFrameListFromCSV(directory, imageInfo) {
  const videoName = directory.split('/').filter(part => part.startsWith('L') && part.includes('V'))[0] || (imageInfo ? imageInfo.split('-')[0] : '');
  const mediaBase = (window.CONFIG && window.CONFIG.MEDIA_BASE) || window.MEDIA_BASE || "http://localhost:8007";
  const csvFilePath = `${mediaBase}/map-keyframes/${videoName}.csv`;
  try {
    const response = await fetch(csvFilePath);
    if (!response.ok) {
      console.warn(`Could not load CSV from ${csvFilePath}, status: ${response.status}`);
      return;
    }
    const csvData = await response.text();

    const lines = csvData.trim().split('\n');
    globalSecondList = {};
    globalFrameList = [];
    lines.forEach(line => {
      const parts = line.split(',');
      if (parts.length >= 2 && !isNaN(parseFloat(parts[1]))) {
        const frameIdx = parseInt(parts[0].trim(), 10);
        const ptsTime = parseFloat(parts[1].trim());
        if (!isNaN(frameIdx)) {
          globalSecondList[String(frameIdx)] = ptsTime;
          globalFrameList.push(frameIdx);
        }
      }
    });

    if (globalFrameList.length === 0) {
      console.error('No valid frames found in CSV');
    }
  } catch (error) {
    console.error('Error loading CSV file:', error);
  }
}
  

// Create a container for a single frame
function createFrameContainer(src, frameNumber, isCurrent, frameInfo) {
  const frameContainer = document.createElement('div');
  frameContainer.className = 'frame-container';

  const frameElement = document.createElement('img');
  frameElement.src = src;
  frameElement.alt = 'Video Frame';
  frameElement.className = 'video-frame' + (isCurrent ? ' current-frame' : '');
  frameElement.dataset.frameNumber = frameNumber;

  const frameNameDiv = document.createElement('div');
  frameNameDiv.className = 'infor';

  frameNameDiv.textContent = `${frameInfo}`;

  const frameElements = document.querySelectorAll('.video-frame');
  frameElements.forEach((frame) => {
    frame.setAttribute('draggable', 'true');
    frame.addEventListener('dragstart', drag);
  });
  

  frameContainer.appendChild(frameElement);
  frameContainer.appendChild(frameNameDiv);

  frameContainer.addEventListener('click', () => {
    updateMainFrame(frameNumber, src.substring(0, src.lastIndexOf('/') + 1), frameInfo);
  });

  console.log(frameContainer);
  return frameContainer;
}


// Update the main frame and surrounding frames
async function updateMainFrame(newFrameNumber, directory, frameInfo) {
  const framesContainer = document.querySelector('.frames-container');
  framesContainer.dataset.currentFrame = newFrameNumber;
  framesContainer.dataset.directory = directory;

  // Remove current-frame-container class from all frame containers
  const allFrameContainers = framesContainer.querySelectorAll('.frame-container');
  allFrameContainers.forEach(container => {
    container.classList.remove('current-frame-container');
  });

  // Find the new current frame and update its classes
  const newCurrentFrame = framesContainer.querySelector(`[data-frame-number="${newFrameNumber}"]`);
  if (newCurrentFrame) {
    const newCurrentContainer = newCurrentFrame.closest('.frame-container');
    newCurrentContainer.classList.add('current-frame-container');
    newCurrentFrame.classList.add('current-frame');
  }

  await updateFrames(framesContainer, directory, newFrameNumber, frameInfo);
}


// Update the frames in the container
async function updateFrames(container, directory, currentFrame, currentFrameInfo) {

  if (globalFrameList.length === 0) {
    console.error('Frame list is empty');
    return;
  }

  const currentIndex = globalFrameList.indexOf(parseInt(currentFrame));
  const nearestIndex = currentIndex !== -1 ? currentIndex : globalFrameList.reduce((prev, curr, idx) => 
    Math.abs(curr - currentFrame) < Math.abs(globalFrameList[prev] - currentFrame) ? idx : prev, 0);

  const start = Math.max(0, nearestIndex - 20);
  const end = Math.min(globalFrameList.length, nearestIndex + 21);
  const framesToShow = globalFrameList.slice(start, end);

  await updateFramesSmooth(container, directory, currentFrame, framesToShow);
}

async function updateFramesSmooth(container, directory, currentFrame, framesToShow) {
  const fragment = document.createDocumentFragment();

  for (let frameNumber of framesToShow) {
    const frameContainer = document.createElement('div');
    frameContainer.className = 'frame-container';
    if (frameNumber.toString() === currentFrame.toString()) {
      frameContainer.classList.add('current-frame-container');
    }
    frameContainer.innerHTML = `
      <img class="video-frame ${frameNumber.toString() === currentFrame.toString() ? 'current-frame' : ''}" data-frame-number="${frameNumber}" alt="Video Frame">
      <div class="infor"></div>
    `;
    fragment.appendChild(frameContainer);
  }

  container.innerHTML = '';
  container.appendChild(fragment);

  const updatePromises = Array.from(container.children).map((frameContainer, index) => {
    const frameNumber = framesToShow[index].toString();
    
    return updateFrameContainer(frameContainer, frameNumber, directory, currentFrame.toString() === frameNumber);
  });

  await Promise.all(updatePromises);

  const currentFrameElement = container.querySelector('.current-frame');
  if (currentFrameElement) {
    currentFrameElement.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'center' });
  }
}


async function updateFrameContainer(frameContainer, frameNumber, directory, isCurrent) {
  const frameElement = frameContainer.querySelector('img');
  const frameNameDiv = frameContainer.querySelector('.infor');

  // Format frame number (with 3-digit padding for standard AIC dataset or keyframe_ format)
  const padded = String(frameNumber).padStart(3, '0');
  const ext = (window.CONFIG && window.CONFIG.IMAGE_EXT) || '.jpg';
  const framePath = directory.includes('keyframe_')
    ? `${directory}keyframe_${frameNumber}${ext}`
    : `${directory}${padded}${ext}`;

  frameElement.src = framePath;
  frameElement.onerror = () => {
    if (!frameElement.dataset.retried) {
      frameElement.dataset.retried = '1';
      frameElement.src = `${directory}keyframe_${frameNumber}.webp`;
    }
  };
  frameElement.className = 'video-frame' + (isCurrent ? ' current-frame' : '');
  frameElement.dataset.frameNumber = frameNumber;
  // Extract video name from the directory path
  const seconds_json = globalSecondList[`${frameNumber}`] !== undefined ? globalSecondList[`${frameNumber}`] : '';
  const videoName = directory.split('/').filter(part => part.startsWith('L') && part.includes('V'))[0] || '';
  const frameInfo = `${videoName}-${seconds_json}`;
  frameNameDiv.textContent = frameInfo;

  frameContainer.className = 'frame-container' + (isCurrent ? ' current-frame-container' : '');

  frameContainer.addEventListener('click', () => {
    updateMainFrame(frameNumber, directory, frameInfo);
  });

  frameElement.setAttribute('draggable', 'true');
  frameElement.addEventListener('dragstart', drag);
  frameContainer.addEventListener('mousedown', handleMiddleClick);
  frameContainer.addEventListener('contextmenu', handleRightClick);

  await new Promise((resolve) => {
    if (frameElement.complete) {
      resolve();
    } else {
      frameElement.onload = resolve;
      frameElement.onerror = resolve;
    }
  });
}



// Navigate between frames
function navigateFrames(direction) {
  const framesContainer = document.querySelector('.frames-container');
  const currentFrame = parseInt(framesContainer.dataset.currentFrame);
  const directory = framesContainer.dataset.directory;
  
  const currentIndex = globalFrameList.indexOf(currentFrame);
  
  if (currentIndex === -1) {
    console.error("Current frame not found in global frame list");
    return;
  }

  const newIndex = currentIndex + direction;
  if (newIndex < 0 || newIndex >= globalFrameList.length) {
    console.log("Reached the end of frame list");
    return;
  }

  const newFrame = globalFrameList[newIndex];
  framesContainer.dataset.currentFrame = newFrame;

  const videoName = framesContainer.querySelector('.infor').textContent.split('-')[0];
  const frameInfo = `${videoName}-${newFrame}`;

  updateMainFrame(newFrame, directory, frameInfo);
  updateCurrentPreview();
}


// Create an image element for a video frame
function createFrameElement(src, className) {
  const frame = document.createElement('img');
  frame.src = src;
  frame.alt = 'Video Frame';
  frame.classList.add('video-frame', className);
  return frame;
}


// Update the source and class of each frame in the container
function updateSurroundingFrames(container, directory, currentFrame) {
  const frames = container.querySelectorAll('.video-frame');
  const data_batch= directory.split('/')[4]
  const videoname=directory.split('/')[6]
  frames.forEach((frame, index) => {
    const frameNumber = currentFrame - 10 + index;
    if (data_batch==='data-batch-1' || (videoname <"L18_V001" && videoname>="L12_V001")){
      frame.src = `${directory}${frameNumber.toString().padStart(4, '0')}.webp`;
      }
      else{
      frame.src = `${directory}${frameNumber.toString().padStart(3, '0')}.webp`;
      }  
    frame.dataset.frameNumber = frameNumber;
    frame.className = 'video-frame' + (frameNumber === currentFrame ? ' current-frame' : ' other-frame');
  });
}


// Handle keyboard navigation
function handleKeyPress(event) {
  const videoPlayer = document.getElementById('vid_details');
  const isVideoPlaying = videoPlayer && !videoPlayer.paused;
  const fullscreenImage = document.getElementById('fullscreen-image-container');
  
  if (fullscreenImage.style.display === 'flex') {
    return; // Exit early if fullscreen image is displayed
  }

  if (isVideoPlaying) {
    return; // Exit if video is playing
  }

  const isInputElement = event.target.tagName === 'TEXTAREA' || event.target.tagName === 'INPUT';

  if (isInputElement) {
    return; // Exit if user is typing in an input
  }

  switch(event.key) {
    case 'ArrowLeft':
      event.preventDefault();
      navigateFrames(-1);
      updateCurrentPreview();
      break;
    case 'ArrowRight':
      event.preventDefault();
      navigateFrames(1);
      updateCurrentPreview();
      break;
    case 'ArrowUp':
      event.preventDefault();
      showCurrentFramePreview();
      break;
    case 'ArrowDown':
      event.preventDefault();
      // exportCurrentFrame();
      hideCurrentFramePreview();
      break;
    case '=':
      event.preventDefault();
      exportCurrentFrame();
      break;
  }
}


// New function to export the current frame
function exportCurrentFrame() {
  const currentFrameContainer = document.querySelector('.current-frame-container');
  if (currentFrameContainer) {
    const imgElement = currentFrameContainer.querySelector('img');
    const inforElement = currentFrameContainer.querySelector('.infor');
    
    if (imgElement && inforElement) {
      const frameId = inforElement.textContent.split('-')[1];
      const imageSrc = imgElement.src;
      const frameInfo = inforElement.textContent;
      
      addImageToExportArea(frameId, imageSrc, frameInfo);
    }
  }
}


// Close the video frames and reset UI
function closeVideoFrames() {
  const divVideoFrames = document.getElementById('video-frames');
  const leftPanel = document.querySelector('.left-panel');
  const rightPanel = document.querySelector('.right-panel');
  
  divVideoFrames.style.display = 'none';
  leftPanel.style.height = 'calc(100vh - 44px)';
  rightPanel.style.height = 'calc(100vh - 44px)';
  
  // Remove the event listener for keyboard navigation
  document.removeEventListener('keydown', handleKeyPress);
  document.removeEventListener('keydown', escapeHandler);
}

const escapeHandler = (event) => {
  if (event.key === 'Escape') {
    closeVideoFrames();
  }
};


//---------------------------------------------------------------------------------------------//
// Play video when right-click


function handleRightClick(event) {
  if (event.button === 2) { // Right mouse button
    event.preventDefault();
    const frameNumber = event.currentTarget.querySelector('.video-frame').dataset.frameNumber;
    const videoName = event.currentTarget.querySelector('.infor').textContent.split('-')[0];
    playVideoFromFrame(videoName, frameNumber);
  }
}



// Update playVideoFromFrame to be more consistent with showVideo
async function playVideoFromFrame(videoName, frameNumber) {
  console.log("videoName: " + videoName);
  console.log("frameNumber: " + frameNumber);
  const detailsDiv = document.getElementById('Details');
  const videoElement = document.getElementById('vid_details');
  let player = null;

  detailsDiv.style.display = 'block';

  const videoSrc = await getVideo(videoName);

  console.log("videoSrc: " + videoSrc);

  if (!videoElement.playerInstance) {
      videoElement.playerInstance = new VideoPlayer('vid_details', videoSrc);
      player = videoElement.playerInstance;
  } else {
      player = videoElement.playerInstance;
      player.videoSrc = videoSrc;
      player.initPlayer();
  }

  // Convert frame number to seconds
  // This conversion should match how frames are handled in your system
  // You might need to adjust this calculation based on your video's actual frame rate

  const timeInSeconds = globalSecondList[`${frameNumber}`]; // Assuming 25 fps

  player.video.currentTime = timeInSeconds;
  player.play();

  console.log("currentTime 2: " + timeInSeconds);

  // Ensure frame navigation still works
  const divVideoFrames = document.getElementById('video-frames');
  if (divVideoFrames.style.display === 'flex') {
      setupNavigationButtons();
      document.addEventListener('keydown', handleKeyPress);
  }
}


