//------------------------ Preview ------------------------//

// Show the left preview with the provided image source and frame information
function showLeftPreview(src, frameInfo, positionIndex) {

  const previewImage = document.getElementById('preview-image');
  const leftPreview = document.querySelector('.left-preview');
  const previewInfor = leftPreview.querySelector('.infor');
  const positionIndexElement = leftPreview.querySelector('.positionIndex');
  
  previewImage.src = src;
  previewInfor.textContent = frameInfo;
  positionIndexElement.textContent = positionIndex;
  leftPreview.classList.add('visible');
  updatePreviewFrameVisibility();
}

// Show the current frame preview on the right side
function showCurrentFramePreview() {
  const rightPreview = document.querySelector('.right-preview');
  updateCurrentPreview();
  rightPreview.classList.add('visible');
  updatePreviewFrameVisibility();
}

// Hide the left preview
function hideLeftPreview() {
  const leftPreview = document.querySelector('.left-preview');
  leftPreview.classList.remove('visible');
  updatePreviewFrameVisibility();
}

// Hide the current frame preview on the right side
function hideCurrentFramePreview() {
  const rightPreview = document.querySelector('.right-preview');
  rightPreview.classList.remove('visible');
  updatePreviewFrameVisibility();
}

// Check the visibility of the preview frame and update its display accordingly
function checkPreviewFrameVisibility() {
  const previewFrame = document.getElementById('preview-frame');
  const leftPreview = document.querySelector('.left-preview');
  const rightPreview = document.querySelector('.right-preview');
  
  if (leftPreview.classList.contains('visible') || rightPreview.classList.contains('visible')) {
    previewFrame.style.display = 'flex';
  } else {
    previewFrame.style.display = 'none';
  }
}

// Update the visibility of the preview frame
function updatePreviewFrameVisibility() {
  const previewFrame = document.getElementById('preview-frame');
  const leftPreview = document.querySelector('.left-preview');
  const rightPreview = document.querySelector('.right-preview');
  
  if (leftPreview.classList.contains('visible') || rightPreview.classList.contains('visible')) {
    previewFrame.style.display = 'block';
  } else {
    previewFrame.style.display = 'none';
  }
}

// Hide the entire preview frame and reset visibility of both previews
function hidePreviewFrame() {
  const previewFrame = document.getElementById('preview-frame');
  const leftPreview = document.querySelector('.left-preview');
  const rightPreview = document.querySelector('.right-preview');
  previewFrame.style.display = 'none';
  leftPreview.classList.remove('visible');
  rightPreview.classList.remove('visible');
}


//-----------------------------------------------------------------------//

// Update the current preview on the right side based on the current frame
function updateCurrentPreview() {
  const currentFrame = document.querySelector('.current-frame');
  const currentPreview = document.getElementById('current-preview');
  const currentPreviewInfo = document.querySelector('.right-preview .infor');
  
  if (currentFrame) {
    const frameContainer = currentFrame.closest('.frame-container');
    currentPreview.src = currentFrame.src;
    currentPreviewInfo.textContent = frameContainer.querySelector('.infor').textContent;
  }
}

//-----------------------------------------------------------------------//

// Check if both previews are hidden and update the display of the preview frame accordingly
function checkAndHidePreviewFrame() {
  const leftPreview = document.querySelector('.left-preview');
  const rightPreview = document.querySelector('.right-preview');
  const previewFrame = document.getElementById('preview-frame');
  
  if (!leftPreview.classList.contains('visible') && !rightPreview.classList.contains('visible')) {
    previewFrame.style.display = 'none';
  } else {
    previewFrame.style.display = 'block';
  }
}

// Main event listener to handle various key presses, mouse events, and focus/blur events
document.addEventListener('DOMContentLoaded', () => {

  const previewFrame = document.getElementById('preview-frame');
  let isAltPressed = false;
  let isPreviewModeEnabled = false;

  let escPressCount = 0;
  const escPressResetTime = 500; // time after press reset, ms

  //Reset the preview state when necessary
  function resetPreviewState() {
    isAltPressed = false;
    isPreviewModeEnabled = false;
  }

  // Hide the preview frame when certain keys are pressed
  function hidePreviewFrame() {
    previewFrame.style.display = 'none';
    document.querySelector('.left-preview').classList.remove('visible');
    document.querySelector('.right-preview').classList.remove('visible');
  }

  // Handle the Escape key press and manage the preview frame visibility
  function handleEscPress() {
    escPressCount++;
    if (escPressCount === 2) {
      hidePreviewFrame();
      escPressCount = 0;
    }
    setTimeout(() => {
      escPressCount = 0;
    }, escPressResetTime);

  }

  // Track when the Alt key is pressed down and prevent the default action.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Alt') {
      isAltPressed = true;
      e.preventDefault();
    } else if (e.altKey && e.key === 'x') {
      isPreviewModeEnabled = !isPreviewModeEnabled;
      e.preventDefault();
    } else if (e.key === 'Escape') {
      handleEscPress();
    }
  });

  // Track when the Alt key is released and prevent the default action.
  document.addEventListener('keyup', (e) => {
    if (e.key === 'Alt') {
      isAltPressed = false;
      e.preventDefault();
    }
  });

  // Show the preview frame with the image when the Alt key is pressed and the mouse is over a result.
  document.addEventListener('mouseover', (e) => {
  if ((isAltPressed || isPreviewModeEnabled) && (e.target.classList.contains('result') || e.target.classList.contains('export-image'))) {
    const imgDis = e.target.closest('.img-dis, .export-image-container');
    const frameInfo = imgDis.querySelector('.infor').textContent;

    const imgElement = imgDis.querySelector('.result, .export-image');
    const positionIndex = imgElement.id;

    showLeftPreview(e.target.src, frameInfo, positionIndex);
    e.preventDefault();
  }
});

  // Reset Alt state when window loses focus
  window.addEventListener('blur', resetPreviewState);

  // Add event listeners for close buttons
  const closeButtons = document.querySelectorAll('.close-preview-button');
  closeButtons.forEach(button => {
    button.addEventListener('click', function() {
      const previewContainer = this.closest('.preview-container');
      if (previewContainer.classList.contains('left-preview')) {
        hideLeftPreview();
      } else {
        hideCurrentFramePreview();
      }
    });
  });
});

// Observer to watch for changes in the video frames and update the current preview accordingly
const videoFramesObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
      const targetElement = mutation.target;
      if (targetElement.classList.contains('current-frame')) {
        updateCurrentPreview();
      }
    }
  });
});

const videoFrames = document.getElementById('video-frames');
if (videoFrames) {
  videoFramesObserver.observe(videoFrames, { attributes: true, subtree: true, attributeFilter: ['class'] });
}





//------------------------------------------------------------------------------------------//


// Show the image in fullscreen mode with navigation controls
function showFullscreenImage(src, isLeftPreview = false) {
  const container = document.getElementById('fullscreen-image-container');
  const image = document.getElementById('fullscreen-image');
  const navigation = document.getElementById('fullscreen-navigation');
  const imageIndex = document.getElementById('image-index');
  
  image.src = src;
  container.style.display = 'flex';
  container.dataset.previewType = isLeftPreview ? 'left' : 'right';

  if (isLeftPreview) {
    const leftPreview = document.querySelector('.left-preview');
    const positionIndex = leftPreview.querySelector('.positionIndex').textContent;
    imageIndex.textContent = positionIndex;
    imageIndex.style.display = 'block';
  } else {
    const rightPreview = document.querySelector('.right-preview');
    const frameInfo = rightPreview.querySelector('.infor').textContent;
    const frameNumber = frameInfo.split('-')[1];
    imageIndex.textContent = frameNumber;
    imageIndex.style.display = 'block';
  }

  navigation.style.display = 'block';
  document.addEventListener('keydown', handleEscapeKey);
}

// Hide the fullscreen image and remove event listeners
function hideFullscreenImage() {
  const container = document.getElementById('fullscreen-image-container');
  container.style.display = 'none';

  // Remove event listener for Escape key
  document.removeEventListener('keydown', handleEscapeKey);
}

// Handle Escape key press to exit fullscreen mode
function handleEscapeKey(event) {
  if (event.key === 'Escape') {
    hideFullscreenImage();
  }
}

// Navigate between images in fullscreen mode using direction
function navigateFullscreenImage(direction) {
  const container = document.getElementById('fullscreen-image-container');
  const image = document.getElementById('fullscreen-image');
  const imageIndex = document.getElementById('image-index');
  const isLeftPreview = container.dataset.previewType === 'left';
  const frame_id= image.src.split('/')[14].split('_')[1]
  console.log(frame_id)
  if (isLeftPreview) {
    const currentIndex = parseInt(imageIndex.textContent);
    const newIndex = currentIndex + direction;
    
    const nextImage = document.querySelector(`.result[id="${newIndex}"], .export-image[id="${newIndex}"]`);
    
    if (nextImage) {
      image.src = nextImage.src;
      imageIndex.textContent = newIndex;
      // Update left preview information
      const leftPreview = document.querySelector('.left-preview');
      const previewImage = leftPreview.querySelector('#preview-image');
      const previewInfo = leftPreview.querySelector('.infor');
      const positionIndexElement = leftPreview.querySelector('.positionIndex');
      
      previewImage.src = nextImage.src;
      previewInfo.textContent = nextImage.closest('.img-dis').querySelector('.infor').textContent;
      positionIndexElement.textContent = newIndex;
    }
  } else {
    // Navigation for right preview
    // Right preview navigation (updated)
    const currentFrame = parseInt(imageIndex.textContent);
    const newFrame = currentFrame + direction;
    
    const framesContainer = document.querySelector('.frames-container');
    const directory = framesContainer.dataset.directory;
    
    // Find the index of the current frame in the globalFrameList
    const currentIndex = globalFrameList.indexOf(currentFrame);
    if (currentIndex === -1) {
      console.error("Current frame not found in global frame list");
      return;
    }
    
    // Calculate the new index
    const newIndex = currentIndex + direction;
    if (newIndex < 0 || newIndex >= globalFrameList.length) {
      console.log("Reached the end of frame list");
      return;
    }
    
    // Get the new frame number from the globalFrameList
    const newFrameNumber = globalFrameList[newIndex];
    
    // Update the fullscreen image
    const newSrc = `${directory}keyframe_${newFrameNumber}.webp`;
    image.src = newSrc;
    imageIndex.textContent = newFrameNumber;
    
    // Update the right preview
    const rightPreview = document.querySelector('.right-preview');
    const currentPreview = rightPreview.querySelector('#current-preview');
    const currentPreviewInfo = rightPreview.querySelector('.infor');
    
    currentPreview.src = newSrc;
    const videoName = directory.split('/').filter(part => part.startsWith('L') && part.includes('V'))[0];
    currentPreviewInfo.textContent = `${videoName}-${newFrameNumber}`;
    
    // Update the main frame in the video frames section
    updateMainFrame(newFrameNumber, directory, `${videoName}-${newFrameNumber}`);
  }
}

// Event listeners for fullscreen image controls
document.addEventListener('DOMContentLoaded', () => {
  const previewFrame = document.getElementById('preview-frame');
  const fullscreenContainer = document.getElementById('fullscreen-image-container');
  const closeFullscreenButton = document.getElementById('close-fullscreen-button');
  const prevFrameButton = document.getElementById('prev-frame-button');
  const nextFrameButton = document.getElementById('next-frame-button');

  previewFrame.addEventListener('dblclick', (e) => {
    if (e.target.tagName === 'IMG') {
      const isLeftPreview = e.target.closest('.preview-container').classList.contains('left-preview');
      showFullscreenImage(e.target.src, isLeftPreview);
    }
  });

  closeFullscreenButton.addEventListener('click', hideFullscreenImage);

  prevFrameButton.addEventListener('click', () => navigateFullscreenImage(-1));
  nextFrameButton.addEventListener('click', () => navigateFullscreenImage(1));

  fullscreenContainer.addEventListener('click', (e) => {
    if (e.target === fullscreenContainer) {
      hideFullscreenImage();
    }
  });
});


const prevFrameButton = document.getElementById('prev-frame-button');
const nextFrameButton = document.getElementById('next-frame-button');

prevFrameButton.addEventListener('click', () => navigateFullscreenImage(-1));
nextFrameButton.addEventListener('click', () => navigateFullscreenImage(1));

// Navigate to the next frame in fullscreen mode
function handleFullscreenKeyPress(event) {
  if (document.getElementById('fullscreen-image-container').style.display === 'flex') {
    if (event.key === 'ArrowLeft') {
      navigateFullscreenImage(-1);
    } else if (event.key === 'ArrowRight') {
      navigateFullscreenImage(1);
    }
  }
}

document.addEventListener('keydown', handleFullscreenKeyPress);

