//------------------------ Start UI ------------------------//


// Adjust interval as needed
// Set the favicon
let data;
const link = document.querySelector("link[rel~='icon']") || (() => {
  const newLink = document.createElement('link');
  newLink.rel = 'icon';
  document.head.appendChild(newLink);
  return newLink;
})();
link.href = './src/Img/icons8-heart-80.png';



//---------------------------------------------------------------------------------------------//
// Translate toggle

// DOMContentLoaded event listener (runs after the HTML document is fully loaded)
document.addEventListener('DOMContentLoaded', function() {
  const translateCheckbox = document.getElementById('translate-checkbox');
  const toggleLabel = document.querySelector('.translate-option');
  
  // Disable transition initially
  toggleLabel.style.transition = 'none';

  // Load saved state
  translateCheckbox.checked = localStorage.getItem('translate-checkbox') === 'true';
  
  // Re-enable transition after a short delay
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      toggleLabel.style.transition = '';
    });
  });
  
  // Save state on change and add animation
  translateCheckbox.addEventListener('change', () => {
    localStorage.setItem('translate-checkbox', translateCheckbox.checked);
    
    // Add animation
    toggleLabel.style.transition = 'all 0.3s ease';
  });
});


// Function to toggle loading indicator
function toggleLoadingIndicator(show) {
  const indicator = document.getElementById('loading-indicator');
  if (indicator) {
    indicator.style.display = show ? 'flex' : 'none';
  }
}


// Function to translate text using Google Translate API
async function translateText(text, sourceLang = 'vi', targetLang = 'en') {
  if (!text) return ''; // Return empty string if text is empty
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;

  try {
    const response = await fetch(url);
    const data = await response.json();
    return data[0].map(item => item[0]).join('');
  } catch (error) {
    console.error('Translation error:', error);
    return text; // Return original text if translation fails
  }
}

//---------------------------------------------------------------------------------------------//

// Debounce function (delays function execution for a certain time)
function debounce(func, delay) {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), delay);
  };
}


// document.addEventListener('DOMContentLoaded', () => {
//   connectWebSocket();
//   performCombinedSearch();
// });



//---------------------------------------------------------------------------------------------//
// Update UI by toggle effect

document.addEventListener('DOMContentLoaded', function() {
  const toggleSwitch = document.getElementById('mode-toggle');
  toggleSwitch.addEventListener('change', togglePanelLayout);
  
  // Initially hide the new-right-panel
  document.querySelector('.show-image-1').style.display = 'block';
  
  // Add event listener for the new-right-panel
  document.getElementById("images-rows").addEventListener('click', (event) => {
    if (event.target.classList.contains('result')) {
      showVideo(event.target);
      event.stopPropagation(); // Prevent the preview frame from being hidden
    } else if (event.target.closest('.similarity_search')) {
      event.stopPropagation();
      const imgElement = event.target.closest('.img-dis').querySelector('img');
      if (imgElement && typeof imgElement.id !== 'undefined') {
        console.log(imgElement.id);
        console.log(data.kq[imgElement.id - 1].id);
        performSimilaritySearch(data.kq[imgElement.id - 1].id);
      } else {
        console.error("Invalid image for similarity search in new-right-panel");
      }
    }
  });
});



// Set up event listener for the right panel
document.getElementById("list-photo").addEventListener('click', (event) => {
  // Check if the clicked element or its parent has the 'result' class
  const resultElement = event.target.closest('.result');
  if (resultElement) {

    // Try different selectors to find the image
    const imgElement = resultElement.querySelector('img') || 
                       resultElement.querySelector('.result-image') || 
                       event.target.closest('img');


    if (event.ctrlKey) {
      // Ctrl+click on a result: perform group search
      event.preventDefault();
      event.stopPropagation();
      
      if (imgElement && data.kq) {

        const index = parseInt(imgElement.id) - 1;
        const imageData = data.kq[index];

        if (imageData && imageData.id) {
          performGroupSearch(imageData.id);
        } else {
          console.error("Invalid image data for group search");
        }
      } else {
        console.error("No image found for group search");
      }
    } else {
      // Normal click on a result: show video
      showVideo(resultElement);
      event.stopPropagation(); // Prevent the preview frame from being hidden
    }
  } else if (event.target.classList.contains('similarity_search')) {
    // Handle similarity search click
    event.stopPropagation();
    const imgElement = event.target.closest('.img-dis').querySelector('img');
    if (imgElement && data.kq) {
      const imageId = data.kq[parseInt(imgElement.id) - 1]?.id;
      if (imageId) {
        performSimilaritySearch(imageId);
      } else {
        console.error("Invalid image ID for similarity search");
      }
    } else {
      console.error("No image found for similarity search");
    }
  }
});


// Function to toggle the layout of the right panel
function togglePanelLayout() {
  const showImage_1 = document.querySelector('.show-image-1');
  const showImage_2 = document.querySelector('.show-image-2');
  
  if (this.checked) {
    // Switch to row slider view
    showImage_1.style.display = 'none';
    showImage_2.style.display = 'block';
  } else {
    // Switch back to grid view
    showImage_1.style.display = 'block';
    showImage_2.style.display = 'none';
  }
}


// Disable right-click context menu on the entire page
document.addEventListener('contextmenu', function(event) {
  event.preventDefault();
  if (event.target.closest('.img-dis')) {
    showVideoFrames(event.target.closest('.img-dis'));
  }

  if (event.target.closest('.export-image-container')) {
    showVideoFrames(event.target.closest('.export-image-container'));
  }
});







//------------------------ Shortcut card ------------------------//

document.addEventListener('DOMContentLoaded', function() {
  const shortcutIcon = document.getElementById('shortcut-icon');
  const shortcutCard = document.getElementById('shortcut-card');
  const shortcutList = document.getElementById('shortcut-list');

  const shortcuts = [
    { key: 'Enter', description: 'Trigger the search button' },
    { key: 'Shift + Enter', description: 'Trigger the filter button' },
    { key: '/', description: 'Focus on the textbox in first search card' },
    { key: 'Shift + /', description: 'Cycle through textboxes' },
    { key: 'Alt + W', description: 'Toggle switch view' },
    { key: 'Alt + E', description: 'Toggle translate' },
    { key: 'Ctrl + I', description: 'Add search OCR textarea' },
    { key: 'Ctrl + J', description: 'Add search object element' },
    { key: 'Ctrl + K', description: 'Add search ASR textarea' },
    { key: 'Ctrl + H', description: 'Add a new search scene' },
    { key: 'Ctrl + Q', description: 'Reset left panel' },
    { key: 'Ctrl + E', description: 'Clear all textareas' },
    { key: 'Alt + R', description: 'Switch text and image tabs in search card 1' },
    { key: 'Alt + T', description: 'Switch text and image tabs in search card 2' },
    { key: 'Alt + A', description: 'Toggle export area' },
    { key: 'Alt + S', description: 'Reset export area' },
    { key: 'Alt + D', description: 'Toggle object list' },
    { key: 'Alt + X', description: 'Enable preview mode' }
  ];

  function populateShortcutList() {
    shortcutList.innerHTML = '';
    shortcuts.forEach(shortcut => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="shortcut-key">${shortcut.key}:</span> ${shortcut.description}`;
      shortcutList.appendChild(li);
    });
  }

  shortcutIcon.addEventListener('click', function() {
    if (shortcutCard.style.display === 'none') {
      populateShortcutList();
      shortcutCard.style.display = 'block';
    } else {
      shortcutCard.style.display = 'none';
    }
  });

  function closeShortcutCard() {
    shortcutCard.style.display = 'none';
  }

  // Close the card when clicking outside of it
  document.addEventListener('click', function(event) {
    if (!shortcutCard.contains(event.target) && event.target !== shortcutIcon) {
      shortcutCard.style.display = 'none';
    }
  });

  // Close the card when pressing Escape key
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
      closeShortcutCard();
    }
  });
});
