// Updated object mapping structure
const objectToFramesMap = new Map();
let isObjectListVisible = false;
let selectedObjects = new Set(); // Store selected objects


// Extract unique objects from results and map them to their frames.
function extractUniqueObjects(results) {
  objectToFramesMap.clear(); // Clear existing mappings

  results.forEach(result => {
    const video = result.entity.video;
    const frameId = result.entity.frame_id;
    
    Object.keys(result.entity.object || {}).forEach(objectType => {
      if (!objectToFramesMap.has(objectType)) {
        objectToFramesMap.set(objectType, []);
      }
      objectToFramesMap.get(objectType).push({ video, frameId });
    });
  });

  return Array.from(objectToFramesMap.keys());
}


// Updates the object list and handles re-selecting previously selected objects.
function updateObjectList(results) {
  const previouslySelectedObjects = new Set(selectedObjects);
  objectList = extractUniqueObjects(results);

  // Retain selected objects that still exist in the updated list
  selectedObjects = new Set([...previouslySelectedObjects].filter(obj => objectToFramesMap.has(obj)));
  
  if (isObjectListVisible) {
    renderObjectList();
  }
}


// Renders the object list UI, creating checkboxes for each object type.
function renderObjectList() {
  const objectListElement = document.getElementById('object-list');
  objectListElement.innerHTML = '';

  objectToFramesMap.forEach((frames, obj) => {
    const item = document.createElement('li');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = obj;
    checkbox.id = `object-${obj}`;
    checkbox.checked = selectedObjects.has(obj); // Set checked state based on selectedObjects
    checkbox.addEventListener('change', handleObjectSelection);

    const label = document.createElement('label');
    label.htmlFor = `object-${obj}`;
    label.textContent = `${obj} (${frames.length})`;

    item.appendChild(checkbox);
    item.appendChild(label);
    objectListElement.appendChild(item);
  });

  updateSelectedObjects();
  filterImagesByObjects(Array.from(selectedObjects)); // Apply filtering based on selected objects
}


// Closes the object list
function closeObjectList() {
  const objectListContent = document.querySelector('.object-list-content');
  if (objectListContent && isObjectListVisible) {
    objectListContent.classList.remove('show');
    isObjectListVisible = false;
  }
}


// Toggles the visibility of the object list.
function toggleObjectList() {
  const objectListContent = document.querySelector('.object-list-content');
  isObjectListVisible = !isObjectListVisible;
  
  if (isObjectListVisible) {
    objectListContent.classList.add('show');

    // Focus vào chat-textarea sau khi submit tên
    document.getElementById('search-bar').focus();

    renderObjectList();
  } else {
    objectListContent.classList.remove('show');
  }
}


// Handles the selection or deselection of an object.
function handleObjectSelection(event) {
  const checkbox = event.target;
  if (checkbox.checked) {
    selectedObjects.add(checkbox.value);
  } else {
    selectedObjects.delete(checkbox.value);
  }

  filterImagesByObjects(Array.from(selectedObjects));
  updateSelectedObjects();
}


// Filters the displayed images based on the selected objects.
function filterImagesByObjects(selectedObjects) {
  const allImages = document.querySelectorAll('.img-dis');

  if (selectedObjects.length === 0) {
    allImages.forEach(imgContainer => imgContainer.style.display = 'block');
    return;
  }

  const relevantFrames = new Set();
  selectedObjects.forEach(obj => {
    const frames = objectToFramesMap.get(obj);
    if (frames) {
      frames.forEach(frame => relevantFrames.add(`${frame.video}-${frame.frameId}`));
    }
  });

  allImages.forEach(imgContainer => {
    const frameInfo = imgContainer.querySelector('.infor').textContent;
    if (relevantFrames.has(frameInfo)) {
      imgContainer.style.display = 'block';
    } else {
      imgContainer.style.display = 'none';
    }
  });
}

// Updates the displayed selected objects list in the UI.
function updateSelectedObjects() {
  const selectedObjectsContainer = document.getElementById('selected-objects');
  if (!selectedObjectsContainer) {
    console.warn("Selected objects container not found in the DOM");
    return;
  }
  
  selectedObjectsContainer.innerHTML = Array.from(selectedObjects).map(value => 
    `<span class="selected-object">${value}</span>`
  ).join(' ');
}


// DOMContentLoaded event listener to initialize functionality once the DOM is fully loaded.
document.addEventListener('DOMContentLoaded', function() {
  // Move all your initialization code here
  const searchBar = document.getElementById('search-bar');

  const objectListButton = document.getElementById('show-object-list');
  if (objectListButton) {
    objectListButton.addEventListener('click', function(event) {
      event.preventDefault(); // Prevent any default button behavior
      toggleObjectList();
    });
  }

  // Modify the close modal event listener
  const closeModal = document.querySelector('.close-modal');
  if (closeModal) {
    closeModal.addEventListener('click', () => closeObjectList());
  }

  if (searchBar) {
    searchBar.addEventListener('input', function() {
      const searchTerm = this.value.toLowerCase();
      const listItems = document.querySelectorAll('#object-list li');

      listItems.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    });
  }

  const clearAllButton = document.querySelector('.clear-all');
  if (clearAllButton) {
    clearAllButton.addEventListener('click', () => {
      selectedObjects.clear();
      const checkboxes = document.querySelectorAll('.object-list-content input[type="checkbox"]');
      checkboxes.forEach(cb => {
        cb.checked = false;
      });
      filterImagesByObjects([]);
      updateSelectedObjects();
    });
  }

  // Add an event listener for the Esc key
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && isObjectListVisible) {
      closeObjectList();
    }
  });

  // Ensure 'results' exists before updating object list
  if (typeof results !== 'undefined') {
    updateObjectList(results);
  } else {
    console.warn("'results' is not defined. Make sure it's available before calling updateObjectList.");
  }
});



