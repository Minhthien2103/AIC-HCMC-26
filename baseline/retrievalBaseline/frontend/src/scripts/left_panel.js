//------------------------ Left Panel ------------------------//

function clearAllTextareas() {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
    textarea.value = '';
    });
}

// Function to focus on the first textbox in search-scene-1
function focusOnFirstTextbox() {
    const firstScene = document.querySelector('#search-scene-1');
    if (firstScene) {
        const textbox = firstScene.querySelector('textarea[name="Text_Query"]');
        if (textbox) {
            textbox.focus();
        }
    }
}

// Function to cycle through Text_Query textboxes
function cycleThroughTextboxes() {
    const scenes = document.querySelectorAll('.Search_Scene');
    if (scenes.length === 0) return;

    const activeElement = document.activeElement;
    let currentSceneIndex = -1;

    for (let i = 0; i < scenes.length; i++) {
        if (scenes[i].contains(activeElement)) {
            currentSceneIndex = i;
            break;
        }
    }

    const nextSceneIndex = (currentSceneIndex + 1) % scenes.length;
    const nextScene = scenes[nextSceneIndex];

    const nextTextQuery = nextScene.querySelector('textarea[name="Text_Query"]');
    if (nextTextQuery) {
        nextTextQuery.focus();
    }
}

//------------------------------------------------------------------------//


document.addEventListener('DOMContentLoaded', function() {
    const modelButtons = document.querySelectorAll('.model-option button');
    if (modelButtons && modelButtons.length > 0) {
        modelButtons.forEach(button => {
            button.addEventListener('click', function() {
                modelButtons.forEach(btn => btn.classList.remove('active'));            
                this.classList.add('active');
                const model = this.className.split(' ')[0];
                switchModel(model);
            });
        });
    }
});

function switchModel(model) {
    // Add your logic here to switch between models
    console.log(`Switched to ${model} model`);
}

//------------------------------------------------------------------------//
// Insert elements

// Insert a textarea for entering OCR (Optical Character Recognition) query
function insertOcrTextarea() {
    const activeElement = document.activeElement;
    const queryGroup = activeElement.closest('.query-group');
    
    if (queryGroup) {
        const textQuery = queryGroup.querySelector('textarea[name="Text_Query"]');
        if (textQuery && textQuery.style.display !== 'none') {
            const queryImageArea = queryGroup.querySelector('.query-content-area');
            
            const existingOcr = queryGroup.querySelector('textarea[name="Ocr_Query"]');
            if (!existingOcr) {
                const ocrContainer = document.createElement('div');
                ocrContainer.className = 'ocr-container';
                
                const newOcrTextarea = document.createElement('textarea');
                newOcrTextarea.name = 'Ocr_Query';
                newOcrTextarea.rows = '2';
                newOcrTextarea.placeholder = 'Search ocr';
                
                const closeButton = document.createElement('button');
                closeButton.innerHTML = '&times;';
                closeButton.className = 'close-ocr-button';
                closeButton.addEventListener('click', function() {
                    ocrContainer.remove();
                });
                
                ocrContainer.appendChild(newOcrTextarea);
                ocrContainer.appendChild(closeButton);
                
                queryImageArea.after(ocrContainer);

                // Focus on the newly created OCR textarea
                newOcrTextarea.focus();
            }
        }
    }
}

// Insert a textarea for entering ASM
function insertAsmTextarea() {
    const activeElement = document.activeElement;
    const queryGroup = activeElement.closest('.query-group');
    
    if (queryGroup) {
        const textQuery = queryGroup.querySelector('textarea[name="Text_Query"]');
        if (textQuery && textQuery.style.display !== 'none') {
            const queryImageArea = queryGroup.querySelector('.query-content-area');
            
            const existingAsm = queryGroup.querySelector('textarea[name="Asm_Query"]');
            if (!existingAsm) {
                const asmContainer = document.createElement('div');
                asmContainer.className = 'asm-container';
                
                const newAsmTextarea = document.createElement('textarea');
                newAsmTextarea.name = 'Asm_Query';
                newAsmTextarea.rows = '2';
                newAsmTextarea.placeholder = 'Search asm';
                
                const closeButton = document.createElement('button');
                closeButton.innerHTML = '&times;';
                closeButton.className = 'close-asm-button';
                closeButton.addEventListener('click', function() {
                    asmContainer.remove();
                });
                
                asmContainer.appendChild(newAsmTextarea);
                asmContainer.appendChild(closeButton);
                
                queryImageArea.after(asmContainer);

                // Focus on the newly created ASM textarea
                newAsmTextarea.focus();
            }
        }
    }
}


//---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------

function insertQunNhiuChienTextarea() {
    const activeElement = document.activeElement;
    const queryGroup = activeElement.closest('.query-group');
    
    if (queryGroup) {
        const textQuery = queryGroup.querySelector('textarea[name="Text_Query"]');
        if (textQuery && textQuery.style.display !== 'none') {
            const queryImageArea = queryGroup.querySelector('.query-content-area');
            
            const existingQunNhiuChien = queryGroup.querySelector('textarea[name="QunNhiuChien_Query"]');
            if (!existingQunNhiuChien) {
                const QunNhiuChienContainer = document.createElement('div');
                QunNhiuChienContainer.className = 'QunNhiuChien-container';
                
                const newQunNhiuChienTextarea = document.createElement('textarea');
                newQunNhiuChienTextarea.name = 'QunNhiuChien_Query';
                newQunNhiuChienTextarea.rows = '2';
                newQunNhiuChienTextarea.placeholder = 'Search QunNhiuChien';
                
                const closeButton = document.createElement('button');
                closeButton.innerHTML = '&times;';
                closeButton.className = 'close-QunNhiuChien-button';
                closeButton.addEventListener('click', function() {
                    QunNhiuChienContainer.remove();
                });
                
                QunNhiuChienContainer.appendChild(newQunNhiuChienTextarea);
                QunNhiuChienContainer.appendChild(closeButton);
                
                queryImageArea.after(QunNhiuChienContainer);

                // Focus on the newly created QunNhiuChien textarea
                newQunNhiuChienTextarea.focus();
            }
        }
    }
}


//---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------


// Insert an object filter
function insertObjectFilter() {
    const activeElement = document.activeElement;
    const searchScene = activeElement.closest('.Search_Scene');
    
    if (searchScene) {
        const queryGroup = searchScene.querySelector('.query-group');
        const textQuery = queryGroup.querySelector('textarea[name="Text_Query"]');
        if (textQuery && textQuery.style.display !== 'none') {
            const objectFilterHTML = `
                <div class="object-filter">
                    <label class="object_label">Obj: </label>
                    <input type="text" class="objectInput" list="suggestions">
                    <input type="text" class="valueInput" data-type="text">
                    <button class="close-filter-button">&times;</button>
                </div>
            `;
            
            const lastElement = queryGroup.querySelector('.object-filter:last-of-type') || 
                                queryGroup.querySelector('.ocr-container') || 
                                queryGroup.querySelector('.query-content-area');
            
            lastElement.insertAdjacentHTML('afterend', objectFilterHTML);
            
            const newObjectFilter = queryGroup.querySelector('.object-filter:last-of-type');
            const newObjectInput = newObjectFilter.querySelector('.objectInput');
            const closeButton = newObjectFilter.querySelector('.close-filter-button');
            
            closeButton.addEventListener('click', function() {
                newObjectFilter.remove();
            });
            
            if (newObjectInput) {
                newObjectInput.focus();
            }
        }
    }
}





// Reset the content of the left panel to its original state
function resetLeftPanel(originalLeftPanel) {
    const searchForm = document.getElementById('Search');
    const scenes = searchForm.querySelectorAll('.Search_Scene');

    scenes.forEach(scene => {
        // Remove all divs with class="ocr-container"
        const ocrContainers = scene.querySelectorAll('.ocr-container');
        ocrContainers.forEach(container => container.remove());

        // Remove all divs with class="asm-container"
        const asmContainers = scene.querySelectorAll('.asm-container');
        asmContainers.forEach(container => container.remove());

        // Remove all divs with class="QunNhiuChien-container"
        const QunNhiuChienContainers = scene.querySelectorAll('.QunNhiuChien-container');
        QunNhiuChienContainers.forEach(container => container.remove());
        
        // Remove all divs with class="object-filter"
        const objectFilters = scene.querySelectorAll('.object-filter');
        objectFilters.forEach(filter => filter.remove());

        // Remove image in image-drop-area
        const imageDropAreas = scene.querySelectorAll('.image-drop-area');
        imageDropAreas.forEach(dropArea => {
            const previewContainer = dropArea.querySelector('.preview-upload-container');
            const fileInput = dropArea.querySelector('input[type="file"]');
            clearImage(previewContainer, dropArea.querySelector('.drop-instruction'), fileInput);
        });

        // Reset tab to text
        switchTab(scene, 'text');

        // Reset mode to temporal
        switchMode(scene, 'temporal-search');
    });
}


// Function to set up the search scene tabs
function setupSearchScene(scene) {
    const textButton = scene.querySelector('.text-button');
    const imageButton = scene.querySelector('.image-button');

    textButton.addEventListener('click', (e) => {
        e.preventDefault();
        switchTab(scene, 'text');
    });
    imageButton.addEventListener('click', (e) => {
        e.preventDefault();
        switchTab(scene, 'image');
    });
    
    // Set up mode buttons
    setupModeButtons(scene);
}


//------------------------------------------------------------------------//
// Change tab

// Event listener change tab text and image
document.addEventListener('DOMContentLoaded', function() {
    const searchScenes = document.querySelectorAll('.Search_Scene');

    searchScenes.forEach(scene => {
        const buttons = scene.querySelectorAll('.tab-buttons button');
        const queryContentArea = scene.querySelector('.query-content-area');
        const textQuery = queryContentArea.querySelector('textarea[name="Text_Query"]');
        const imageDropArea = queryContentArea.querySelector('.image-drop-area');

        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = button.className.split('-')[0];
                switchTab(scene, tabName);
            });
        });

        if (imageDropArea) {
            const fileInput = imageDropArea.querySelector('input[type="file"]');
            const previewContainer = imageDropArea.querySelector('.preview-upload-container');
            setupImageUpload(imageDropArea, fileInput, previewContainer);
        }
    });
});


// Switch between tabs
function switchTab(scene, tabName) {
    const buttons = scene.querySelectorAll('.tab-buttons button');
    const queryContentArea = scene.querySelector('.query-content-area');
    const textQuery = queryContentArea.querySelector('textarea[name="Text_Query"]');
    const imageDropArea = queryContentArea.querySelector('.image-drop-area');
    const ocrTextarea = scene.querySelector('textarea[name="Ocr_Query"]');
    const objectFilters = scene.querySelectorAll('.object-filter');

    buttons.forEach(button => button.classList.remove('active'));
    
    // Hide all content areas
    [textQuery, imageDropArea].forEach(el => {
        if (el) el.style.display = 'none';
    });

    // Hide OCR textarea and object filters
    if (ocrTextarea) ocrTextarea.style.display = 'none';
    objectFilters.forEach(filter => {
        if (filter) filter.style.display = 'none';
    });

    switch (tabName) {
        case 'text':
            scene.querySelector('.text-button').classList.add('active');
            if (textQuery) textQuery.style.display = 'block';
            // Show OCR textarea and object filters only for text tab
            if (ocrTextarea) ocrTextarea.style.display = 'block';
            objectFilters.forEach(filter => {
                if (filter) filter.style.display = 'block';
            });
            break;
        case 'image':
            scene.querySelector('.image-button').classList.add('active');
            if (imageDropArea) imageDropArea.style.display = 'flex';
            break;
    }
}


//------------------------------------------------------------------------//

// Function to switch between modes (guarded if mode buttons removed)
function switchMode(scene, modeName) {
    if (!scene) return;
    const modeButtons = scene.querySelectorAll('.mode-button button');
    if (!modeButtons || modeButtons.length === 0) return;
    modeButtons.forEach(button => button.classList.remove('active'));
    
    const activeButton = scene.querySelector(`.${modeName}`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

// Function to set up mode buttons for a search scene (guarded)
function setupModeButtons(scene) {
    if (!scene) return;
    const temporalButton = scene.querySelector('.temporal-search');
    const expansionButton = scene.querySelector('.query-expansion');

    if (!temporalButton || !expansionButton) return;

    temporalButton.addEventListener('click', (e) => {
        e.preventDefault();
        switchMode(scene, 'temporal-search');
    });

    expansionButton.addEventListener('click', (e) => {
        e.preventDefault();
        switchMode(scene, 'query-expansion');
    });

    // Set temporal as active by default
    switchMode(scene, 'temporal-search');
}



//------------------------------------------------------------------------//
// Upload image

// Set up image upload functionality
function setupImageUpload(dropZone, fileInput, previewContainer) {
    const dropInstruction = dropZone.querySelector('.drop-instruction');
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        handleImageUpload(file, previewContainer, dropInstruction);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleImageUpload(file, previewContainer, dropInstruction);
    });
}


// Handle the image upload and display the image
function handleImageUpload(file, previewContainer, dropInstruction) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imgContainer = document.createElement('div');
            imgContainer.className = 'image-preview-container';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            img.id = 'Img-review';
            
            const removeButton = document.createElement('button');
            removeButton.innerHTML = '&times;';
            removeButton.className = 'remove-image-button';
            removeButton.addEventListener('click', (event) => {
                event.stopPropagation(); // Prevent triggering the dropZone click event
                clearImage(previewContainer, dropInstruction, imgContainer.closest('.image-drop-area').querySelector('input[type="file"]'));
            });
            
            imgContainer.appendChild(img);
            imgContainer.appendChild(removeButton);
            
            previewContainer.innerHTML = '';
            previewContainer.appendChild(imgContainer);
            dropInstruction.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

// Clear the image from the preview container and reset the file input
function clearImage(previewContainer, dropInstruction, fileInput) {
    previewContainer.innerHTML = '';
    dropInstruction.style.display = 'block';
    fileInput.value = ''; // Clear the file input value
}






//------------------------------------------------------------------------//
// Add new search scene (Dynamic TRAKE Events)

function addNewSearchScene() {
    const searchForm = document.getElementById('Search');
    if (!searchForm) return;
    const existingScenes = searchForm.querySelectorAll('.Search_Scene');
    const newSceneNumber = existingScenes.length + 1;

    // Determine current task mode for placeholder/title
    const taskSelect = document.getElementById('global-task-select');
    const currentTask = taskSelect ? taskSelect.value : 'trake';
    const titleLabel = currentTask === 'trake' ? `Sự kiện ${newSceneNumber} (Event ${newSceneNumber})` : `Cảnh ${newSceneNumber} (Scene ${newSceneNumber})`;

    // Create a new compact search scene without mode-button
    const newScene = document.createElement('div');
    newScene.className = 'Search_Scene';
    newScene.id = `search-scene-${newSceneNumber}`;
    newScene.innerHTML = `
        <div class="scene-header-bar">
            <div class="scene-header-title">
                <i class="fa fa-step-forward"></i>
                <span id="scene-${newSceneNumber}-title">${titleLabel}</span>
            </div>
            <div class="scene-header-controls">
                <div class="tab-buttons">
                    <button type="button" class="text-button active">
                        <i class="fa fa-font"></i> Text
                    </button>
                    <button type="button" class="image-button">
                        <i class="fa fa-image"></i> Ảnh
                    </button>
                </div>
                <button type="button" class="close-scene-btn" title="Xóa sự kiện này">×</button>
            </div>
        </div>
        <div class="query-group">
            <div class="query-content-area">
                <textarea name="Text_Query" id="Text-Query-${newSceneNumber}" rows="3" placeholder="Nhập mô tả sự kiện ${newSceneNumber}..."></textarea>
                <div class="image-drop-area" style="display: none;">
                    <p class="drop-instruction">Kéo thả ảnh vào đây hoặc click để chọn ảnh</p>
                    <input type="file" accept="image/*" style="display: none;" id="Image-Query-${newSceneNumber}">
                    <div class="preview-upload-container"></div>
                </div>
            </div>
        </div>
    `;

    // Append to form
    searchForm.appendChild(newScene);

    // Setup close button handler
    const closeBtn = newScene.querySelector('.close-scene-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            newScene.remove();
            renumberScenes();
        });
    }

    // Set up tabs
    setupSearchScene(newScene);

    // Set up image upload
    const imageDropArea = newScene.querySelector('.image-drop-area');
    if (imageDropArea) {
        const newFileInput = imageDropArea.querySelector('input[type="file"]');
        const newPreviewContainer = imageDropArea.querySelector('.preview-upload-container');
        setupImageUpload(imageDropArea, newFileInput, newPreviewContainer);
    }

    // Setup enter key press
    const newTextarea = newScene.querySelector('textarea[name="Text_Query"]');
    if (newTextarea && typeof handleEnterKeyPress === 'function') {
        newTextarea.addEventListener('keydown', handleEnterKeyPress);
    }

    if (newTextarea) {
        newTextarea.focus();
    }
}

// Renumber scenes sequentially
function renumberScenes() {
    const scenes = document.querySelectorAll('#Search .Search_Scene');
    const taskSelect = document.getElementById('global-task-select');
    const currentTask = taskSelect ? taskSelect.value : 'trake';

    scenes.forEach((scene, index) => {
        const num = index + 1;
        scene.id = `search-scene-${num}`;
        const titleSpan = scene.querySelector('.scene-header-title span');
        if (titleSpan) {
            titleSpan.id = `scene-${num}-title`;
            if (currentTask === 'trake') {
                titleSpan.textContent = `Sự kiện ${num}`;
            } else if (currentTask === 'kis') {
                titleSpan.textContent = num === 1 ? 'Mô tả KIS' : `Bối cảnh ${num}`;
            } else {
                titleSpan.textContent = num === 1 ? 'Câu hỏi VQA' : `Bối cảnh ${num}`;
            }
        }
        const ta = scene.querySelector('textarea[name="Text_Query"]');
        if (ta) {
            ta.id = num === 1 ? 'Text-Query-First' : (num === 2 ? 'Text-Query-Second' : `Text-Query-${num}`);
        }
    });
}

// Set up event listeners when the document loads
document.addEventListener('DOMContentLoaded', function() {
    // Set up existing search scenes
    const searchScenes = document.querySelectorAll('.Search_Scene');
    searchScenes.forEach(setupSearchScene);

    // Setup pre-existing close scene buttons
    document.querySelectorAll('#Search .Search_Scene .close-scene-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const scene = btn.closest('.Search_Scene');
            if (scene) {
                scene.remove();
                renumberScenes();
            }
        });
    });

    // Wire Add Event button for TRAKE
    const addBtn = document.getElementById('add-scene-button');
    if (addBtn) {
        addBtn.addEventListener('click', function(e) {
            e.preventDefault();
            addNewSearchScene();
        });
    }

    // Wire Clear Scenes button
    const clearBtn = document.getElementById('clear-scenes-button');
    if (clearBtn) {
        clearBtn.addEventListener('click', function(e) {
            e.preventDefault();
            clearAllTextareas();
        });
    }

    // Wire Task Selector Change
    const taskSelect = document.getElementById('global-task-select');
    if (taskSelect) {
        taskSelect.addEventListener('change', function() {
            renumberScenes();
            const firstSceneTa = document.querySelector('#search-scene-1 textarea[name="Text_Query"]');
            if (firstSceneTa) {
                if (this.value === 'kis') {
                    firstSceneTa.placeholder = 'Nhập mô tả cảnh KIS (VD: người phụ nữ mặc áo xanh, xe cứu hỏa)...';
                } else if (this.value === 'trake') {
                    firstSceneTa.placeholder = 'Nhập mô tả Sự kiện 1 (Sự kiện diễn ra đầu tiên trong chuỗi)...';
                } else if (this.value === 'vqa') {
                    firstSceneTa.placeholder = 'Nhập câu hỏi VQA (VD: Có bao nhiêu chiếc xe hơi màu đỏ?...)...';
                }
            }
        });
    }
});

function removeAddedScenes() {
    const searchForm = document.getElementById('Search');
    if (!searchForm) return;
    const scenes = searchForm.querySelectorAll('.Search_Scene');
    scenes.forEach((scene, index) => {
        if (index >= 2) {
            scene.remove();
        }
    });
    renumberScenes();
}