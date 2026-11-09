//------------------------ Short cut ---------

document.addEventListener('DOMContentLoaded', function() {
    const originalLeftPanel = document.querySelector('.left-panel').cloneNode(true);

    // Enter: trigger the search button
    document.addEventListener('keydown', async function(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            const activeElement = document.activeElement;
            const searchScene = activeElement.closest('.Search_Scene');
    
            console.log("huhuhuhu");
            if (searchScene) {
                const isTextInput = activeElement.matches('textarea[name="Text_Query"]') ||
                                    activeElement.matches('textarea[name="Ocr_Query"]') ||
                                    activeElement.matches('textarea[name="Asm_Query"]') ||
                                    activeElement.matches('textarea[name="QunNhiuChien_Query"]') ||
                                    activeElement.closest('.object-filter');
                
                const isImageTab = searchScene.querySelector('.image-button').classList.contains('active');
    
                if (isTextInput || isImageTab) {
                    event.preventDefault(); // Prevent default Enter behavior
                    
                    
                    // Collect text queries before performing search
                    collectTextQueries();


                    // Get all search scenes
                    const searchScenes = document.querySelectorAll('.Search_Scene');
                    let filledQueriesCount = 0;
    
                    // Check how many search groups have content
                    for (const scene of searchScenes) {
                        const query = await getQueryContent(scene);
                        if (query.content) {
                            filledQueriesCount++;
                        }
                    }
    
                    toggleLoadingIndicator(true);
    
                    // Perform search based on how many search scenes have content
                    if (isQuickSearch) {
                        // Quick search mode
                        // Perform search based on how many search scenes have content
                        if (filledQueriesCount === 1) {
                            await performPagnitionCombinedSearch();
                        } else if (filledQueriesCount > 1) {
                            await performCombinedSearch();
                        }
                    } else {
                        // Normal search mode
                        await performCombinedSearch();
                    }
    
                    toggleLoadingIndicator(false);  // Hide loading indicator when done
                }
            }
        }
    });
    

    // Shift + Enter: trigger the filter button
    document.addEventListener('keydown', function(event) {
        if (event.shiftKey && event.key === 'Enter') {
            const activeElement = document.activeElement;
            const isRelevantInput = 
                activeElement.matches('textarea[name="Text_Query"]') ||
                activeElement.matches('textarea[name="Ocr_Query"]') ||
                activeElement.matches('textarea[name="Asm_Query"]') ||
                activeElement.matches('textarea[name="QunNhiuChien_Query"]') ||
                activeElement.closest('.object-filter');
    
            if (isRelevantInput) {
                event.preventDefault();
                handleFilterAction();
            }
        }
    });

    // Add event listener for the search button
    const searchButton = document.getElementById('search-button');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            toggleLoadingIndicator(true);

            // Collect text queries before performing search
            collectTextQueries();
            
            performCombinedSearch();
        });
    }

    // Alt + w: Toggle switch view
    const toggleSwitch = document.getElementById('mode-toggle');
    toggleSwitch.addEventListener('change', togglePanelLayout);
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 'w') {
            event.preventDefault();
            toggleSwitch.checked = !toggleSwitch.checked;
            togglePanelLayout.call(toggleSwitch);
        }
    });

    // Alt + e: Toggle translate
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 'e') {
            event.preventDefault();
            const translateOptionCheckbox = document.querySelector('.translate-option .toggle-checkbox');
            if (translateOptionCheckbox) {
                translateOptionCheckbox.checked = !translateOptionCheckbox.checked;
            }
        }
    });

    // Alt + d: Toggle object list
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 'd') {
            event.preventDefault();
            const objectListButton = document.getElementById('show-object-list');
            if (objectListButton) {
                objectListButton.click(); // Simulate button click to toggle the list
            }
        }
    });

    // Ctrl + i: Add search OCR textarea
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'i') {
            event.preventDefault();
            insertOcrTextarea();
        }
    });

    // Ctrl + k: Add search ASM textarea
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'k') {
            event.preventDefault();
            insertAsmTextarea();
        }
    });

    // Ctrl + l: Add search QunNhiuChien textarea
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'l') {
            event.preventDefault();
            insertQunNhiuChienTextarea();
        }
    });

    // Ctrl + j: Add search object element
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'j') {
            event.preventDefault();
            insertObjectFilter();
        }
    });

    // Ctrl + h: Add a new search scene
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'h') {
            event.preventDefault();
            addNewSearchScene();
        }
    });

    // Ctrl + q: Delete search OCR and search object element
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'q') {
            event.preventDefault();
            resetLeftPanel(originalLeftPanel);
            clearAllTextareas();
            removeAddedScenes();
            focusOnFirstTextbox();
        }
    });
    
    // Event listener for keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Slash (/): Focus on the first textbox in search-scene-1
        if (event.key === '/' && !event.shiftKey) {
            event.preventDefault();
            focusOnFirstTextbox();
        }
        
        // Shift + Slash (?): Cycle through Text_Query textboxes
        if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
            event.preventDefault();
            cycleThroughTextboxes();
        }
    });

    // Ctrl + e: Clear all textareas
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'e') {
            event.preventDefault();
            clearAllTextareas();
        }
    });

    // Alt + r: Switch between text and image tabs in search-scene-1
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 'r') {
            event.preventDefault();
            const scene1 = document.getElementById('search-scene-1');
            const textButton = scene1.querySelector('.text-button');
            const imageButton = scene1.querySelector('.image-button');
            const isTextActive = textButton.classList.contains('active');
            switchTab(scene1, isTextActive ? 'image' : 'text');
        }
    });

    // Alt + t: Switch between text and image tabs in search-scene-2
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 't') {
            event.preventDefault();
            const scene2 = document.getElementById('search-scene-2');
            const textButton = scene2.querySelector('.text-button');
            const imageButton = scene2.querySelector('.image-button');
            const isTextActive = textButton.classList.contains('active');
            switchTab(scene2, isTextActive ? 'image' : 'text');
        }
    });

    // Alt + a: Toggle export area
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 'a') {
          event.preventDefault();
          toggleExportArea();
        }
    });

    // Alt + x: Press reset-export-button to delete all images in export area
    document.addEventListener('keydown', function(event) {
        if (event.altKey && event.key === 's') {
            event.preventDefault();
            document.getElementById('reset-export').click();
        }
    });

    
    // Ctrl + s: Trigger submit button
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 's') {
            event.preventDefault();
            document.getElementById('submit-button').click();
        }
    });
});

