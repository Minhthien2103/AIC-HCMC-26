
//------------------------ Queries history ------------------------//


// Helper function to collect all text queries
function collectTextQueries() {
    const textareas = document.querySelectorAll('textarea[name="Text_Query"]');
    textareas.forEach(textarea => {
        if (textarea.value.trim()) {
            addUniqueQueryToHistory(textarea.value);
        }
    });
}

// Helper function to create copy button
function createCopyButton() {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
        </svg>
    `;
    button.title = "Copy to clipboard";
    return button;
}


// Function to add copy button to query item
function addCopyButtonToQueryItem(queryItem) {
    const copyButton = createCopyButton();
    copyButton.addEventListener('click', (e) => {
        e.stopPropagation();
        const textToCopy = queryItem.textContent;
        navigator.clipboard.writeText(textToCopy).then(() => {
            // Optional: Visual feedback
            copyButton.style.color = '#00ff00';
            setTimeout(() => {
                copyButton.style.color = '';
            }, 1000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    });
    queryItem.appendChild(copyButton);
}


// Helper function to add unique query to history
function addUniqueQueryToHistory(query) {
    const queriesContainer = document.querySelector('#queries-history .query-container');
    if (!query || query.trim() === '' || !queriesContainer) return;
    
    const trimmedQuery = query.trim();
    const existingQueries = Array.from(queriesContainer.children).map(item => 
        item.childNodes[0].textContent.trim()); // Updated to get text excluding button
    
    if (!existingQueries.includes(trimmedQuery)) {
        const queryItem = document.createElement('div');
        queryItem.className = 'query-item';
        
        // Create a text node for the query
        const textNode = document.createTextNode(trimmedQuery);
        queryItem.appendChild(textNode);
        
        // Add copy button
        addCopyButtonToQueryItem(queryItem);
        
        queriesContainer.prepend(queryItem);
    }
}


// Add shared query to shared queries section
function addToSharedQueries(query) {
    const sharedContainer = document.querySelector('#shared-queries .query-container');
    if (!sharedContainer) return;

    const queryItem = document.createElement('div');
    queryItem.className = 'query-item shared';
    
    // Create a text node for the query
    const textNode = document.createTextNode(query);
    queryItem.appendChild(textNode);
    
    // Add copy button
    addCopyButtonToQueryItem(queryItem);
    
    sharedContainer.prepend(queryItem);
}


// Helper function to handle sharing query
function handleShareQuery() {
    const shareInput = document.getElementById('share-input');
    const sharedQuery = shareInput.value.trim();
    if (sharedQuery !== '') {
        const queryData = {
            type: 'shared_query',
            query: sharedQuery
        };
        querySocket.send(JSON.stringify(queryData));
        shareInput.value = '';
        adjustShareInputHeight(shareInput);
    }
}

// Helper function to set up clear buttons
function setupClearButtons() {
    // Handle clear buttons
    const clearHistoryButton = document.getElementById('clear-history');
    const clearSharedButton = document.getElementById('clear-shared');

    if (clearHistoryButton) {
        clearHistoryButton.addEventListener('click', () => {
            const historyContainer = document.querySelector('#queries-history .query-container');
            if (historyContainer) {
                historyContainer.innerHTML = '';
            }
        });
    }

    if (clearSharedButton) {
        clearSharedButton.addEventListener('click', () => {
            const sharedContainer = document.querySelector('#shared-queries .query-container');
            if (sharedContainer) {
                sharedContainer.innerHTML = '';
            }
        });
    }
}


// Auto-resize share input function
function adjustShareInputHeight(shareInput) {
    shareInput.style.height = 'auto';
    shareInput.style.height = (shareInput.scrollHeight) + 'px';
}



// Helper function to set up share functionality
function setupShareFunctionality() {
    const shareButton = document.getElementById('share-button');
    const shareInput = document.getElementById('share-input');
    
    if (shareButton) {
        shareButton.addEventListener('click', handleShareQuery);
    }
    
    if (shareButton && shareInput) {
        // Add input event listener for auto-resize
        shareInput.addEventListener('input', () => adjustShareInputHeight(shareInput));
        
        // Prevent default Enter behavior in share input
        shareInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                shareButton.click();
            }
        });
        
        shareButton.addEventListener('click', handleShareQuery);
    }
}



document.addEventListener('DOMContentLoaded', () => {
    const queriesIcon = document.getElementById('queries-icon');
    
    // Initialize WebSocket connection
    connectQueryWebSocket();
    
    // Set up event listeners
    queriesIcon.addEventListener('click', toggleQueriesPopup);

    // Set up share functionality
    setupShareFunctionality();

    // Set up clear buttons
    setupClearButtons();
    
    // Add copy buttons to existing query items
    document.querySelectorAll('.query-item').forEach(addCopyButtonToQueryItem);
});



function toggleQueriesPopup() {
    const queriesIcon = document.getElementById('queries-icon');
    const queriesPopup = document.getElementById('queries-popup');
    
    queriesIcon.classList.toggle('active');
    queriesPopup.classList.toggle('active');    
}