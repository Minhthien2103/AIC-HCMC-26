
let isQuickSearch = false;


// Function to toggle export mode
function toggleQuickSearchMode() {
    const lightingQuickSearchButton = document.getElementById('quick-search');
    isQuickSearch = !isQuickSearch;
    
    if (isQuickSearch) {
        lightingQuickSearchButton.innerHTML = '<img src="src/Img/icon-lighting-yellow.png" alt="icon">';
        lightingQuickSearchButton.title = 'Switch to quick search mode';
        cleanupSearchResults();
    } else {
        lightingQuickSearchButton.innerHTML = '<img src="src/Img/icon-lighting-grey.png" alt="icon">';
        lightingQuickSearchButton.title = 'Switch to normal search mode';
        resetSearch()
    }
}



// Add this to your DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
    const openExportSocketButton = document.getElementById('quick-search');
    openExportSocketButton.addEventListener('click', toggleQuickSearchMode);
});
  
  

//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//



let Pagnitionsocket = null; // WebSocket variable
let currentPage = 0;  // Keep track of the current page
let currentModelType = 'clip'; // Default model type
let currentModeType = 'search'; // Default mode type

// Define observer to detect the last image in the list
const batchObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && entry.target.id.startsWith('page-end-')) {
            currentPage++;
            requestNextBatch(currentPage);
            observer.unobserve(entry.target);  // Unobserve to prevent multiple triggers
        }
    });
}, {
    rootMargin: '200px'
});
// Establish WebSocket connection
function connectPagnitionWebSocket() {
    const wsUrl = (window.CONFIG && window.CONFIG.BACKEND_WS) || window.BACKEND_WS || "ws://localhost:8006";
    Pagnitionsocket = new WebSocket(`${wsUrl}/ws/pagnition`);

    Pagnitionsocket.onopen = function(event) {
        console.log("WebSocket connection established");
    };

    Pagnitionsocket.onmessage = function(event) {
        data= JSON.parse(event.data);

        // Check if results are returned
        if (data.kq && data.kq.length > 0) {

            // Update UI with the new batch of results
            updateUIWithPagnition(data.kq, data.page);
        } else {
            console.log("No more results");
        }
    };

    Pagnitionsocket.onclose = function(event) {
        console.log("WebSocket connection closed", event);
    };

    Pagnitionsocket.onerror = function(error) {
        console.error("WebSocket error", error);
    };
}

// Request the next batch from the backend
function requestNextBatch(page) {
    if (Pagnitionsocket.readyState === WebSocket.OPEN) {
        let message = {
            type: 'pagination_query',
            model: currentModelType,  // Use the current model
            mode: currentModeType,    // Use the current mode
            page: page                // Send the current page number
        };
        
        Pagnitionsocket.send(JSON.stringify(message));  // Send request for the next batch
    } else {
        console.error('WebSocket is not open. ReadyState:', Pagnitionsocket.readyState);
        // Optionally handle reconnection here
        connectWebSocket();
    }
}

// Function to update UI and trigger the observer on the last item
function updateUIWithPagnition(results, page) {
    // Call updateRightPanel_list to update the UI with the new batch of results
    const updatedDivs = updatePagnitionRightPanel_list(results, page);

    // Observe the last image for infinite scrolling
    const lastDiv = updatedDivs[updatedDivs.length - 1];
    if (lastDiv) {
        lastDiv.id = `page-end-${page}`;  // Mark the last item in the batch
        batchObserver.observe(lastDiv);  // Observe the last item for infinite scrolling
    }
}

// Start WebSocket connection on page load
window.onload = function() {
    connectPagnitionWebSocket();
};


function createPagnitionImageDiv(result, index) {
    const frameInfo = `${result.entity.video}-${result.entity.time}`;
    const imgUrl = typeof resolveResultImageUrl === 'function' ? resolveResultImageUrl(result) : '';
    
    const div = document.createElement('div');
    div.className = 'img-dis';
    div.innerHTML = `
        <img alt="" class="result" loading="lazy" height="100px" id="${index}" data-src="${imgUrl}" src="${imgUrl}">
        <div class="infor">${frameInfo}</div>
        <div name="similarity_search" class="similarity_search"></div>
        <div class="export_icon"></div>
    `;

    const img = div.querySelector('img');
    img.setAttribute('draggable', 'true');
    img.addEventListener('dragstart', drag);


    // Add middle click event listener
    div.addEventListener('mousedown', (event) => {
        if (event.button === 1) { // Middle mouse button
            event.preventDefault(); // Prevent default middle-click behavior
            const frameId = result.entity.frame_id;
            const imagePath = img.dataset.src;
            addImageToExportArea(frameId, imagePath, frameInfo);
        }
    });

    // Using unified event listeners
    const exportIcon = div.querySelector('.export_icon');
    exportIcon.addEventListener('click', () => {
        const imagePath = img.dataset.src;
        const frameId = result.entity.frame_id;
        addImageToExportArea(frameId, imagePath, frameInfo);
    });

    

    return div;
}

function updatePagnitionRightPanel_list(results, page) {
    const listPhoto = document.getElementById("list-photo");
    const fragment = document.createDocumentFragment();

    const startingIndex = listPhoto.children.length;

    const updatedDivs = results.map((result, index) => {
        const globalIndex = startingIndex + index;
        const div = createPagnitionImageDiv(result, globalIndex + 1);
        fragment.appendChild(div);

        // Observe the image for lazy-loading
        const img = div.querySelector('img');
        imageObserver.observe(img);

        return div;
    });

    listPhoto.appendChild(fragment);

    return Array.from(listPhoto.children);
}

function resetSearch() {
    const listPhoto = document.getElementById("list-photo");

    // Clear all existing results
    while (listPhoto.firstChild) {
        listPhoto.removeChild(listPhoto.firstChild);
    }

    // Reset pagination state
    currentPage = 0;
}
