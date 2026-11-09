// Add an event listener for the 'keydown' event to listen for the Enter key.
// When the Enter key is pressed, prevent the default behavior, show the loading indicator,
// perform the combined search, and then hide the loading indicator.
// document.addEventListener('keydown', function(event) {
//     if (event.key === "Enter") {
//         event.preventDefault(); // Prevent default Enter behavior
//         toggleLoadingIndicator(true)
//         performCombinedSearch();
//         toggleLoadingIndicator(false)
//     }
// });


// Perform a combined search based on the content of two search scenes.
// Determine the type of queries (text or image) and perform the appropriate search.
async function performCombinedSearch() {
    const firstSearchScene = document.getElementById('search-scene-1');
    const secondSearchScene = document.getElementById('search-scene-2');

    const firstQuery = getQueryContent(firstSearchScene);
    const secondQuery = getQueryContent(secondSearchScene);
    // console.log(firstQuery)
    // console.log(secondQuery)
    if (firstQuery.type === 'image' || secondQuery.type === 'image') {
        await performMixedSearch(firstQuery, secondQuery);
    } else {
        performSearch(firstQuery.content, secondQuery.content);
    }
    // console.log('PerformCombinedSearch');
}


// Get the content of a search scene. Determine whether the query is text or image,
// and return the query type and content.
function getQueryContent(searchScene) {
    const textArea = searchScene.querySelector('textarea[name="Text_Query"]');
    const imageDropArea = searchScene.querySelector('.image-drop-area');
    // console.log('getQueryContent');
    
    if (textArea.style.display !== 'none') {
        return { type: 'text', content: textArea.value };
    } else if (imageDropArea.style.display === 'flex') {
        const img = imageDropArea.querySelector('img');
        return img ? { type: 'image', content: img.src } : { type: 'text', content: '' };
    }
    return { type: 'text', content: '' };
}


// Perform a mixed search when one or both queries are images.
// Send the appropriate data to the backend based on the query types.
async function performMixedSearch(firstQuery, secondQuery) {
    let firstImageData = null;
    let secondImageData = null;
    let firstTextQuery = '';
    let secondTextQuery = '';
    
    if (firstQuery.type === 'image') {
        firstImageData= firstQuery.content.split(',')[1]
    } else {
        firstTextQuery = firstQuery.content;
    }

    if (secondQuery.type === 'image') {
        secondImageData= secondQuery.content.split(',')[1]        
    } else {
        secondTextQuery = secondQuery.content;
    }
    

    // Now handle all possible combinations
    if (firstImageData && secondImageData) {
        // Image-Image query
        await sendDoubleImageQueryToBackend(firstImageData, secondImageData);
    } else if (firstImageData || secondImageData) {
        // Mixed Image-Text query
        await sendMixedQueryToBackend(firstImageData || secondImageData, firstTextQuery || secondTextQuery);
    } else {
        // Text-Text query (though this should be handled by performSearch in the parent function)
        performSearch(firstTextQuery, secondTextQuery);
    }
}


// Send a mixed query (image and text) to the backend.
// Handle the response and update the UI with the search results.
async function sendMixedQueryToBackend(imageData, textQuery) {
    try {
        console.log('go in sendMixedQuerytoBackend');
        const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
        const response = await fetch(`${backendBase}/mixed_query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                image: imageData,
                text: textQuery
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        data = await response.json();
        updateUIWithSearchResults(data.kq);
    } catch (error) {
        console.error('Error performing mixed search:', error);
    }
}


// Send a double image query to the backend.
// Handle the response and update the UI with the search results.
async function sendDoubleImageQueryToBackend(firstImageData, secondImageData) {
    try {
        console.log('go in send DoubleImageQueryToBackEnd');
        const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
        const response = await fetch(`${backendBase}/double_image_query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                firstImage: firstImageData,
                secondImage: secondImageData
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        data = await response.json();
        updateUIWithSearchResults(data.kq);
    } catch (error) {
        console.error('Error performing double image search:', error);
    }
}
