
async function performCombinedSearch() {
    const searchScenes = document.querySelectorAll('.Search_Scene');
    
    // Initialize arrays to hold queries for each search type
    const temporalQueries = [];
    const expansionQueries = [];

    for (const scene of searchScenes) {
        const query = await getQueryContent(scene);
        
        // Determine the search type for the current scene
        const isTemporalActive = scene.querySelector('.temporal-search').classList.contains('active');
        const searchType = isTemporalActive ? 'temporal' : 'expansion';

        // Add the query to the appropriate group based on search type
        if (searchType === 'temporal') {
            temporalQueries.push({
                type: query.type,
                content: query.type === 'image' ? query.content.split(',')[1] : query.content
            });
        } else {
            expansionQueries.push({
                type: query.type,
                content: query.type === 'image' ? query.content.split(',')[1] : query.content
            });
        }
    }

    if (socket.readyState === WebSocket.OPEN) {
        // Create the final message object with grouped queries
        let message = {
            type: 'multi_query',
            temporal: temporalQueries,   // Group for temporal queries
            expansion: expansionQueries  // Group for expansion queries
        };

        requestTime = performance.now();
        socket.send(JSON.stringify(message));
        console.log("Grouped queries sent at:", new Date().toISOString());
    } else {
        console.error('WebSocket is not open. ReadyState:', socket.readyState);
        // Attempt to reconnect
        connectWebSocket();
    }
}
async function getQueryContent(searchScene) {
    const textArea = searchScene.querySelector('textarea[name="Text_Query"]');
    const imageDropArea = searchScene.querySelector('.image-drop-area');
    
    if (textArea.style.display !== 'none') { 
        const originalText = textArea.value;
        if (document.getElementById('translate-checkbox').checked){
            const translatedText = await translateText(originalText);
            textArea.value = translatedText;
            return { type: 'text', content: translatedText };
        }
        else
            return { type: 'text', content: originalText };
    } else if (imageDropArea.style.display === 'flex') {
        const img = imageDropArea.querySelector('img');
        return img ? { type: 'image', content: img.src } : { type: 'text', content: '' };
    }
    return { type: 'text', content: '' };
}
