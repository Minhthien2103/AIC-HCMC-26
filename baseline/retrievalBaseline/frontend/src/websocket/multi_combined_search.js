async function performCombinedSearch() {
    const searchScenes = document.querySelectorAll('.Search_Scene');
    const queries = [];

    // Determine model (defaults to clip)
    const activeModelButton = document.querySelector('.model-option button.active');
    const modelType = activeModelButton ? activeModelButton.className.split(' ')[0] : 'clip';

    for (const scene of searchScenes) {
        // Determine mode (defaults to temporal-search)
        const activeModeButton = scene.querySelector('.mode-button button.active');
        const modeType = activeModeButton ? activeModeButton.className.split(' ')[0] : 'temporal-search';

        const query = await getQueryContent(scene);
        if (query.content) {  // Only add non-empty queries
            queries.push({
                ...query,
                mode: modeType // Include mode type for each search scene
            });
        }
    }

    // Check WebSocket status and send the query if open
    if (socket.readyState === WebSocket.OPEN) {
        let message = {
            type: 'multi_query',
            model: modelType,  // Include the active model type in the message
            queries: queries.map(q => ({
                type: q.type,
                content: q.type === 'image' ? q.content.split(',')[1] : q.content,
                mode: q.mode,  // Include the mode for each search scene in the message
                detail: q.detail
            }))
        };

        requestTime = performance.now();
        // console.log(message)
        socket.send(JSON.stringify(message));
        console.log("Query sent at:", new Date().toISOString());
    } else {
        console.error('WebSocket is not open. ReadyState:', socket.readyState);
        // Attempt to reconnect
        connectWebSocket();
    }
}




// This function retrieves the query content from a search scene.
// Depending on whether it's a text, image, or sound query, it fetches the appropriate data.
// It also handles translation if the 'translate-checkbox' is checked for text queries.

async function getQueryContent(searchScene) {
    const textArea = searchScene.querySelector('textarea[name="Text_Query"]');
    const imageDropArea = searchScene.querySelector('.image-drop-area');
    const soundTextArea = searchScene.querySelector('textarea[name="Sound_Text"]');
    let detailsTextArea = "";
    if (searchScene.querySelector('textarea[name="QunNhiuChien_Query"]') !== null) {
        detailsTextArea = searchScene.querySelector('textarea[name="QunNhiuChien_Query"]').value;
    }

    if (textArea && textArea.style.display !== 'none') { 
        return { type: 'text', content: textArea.value, detail: detailsTextArea };
    } else if (imageDropArea && imageDropArea.style.display === 'flex') {
        const img = imageDropArea.querySelector('img');
        if (img) {
            return { type: 'image', content: img.src, detail: detailsTextArea };
        }
    } else if (soundTextArea && soundTextArea.style.display !== 'none') {
        return { type: 'sound', content: soundTextArea.value };
    }
    return { type: 'text', content: '', detail: detailsTextArea };
}   




//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//


// This function performs a paginated combined search similar to performCombinedSearch.
// It resets the search before gathering queries and sending them through a WebSocket (Pagnitionsocket).

async function performPagnitionCombinedSearch() {
    resetSearch();
    const searchScenes = document.querySelectorAll('.Search_Scene');
    const queries = [];

    // Determine model (defaults to clip)
    const activeModelButton = document.querySelector('.model-option button.active');
    const modelType = activeModelButton ? activeModelButton.className.split(' ')[0] : 'clip';

    // Determine mode (defaults to temporal-search)
    const activeModeButton = document.querySelector('.mode-button button.active');
    const modeType = activeModeButton ? activeModeButton.className.split(' ')[0] : 'temporal-search';

    for (const scene of searchScenes) {
        const query = await getQueryContent(scene);
        if (query.content) {  // Only add non-empty queries
            queries.push(query);
        }
    }

    // Check WebSocket status and send the query if open
    if (Pagnitionsocket.readyState === WebSocket.OPEN) {
        let message = {
            type: 'multi_query',
            model: modelType,  // Include the active model type in the message
            mode: modeType,    // Include the active mode type in the message
            queries: queries.map(q => ({
                type: q.type,
                content: q.type === 'image' ? q.content.split(',')[1] : q.content
            })),
        };

        requestTime = performance.now();
        Pagnitionsocket.send(JSON.stringify(message));
        console.log("Query sent at:", new Date().toISOString());
    } else {
        console.error('WebSocket is not open. ReadyState:', socket.readyState);
        // Attempt to reconnect
        connectWebSocket();
    }
}