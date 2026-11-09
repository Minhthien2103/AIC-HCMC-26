let socket;
let requestTime;

function getBackendWs() {
    return (window.CONFIG && window.CONFIG.BACKEND_WS) || window.BACKEND_WS || "ws://localhost:8006";
}

// Function to connect to the main WebSocket
function connectWebSocket() {
    socket = new WebSocket(`${getBackendWs()}/ws`);
    
    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    // Event handler for receiving messages from the WebSocket
    socket.onmessage = (event) => {
        const receiveTime = performance.now();
        console.log(`Data received from backend. Time elapsed: ${receiveTime - requestTime} ms`);
        
        try {
            data = JSON.parse(event.data);
            if (data.kq) {
                // Update the UI with search results
                updateUIWithSearchResults(data.kq);
                const updateCompleteTime = performance.now();
                console.log(`UI update completed. Total time: ${updateCompleteTime - requestTime} ms`);
                console.log(`---------------------------------------------------------------------`);
                toggleLoadingIndicator(false);
            } else {
                console.error("Received data does not contain 'kq' property:", data);
            }
        } catch (error) {
            console.error("Error parsing received data:", error);
        }
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed');
        // Attempt to reconnect
        setTimeout(connectWebSocket, 5000);
    };
}

// Connect the WebSocket when the page loads
// Make sure to call these connection functions when the page loads
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    connectWebSocketcrossing();
});

let socket_share;

function sendVqaInputUpdate(frameId, vqaInput) {
    if (socket_share && socket_share.readyState === WebSocket.OPEN) {
        const message = JSON.stringify({
            type: 'vqa_input_update',
            frameId: frameId,
            vqaInput: vqaInput
        });
        socket_share.send(message);
    } else {
        console.error('WebSocket is not open. Unable to send VQA input update.');
    }
}

function connectWebSocketcrossing() {
  socket_share = new WebSocket(`${getBackendWs()}/ws/share_image`);
  
  socket_share.onopen = () => {
    console.log('Share WebSocket connection established');
  };

  socket_share.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'image_share') {
        if (data.frameId && data.src && data.frameInfo) {
          addImageToExportArea(data.frameId, data.src, data.frameInfo, false);
        } else {
          console.error("Received image data does not contain expected properties:", data);
        }
      } else if (data.type === 'vqa_input_update') {
        if (data.frameId && data.vqaInput !== undefined) {
          updateVqaInput(data.frameId, data.vqaInput);
        } else {
          console.error("Received VQA input update does not contain expected properties:", data);
        }
      }
    } catch (error) {
      console.error("Error parsing received data:", error);
    }
  };

  socket_share.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  socket_share.onclose = () => {
    console.log('WebSocket connection closed');
    setTimeout(connectWebSocketcrossing, 5000);
  };
}




///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///

// WebSocket for similarity search
let similaritySearchSocket;

// Connect to the WebSocket for similarity search
function connectSimilaritySearchWebSocket() {
    similaritySearchSocket = new WebSocket(`${getBackendWs()}/ws/similarity_search`);
    
    similaritySearchSocket.onopen = () => {
        console.log('Similarity Search WebSocket connection established');
    };

    // Event handler for receiving similarity search results
    similaritySearchSocket.onmessage = (event) => {
        const receiveTime = performance.now();
        console.log(`Similarity Search data received from backend. Time elapsed: ${receiveTime - requestTime} ms`);
        
        try {
            data = JSON.parse(event.data);
            if (data.kq) {
                // Update the UI with search results
                updateUIWithSearchResults(data.kq);
                const updateCompleteTime = performance.now();
                console.log(`UI update completed. Total time: ${updateCompleteTime - requestTime} ms`);
                toggleLoadingIndicator(false);
            } else if (data.error) {
                console.error("Error from server:", data.error);
                toggleLoadingIndicator(false);
            } else {
                console.error("Received data does not contain 'kq' property:", data);
                toggleLoadingIndicator(false);
            }
        } catch (error) {
            console.error("Error parsing received data:", error);
            toggleLoadingIndicator(false);
        }
    };

    similaritySearchSocket.onerror = (error) => {
        console.error('Similarity Search WebSocket error:', error);
        toggleLoadingIndicator(false);
    };

    similaritySearchSocket.onclose = () => {
        console.log('Similarity Search WebSocket connection closed');
        // Attempt to reconnect
        setTimeout(connectSimilaritySearchWebSocket, 5000);
    };
}

// Function to perform similarity search
function performSimilaritySearch(vectorId) {
    if (similaritySearchSocket.readyState === WebSocket.OPEN) {
        requestTime = performance.now();
        toggleLoadingIndicator(true);
        similaritySearchSocket.send(JSON.stringify({ vector: vectorId }));
    } else {
        console.error('Similarity Search WebSocket is not open');
        toggleLoadingIndicator(false);
    }
}




///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///

// WebSocket for filter queries
let filterSocket;

// Connect to the WebSocket for filtering queries
function connectFilterWebSocket() {
    filterSocket = new WebSocket(`${getBackendWs()}/ws/filter_query`);
    
    filterSocket.onopen = () => {
        console.log('Filter WebSocket connection established');
    };

    filterSocket.onmessage = (event) => {
        const receiveTime = performance.now();
        console.log(`Filter data received from backend. Time elapsed: ${receiveTime - requestTime} ms`);
        
        try {
            data = JSON.parse(event.data);
            if (data.kq) {
                updateUIWithSearchResults(data.kq);
                const updateCompleteTime = performance.now();
                console.log(`UI update completed. Total time: ${updateCompleteTime - requestTime} ms.`);
                toggleLoadingIndicator(false);
            } else if (data.error) {
                console.error("Error from server:", data.error);
            } else {
                console.error("Received data does not contain 'kq' property:", data);
            }
        } catch (error) {
            console.error("Error parsing received data:", error);
        }
    };

    filterSocket.onerror = (error) => {
        console.error('Filter WebSocket error:', error);
    };

    filterSocket.onclose = () => {
        console.log('Filter WebSocket connection closed');
        // Attempt to reconnect
        setTimeout(connectFilterWebSocket, 5000);
    };
}


///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///



let logSocket;

function connectLogWebSocket() {
  logSocket = new WebSocket(`${getBackendWs()}/ws/log`);
  
  logSocket.onopen = function(e) {
    console.log("[open] Log WebSocket connection established");
  };

  logSocket.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log(`[message] Log server says: ${response.message}`);
  };

  logSocket.onclose = function(event) {
    if (event.wasClean) {
      console.log(`[close] Log WebSocket connection closed cleanly, code=${event.code} reason=${event.reason}`);
    } else {
      console.log('[close] Log WebSocket connection died');
    }
  };

  logSocket.onerror = function(error) {
    console.log(`[error] ${error.message}`);
  };
}

// Call this function when your app initializes
connectLogWebSocket();




///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///

// send message

let querySocket;

function connectQueryWebSocket() {
    querySocket = new WebSocket(`${getBackendWs()}/ws/share_query`);
    
    querySocket.onopen = () => {
        console.log('Query WebSocket connection established');
    };

    querySocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'shared_query') {
            addToSharedQueries(data.query);
        }
    };

    querySocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    querySocket.onclose = () => {
        console.log('WebSocket connection closed');
        setTimeout(connectQueryWebSocket, 5000);
    };

}



///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///



let groupSearchSocket;

// Function to connect to the group search WebSocket
function connectGroupSearchWebSocket() {
    groupSearchSocket = new WebSocket(`${getBackendWs()}/ws/group_search`);
    
    groupSearchSocket.onopen = () => {
        console.log('Group Search WebSocket connection established');
    };

    groupSearchSocket.onmessage = (event) => {
        const receiveTime = performance.now();
        console.log(`Group Search data received from backend. Time elapsed: ${receiveTime - requestTime} ms`);
        
        try {
            data = JSON.parse(event.data);
            if (data.kq) {
                // Update the UI with group search results
                updateUIWithSearchResults(data.kq);
                const updateCompleteTime = performance.now();
                console.log(`UI update completed. Total time: ${updateCompleteTime - requestTime} ms`);
                toggleLoadingIndicator(false);
            } else if (data.error) {
                console.error("Error from server:", data.error);
                toggleLoadingIndicator(false);
            } else {
                console.error("Received data does not contain 'results' property:", data);
                toggleLoadingIndicator(false);
            }
        } catch (error) {
            console.error("Error parsing received data:", error);
            toggleLoadingIndicator(false);
        }
    };

    groupSearchSocket.onerror = (error) => {
        console.error('Group Search WebSocket error:', error);
        toggleLoadingIndicator(false);
    };

    groupSearchSocket.onclose = () => {
        console.log('Group Search WebSocket connection closed');
        // Attempt to reconnect
        setTimeout(connectGroupSearchWebSocket, 5000);
    };
}

connectGroupSearchWebSocket()





///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///



let alertSocket;

function connectAlertWebSocket() {
    alertSocket = new WebSocket(`${getBackendWs()}/ws/alerts`);
    
    alertSocket.onopen = () => {
        console.log('Alert WebSocket connection established');
    };

    alertSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'alert' && data.message) {
                showTemporaryAlert(data.message);
            }
        } catch (error) {
            console.error("Error parsing alert data:", error);
        }
    };

    alertSocket.onerror = (error) => {
        console.error('Alert WebSocket error:', error);
    };

    alertSocket.onclose = () => {
        console.log('Alert WebSocket connection closed');
        setTimeout(connectAlertWebSocket, 5000);
    };
}

function sendAlertViaWebSocket(message) {
    if (alertSocket && alertSocket.readyState === WebSocket.OPEN) {
        const alertData = {
            type: 'alert',
            message: message
        };
        alertSocket.send(JSON.stringify(alertData));
    } else {
        console.error('Alert WebSocket is not open. Unable to send alert.');
    }
}




///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///
