//------------------------ Export ------------------------//

// Toggle the visibility of the export area and adjust the width of the content wrapper accordingly.
function toggleExportArea() {
  const exportArea = document.getElementById('export-area');
  const contentWrapper = document.querySelector('.content-wrapper');
  
  exportArea.classList.toggle('show');
  contentWrapper.classList.toggle('export-active');
}

// Initialize event listeners after the DOM content is loaded.
// Add a click event listener to the export button to toggle the export area.
// Handle window resize events to adjust the width of the content wrapper.
document.addEventListener('DOMContentLoaded', function() {
  const exportButton = document.getElementById('export-button');
  exportButton.addEventListener('click', toggleExportArea);
  
  window.addEventListener('resize', function() {
      const exportArea = document.getElementById('export-area');
      const contentWrapper = document.querySelector('.content-wrapper');
      
      if (exportArea.classList.contains('show')) {
          contentWrapper.style.width = '80%';
      }
  });
});

//---------------------------------------------------------------------------------------------//

let activeTask = 'kis';
let vqaInputs = {};

// Toggles between KIS and VQA tasks, updating the UI
function toggleTask(task) {
  activeTask = task;
  const kisButton = document.getElementById('kis');
  const vqaButton = document.getElementById('vqa');
  const exportImages = document.getElementById('export-images');

  // Update export area
  if (task === 'kis') {
    kisButton.classList.add('active');
    vqaButton.classList.remove('active');
    exportImages.classList.remove('vqa-mode');
  } else {
    kisButton.classList.remove('active');
    vqaButton.classList.add('active');
    exportImages.classList.add('vqa-mode');
  }

  // Update fullscreen mode based on active task
  const kisFullscreenButton = document.getElementById('kis-fullscreen');
  const vqaFullscreenButton = document.getElementById('vqa-fullscreen');

  if (task === 'kis') {
    kisFullscreenButton.classList.add('active');
    vqaFullscreenButton.classList.remove('active');
  } else {
    kisFullscreenButton.classList.remove('active');
    vqaFullscreenButton.classList.add('active');
  }

  updateExportArea();
  loadExportContent();
}

//---------------------------------------------------------------------------------------------//
// Export toggle

let exportedImages = [];

// Resets the export area by clearing the list of exported images.
function resetExportArea() {
  exportedImages = [];
  updateExportArea();
}

// Handles the "drop" event to prevent default behavior (file dragging).
function allowDrop(ev) {
  ev.preventDefault();
}


// Update the export area UI with the list of exported images.
function updateExportArea() {
  const exportImages = document.getElementById('export-images');
  
  const htmlContent = exportedImages.map((img, index) => `
    <div class="export-image-container">
      <img src="${img.src}" class="export-image" title="Frame ID: ${img.frameId}">
      <button class="delete-button" onclick="deleteExportImage(${index})">×</button>
      <div class="infor">${img.frameInfo}</div>
      ${activeTask === 'vqa' ? `
        <input 
          type="text" 
          class="vqa-input" 
          value="${vqaInputs[img.frameId] || ''}" 
          placeholder="VQA Input" 
        >
      ` : ''}
    </div>
  `).join('');
  
  exportImages.innerHTML = htmlContent;
}

// Updates the VQA input for a specific frame ID in the export area.
function updateVqaInput(frameId, vqaInput) {
  vqaInputs[frameId] = vqaInput;
  updateExportArea();
}

// Sends an update to the WebSocket server with the new VQA input for a specific frame ID.
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


//---------------------------------------------------------------------------------------------//
// Export Logging

// Logs export data to the WebSocket server
function logExport(fileName, taskType, topImages) {
  if (logSocket && logSocket.readyState === WebSocket.OPEN) {
    // Create a JSON object with the logging data
    const logData = {
      file_name: fileName,
      task_type: taskType,
      top_images: topImages
    };
    // Send the logging data through the WebSocket connection
    logSocket.send(JSON.stringify(logData));
  } else {
    console.error('Log WebSocket is not open. Unable to send log data.');
  }
}

//---------------------------------------------------------------------------------------------//
// Export to CSV

// Triggers the export process to a CSV file based on the currently active task (KIS or VQA).
function exportToCSV() {
  const fileName = document.getElementById('filenameInput').value || getSuggestedFileName();

  if (activeTask === 'kis') {
    exportKisCSV(fileName);
  } else if (activeTask === 'vqa') {
    exportVqaCSV(fileName);
  }
}

// Export the images to a CSV file, including any necessary additional images to reach a total of 100.
function exportKisCSV(fileName) {
  const currentResults = getCurrentResults();
  const sortedResults = currentResults.sort((a, b) => b.score - a.score);
  
  const exportData = exportedImages.slice(0, 100);
  
  if (exportData.length < 100) {
    const remainingCount = 100 - exportData.length;
    const additionalImages = sortedResults
      .filter(result => !exportedImages.some(img => img.frameId === result.entity.frame_id))
      .slice(0, remainingCount)
      .map(result => ({
        frameId: result.entity.frame_id,
        src: result.entity.path
      }));
    exportData.push(...additionalImages);
    console.log(additionalImages);
  }

  const csvData = exportData.map(item => {
    const pathParts = item.src.split('/');
    const videoName = pathParts[pathParts.length - 3];
    const imageName = pathParts[pathParts.length - 1];
    const imageNumber = imageName.split('_')[1].split('.')[0];
    return `${videoName},${imageNumber}`;
  });

  const csvContent = csvData.join('\n');
  downloadCSV(csvContent, fileName);

  const topImages = csvData.slice(0, 5);
  logExport(fileName, 'kis', topImages);
}

function exportVqaCSV(fileName) {
  const csvData = exportedImages.map(item => {
    const pathParts = item.src.split('/');
    const videoName = pathParts[pathParts.length - 3];
    const imageName = pathParts[pathParts.length - 1];
    const imageNumber = imageName.split('_')[1].split('.')[0];
    const vqaInput = vqaInputs[item.frameId] || '0';
    return `${videoName},${imageNumber},${vqaInput}`;
  });

  const csvContent = csvData.join('\n');
  downloadCSV(csvContent, fileName);

  const topImages = csvData.slice(0, 5);
  logExport(fileName, 'vqa', topImages);
}

function downloadCSV(csvContent, fileName) {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}


// Delete an image from the export area and update the UI.
function deleteExportImage(index) {
  exportedImages.splice(index, 1);
  updateExportArea();
}


// Add drag event listeners to image elements in the right panel.
function addDragListeners() {
  document.querySelectorAll('.right-panel .img-dis img, .frame-container img, .preview-image-wrapper img, .current-preview-wrapper img').forEach(img => {
    img.setAttribute('draggable', 'true');
    img.addEventListener('dragstart', drag);
  });
}


// Initialize additional event listeners after the DOM content is loaded.
// Add event listeners for buttons, drag-and-drop functionality, and window resize events.
document.addEventListener('DOMContentLoaded', function() {
  // Connect WebSockets
  connectWebSocketcrossing();
  connectSimilaritySearchWebSocket();
  connectFilterWebSocket();

  // Get DOM elements
  const exportButton = document.getElementById('export-button');
  // const submitButton = document.getElementById('submit-button');
  const exportArea = document.getElementById('export-area');
  const openExportButton = document.getElementById('open-export');
  const resetExportButton = document.getElementById('reset-export');
  const kisButton = document.getElementById('kis');
  const vqaButton = document.getElementById('vqa');
  const contentWrapper = document.querySelector('.content-wrapper');

  // Add event listeners
  exportButton.addEventListener('click', toggleExportArea);
  openExportButton.addEventListener('click', toggleExportArea);
  // submitButton.addEventListener('click', exportToCSV);
  exportArea.addEventListener('dragover', allowDrop);
  exportArea.addEventListener('drop', drop);
  resetExportButton.addEventListener('click', resetExportArea);
  kisButton.addEventListener('click', () => toggleTask('kis'));
  vqaButton.addEventListener('click', () => toggleTask('vqa'));

  // Add drag listeners to images
  addDragListeners();

  // Handle window resize
  window.addEventListener('resize', function() {
    if (exportArea.classList.contains('show')) {
      contentWrapper.style.width = '80%';
    }
  });
});


//---------------------------------------------------------------------------------------------//

// Open the export area if it is closed
function openExportAreaIfClosed() {
  const exportArea = document.getElementById('export-area');
  const contentWrapper = document.querySelector('.content-wrapper');
  
  if (!exportArea.classList.contains('show')) {
    exportArea.classList.add('show');
    contentWrapper.classList.add('export-active');
    contentWrapper.style.width = '80%';
  }
}


// Add an image to the export area if it is not already present.
// Open the export area if it is closed and update the export area UI.
function addImageToExportArea(frameId, imageSrc, frameInfo, shouldBroadcast = true) {
  frameId = parseInt(frameId, 10);

  const pathParts = imageSrc.split('/');
  const videoFramePart = `${pathParts[pathParts.length - 2]}/${pathParts[pathParts.length - 1].split('.')[0]}`;

  const existingImageIndex = exportedImages.findIndex(img => img.frameId === frameId && img.videoFramePart === videoFramePart);

  if (existingImageIndex === -1) {
    exportedImages.push({ frameId, videoFramePart, src: imageSrc, frameInfo });
    openExportAreaIfClosed();
    updateExportArea();

    if (shouldBroadcast && socket_share && socket_share.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'image_share',
        frameId: frameId,
        src: imageSrc,
        frameInfo: frameInfo
      });
      socket_share.send(message);
    }
  } else {
    console.log('Image already exists in the export area');
  }
}


// Handle middle mouse button click on an image to add it to the export area.
function handleMiddleClick(event) {
  if (event.button === 1) { // Middle mouse button
    event.preventDefault();
    const container = event.target.closest('.img-dis, .frame-container, .preview-image-wrapper, .current-preview-wrapper');
    if (container) {
      const imgElement = container.querySelector('img');
      const inforElement = container.querySelector('.infor');
      if (imgElement && inforElement) {
        const frameId = inforElement.textContent.split('-')[1];
        const imageSrc = imgElement.src;
        addImageToExportArea(frameId, imageSrc, inforElement.textContent);
      }
    }
  }
}


//-----------------------------------------------------------------------//
// Drag - drop image in list photo and frame container

function drag(ev) {
  ev.stopPropagation();

  const imgElement = ev.target;
  let frameId, imageSrc, frameInfo;

  const container = imgElement.closest('.img-dis, .frame-container, .preview-image-wrapper, .current-preview-wrapper');
  
  if (container) {
    const inforElement = container.querySelector('.infor');
    if (inforElement) {
      frameInfo = inforElement.textContent;
      frameId = frameInfo.split('-')[1];
      imageSrc = imgElement.src;

      ev.dataTransfer.setData('application/json', JSON.stringify({
        type: 'image',
        frameId: frameId,
        imageSrc: imageSrc,
        frameInfo: frameInfo
      }));
    } else {
      console.error('Info element not found');
    }
  } else {
    console.error('Container not found');
  }
}

let isImageDragging = false;

function drop(ev) {
  ev.preventDefault();
  if (isImageDragging) return;
  isImageDragging = true;

  const data = ev.dataTransfer.getData('application/json');
  
  if (data) {
    try {
      const parsedData = JSON.parse(data);
      
      if (parsedData.type === 'image' && parsedData.frameId && parsedData.imageSrc) {
        addImageToExportArea(parsedData.frameId, parsedData.imageSrc, parsedData.frameInfo);
      } else {
        console.error('Invalid data format');
      }
    } catch (error) {
      console.error('Error parsing dropped data:', error);
    }
  }

  setTimeout(() => { isImageDragging = false; }, 100);
}



//-----------------------------------------------------------------------//
// Export full screen

// Function to show the fullscreen overlay
function showExportFullscreen() {
  document.getElementById('export-fullscreen-overlay').style.display = 'block';
  loadExportContent();
}

// Function to hide the fullscreen overlay
function hideExportFullscreen() {
  document.getElementById('export-fullscreen-overlay').style.display = 'none';
  openExportAreaIfClosed();
}

// Function to load export content
function loadExportContent() {
  const exportFullscreenContent = document.getElementById('export-fullscreen-content');
  exportFullscreenContent.innerHTML = '';

  if (activeTask === 'kis') {
    // console.log('kis content loading');
    exportedImages.forEach(img => {
      const container = document.createElement('div');
      container.className = 'export-item';

      const imgElement = document.createElement('img');
      imgElement.src = img.src;
      imgElement.alt = `Frame ID: ${img.frameId}`;

      const infoElement = document.createElement('div');
      infoElement.className = 'info';
      infoElement.textContent = img.frameInfo;

      container.appendChild(imgElement);
      container.appendChild(infoElement);
      exportFullscreenContent.appendChild(container);
    });
  } else if (activeTask === 'vqa') {
    // console.log('vqa content loading');
    exportedImages.forEach(img => {
      const container = document.createElement('div');
      container.className = 'export-item';

      const imgElement = document.createElement('img');
      imgElement.src = img.src;
      imgElement.alt = `Frame ID: ${img.frameId}`;

      const infoElement = document.createElement('div');
      infoElement.className = 'info';
      infoElement.textContent = img.frameInfo;

      const vqaInput = document.createElement('input');
      vqaInput.type = 'text';
      vqaInput.className = 'vqa-input';
      vqaInput.value = vqaInputs[img.frameId] || '';
      vqaInput.placeholder = 'VQA Input';
      vqaInput.onchange = (e) => {
        vqaInputs[img.frameId] = e.target.value;
        sendVqaInputUpdate(img.frameId, e.target.value);
      };

      container.appendChild(imgElement);
      container.appendChild(infoElement);
      container.appendChild(vqaInput);
      exportFullscreenContent.appendChild(container);
    });
  }
}


// Event listeners for kis fullscreen and vqa fullscreen
document.getElementById('kis-fullscreen').addEventListener('click', () => {
  toggleTask('kis');
});

document.getElementById('vqa-fullscreen').addEventListener('click', () => {
  toggleTask('vqa');
});


document.getElementById('reset-export-fullscreen').addEventListener('click', () => {
  document.getElementById('export-fullscreen-content').innerHTML = '';
  resetExportArea()
});

document.getElementById('close-export-fullscreen').addEventListener('click', hideExportFullscreen);

// Add event listener to the open-export button
document.getElementById('open-export').addEventListener('click', showExportFullscreen);


// Function to check if the export fullscreen overlay is visible
function isExportFullscreenVisible() {
  return document.getElementById('export-fullscreen-overlay').style.display === 'block';
}

// Event listener for the 'Escape' key
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && isExportFullscreenVisible()) {
    hideExportFullscreen();
  }
});



//-----------------------------------------------------------------------//

function getSuggestedFileName() {
  const kisButton = document.getElementById('kis');
  const vqaButton = document.getElementById('vqa');

  if (kisButton.classList.contains('active')) {
    return 'query-p3--kis.csv'; // Task KIS
  } else if (vqaButton.classList.contains('active')) {
    return 'query-p3--qa.csv'; // Task QA
  }
}

// Hiển thị exportModal khi nhấn nút submit trên thanh header
document.getElementById('submit-button').addEventListener('click', function() {
  // const filenameInput = document.getElementById('filenameInput');
  // filenameInput.value = getSuggestedFileName();

  // document.getElementById('exportModal').style.display = 'block';
  submit_to_dres_v2()
});

// Hiển thị exportModal khi nhấn nút submit ở fullscreen
document.getElementById('submit-fullscreen-button').addEventListener('click', function() {
  // Gợi ý tên file vào input
  const filenameInput = document.getElementById('filenameInput');
  filenameInput.value = getSuggestedFileName();

  // Hiển thị modal
  document.getElementById('exportModal').style.display = 'block';
});

// Đóng modal khi người dùng nhấn cancel
document.querySelector('.export-modal-btn.cancel').addEventListener('click', function() {
  document.getElementById('exportModal').style.display = 'none';
});

// Xử lý xuất file khi người dùng nhấn confirm
document.querySelector('.export-modal-btn.confirm').addEventListener('click', function() {
  const fileName = document.getElementById('filenameInput').value;

  exportToCSV(fileName);
  // submit_to_dres();

  document.getElementById('exportModal').style.display = 'none';
});



//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//
// Open or close door share image mode


// Add this at the beginning of your JavaScript file
let isExportShared = true;

// Function to toggle export mode
function toggleExportMode() {
  const openExportSocketButton = document.getElementById('open-export-socket');
  isExportShared = !isExportShared;
  
  if (isExportShared) {
    openExportSocketButton.innerHTML = '<img src="src/Img/icon-door-open.png" alt="icon">';
    openExportSocketButton.title = 'Switch to private mode';
  } else {
    openExportSocketButton.innerHTML = '<img src="src/Img/icon-door-close.png" alt="icon">';
    openExportSocketButton.title = 'Switch to shared mode';
  }
}


// Modify the addImageToExportArea function
function addImageToExportArea(frameId, imageSrc, frameInfo, shouldBroadcast = true) {
  frameId = parseFloat(frameId);

  const pathParts = imageSrc.split('/');
  const videoFramePart = `${pathParts[pathParts.length - 2]}/${pathParts[pathParts.length - 1].split('.')[0]}`;

  const existingImageIndex = exportedImages.findIndex(img => img.frameId === frameId && img.videoFramePart === videoFramePart);

  if (existingImageIndex === -1) {
    exportedImages.push({ frameId, videoFramePart, src: imageSrc, frameInfo });
    openExportAreaIfClosed();
    updateExportArea();

    if (isExportShared && shouldBroadcast && socket_share && socket_share.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'image_share',
        frameId: frameId,
        src: imageSrc,
        frameInfo: frameInfo
      });
      socket_share.send(message);
    }
  } else {
    console.log('Image already exists in the export area');
  }
}


// Add this to your DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
  const openExportSocketButton = document.getElementById('open-export-socket');
  openExportSocketButton.addEventListener('click', toggleExportMode);
});


// Modify the WebSocket message handler
socket_share.onmessage = (event) => {
  if (!isExportShared) return; // Don't process messages if not in shared mode

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



// //------------------------------------------------------------------------------------------------------------------------------//
// //------------------------------------------------------------------------------------------------------------------------------//
// //------------------------------------------------------------------------------------------------------------------------------//
// //------------------------------------------------------------------------------------------------------------------------------//




// function getFirstResultForKIS(){
//   console.log(exportedImages)
//   return exportedImages[0]
// }

// function getFirstResultForVQA(){
//   const vqa=document.getElementsByClassName("vqa-input") 
//   return [exportedImages[0],vqa[0].value]
// }



// async function submit_to_dres(){
//   let frame_info=0
//   if (activeTask=="kis"){
//   frame_info= getFirstResultForKIS()
//   } else{
//   frame_info= getFirstResultForVQA()  
//   }
//   const item=frame_info.frameInfo.split('-')[0];
//   const frame= parseInt(frame_info.frameId)*1000;
//   const sessionID=localStorage.getItem('sessionId');
//   const url= `http://192.168.20.164:5000/api/v1/submit?item=${item}&frame=${frame}&session=${sessionID}`
//   const result = await fetch(url, {
//     method: "GET",
//   })
//   .then(response => {
//     if (!response.ok) {
//       // If the response status is 401 (Unauthorized)
//       if (response.status === 401) {
//         alert("Unauthorized access. Please check your credentials.");
//       } 
//       // If the response status is 404 (Not Found)
//       else if (response.status === 404) {
//         alert("WRONG!!");
//       } 
//       // Other possible errors
//       else {
//         alert(`Error: ${response.status}`);
//       }
//       throw new Error(`HTTP status code ${response.status}`);
//     }
//     return response.json();
//   })
//   .then(data => {
//     console.log('Success:', data);
//     // Process data
//   })
//   .catch(error => {
//     console.error('Error:', error);
//   });
// }

// function submit_to_dres_but_dare_devil(item, frame){
//   const url= `http://192.168.20.164:6001/api/v1/submit?item=${item}&frame=${frame}`
//   console.log(url)
//   fetch(url, {
//     method: "GET",
//   });
// }


// async function submit_to_dres_v2(){
//     let frame_info=0

//     const evaluationID = localStorage.getItem('evaluationID');
//     const contestSessionID = localStorage.getItem('contestSessionID');
//     const contestURL=`https://eventretrieval.one/api/v2/submit/${evaluationID}?session=${contestSessionID}`;

//     if (activeTask=="kis"){
//     const frame_info= getFirstResultForKIS()
//     const item=frame_info.frameInfo.split('-')[0];
//     const frame= parseFloat(frame_info.frameId.toFixed(2))*1000.0;
//     const contestresult= await fetch(contestURL,{
//       method:"POST",
//       body:JSON.stringify({"answerSets": [{
//         "answers": [
//         {
//           "mediaItemName": item,
//           "start": frame,
//           "end": frame
//         }]
//       }]
//      })
//     }).then()
//     } else{
//     const frame_info= getFirstResultForVQA() 
//     const item=frame_info[0].frameInfo.split('-')[0];
//     const answer_vqa= frame_info[1]
//     const frame= parseFloat(frame_info[0].frameId.toFixed(2))*1000.0;
//     const contestresult= await fetch(contestURL,{
//       method:"POST",
//       body:JSON.stringify({"answerSets": [{
//         "answers": [
//         {
//           "text": `${answer_vqa}-${item}-${frame}`
//         }]
//       }]
//      })
//     }).then()
//     }
// }