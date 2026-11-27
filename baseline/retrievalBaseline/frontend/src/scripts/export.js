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

// Toggles between KIS-T, KIS-V, KIS-QA, and TRAKE tasks, updating the UI
function toggleTask(task) {
  activeTask = task;
  const kisButton = document.getElementById('kis');
  const kisVButton = document.getElementById('kis-v');
  const kisQAButton = document.getElementById('kis-qa');
  const vqaButton = document.getElementById('vqa');
  const trakeButton = document.getElementById('trake');
  const exportImages = document.getElementById('export-images');
  const globalTaskSelect = document.getElementById('global-task-select');

  if (globalTaskSelect && globalTaskSelect.value !== task) {
    globalTaskSelect.value = task;
  }

  [kisButton, kisVButton, kisQAButton, vqaButton, trakeButton].forEach(btn => {
    if (btn) btn.classList.remove('active');
  });

  const scene1 = document.getElementById('search-scene-1');
  const scene1Title = document.getElementById('scene-1-title');
  const scene2Title = document.getElementById('scene-2-title');
  const textQuery1 = document.getElementById('Text-Query-First');

  if (task === 'kis') {
    if (kisButton) kisButton.classList.add('active');
    if (exportImages) exportImages.classList.remove('vqa-mode');
    if (scene1Title) scene1Title.textContent = "Mô tả Cảnh (KIS-T)";
    if (scene2Title) scene2Title.textContent = "Mô tả phụ (Tuỳ chọn)";
    if (textQuery1) textQuery1.placeholder = "Nhập mô tả cảnh KIS-T (VD: người phụ nữ mặc áo xanh, xe cứu hỏa)...";
    if (scene1) {
      const textBtn = scene1.querySelector('.text-button');
      if (textBtn) textBtn.click();
    }
  } else if (task === 'kis-v') {
    if (kisVButton) kisVButton.classList.add('active');
    if (exportImages) exportImages.classList.remove('vqa-mode');
    if (scene1Title) scene1Title.textContent = "Ảnh Mẫu (KIS-V)";
    if (scene2Title) scene2Title.textContent = "Mô tả phụ (Tuỳ chọn)";
    if (scene1) {
      const imgBtn = scene1.querySelector('.image-button');
      if (imgBtn) imgBtn.click();
    }
  } else if (task === 'kis-qa' || task === 'vqa') {
    if (kisQAButton) kisQAButton.classList.add('active');
    if (vqaButton) vqaButton.classList.add('active');
    if (exportImages) exportImages.classList.add('vqa-mode');
    if (scene1Title) scene1Title.textContent = "Câu hỏi KIS-QA";
    if (scene2Title) scene2Title.textContent = "Ngữ cảnh phụ (Tuỳ chọn)";
    if (textQuery1) textQuery1.placeholder = "Nhập câu hỏi Q&A (VD: Vận động viên mang số áo bao nhiêu?)...";
    if (scene1) {
      const textBtn = scene1.querySelector('.text-button');
      if (textBtn) textBtn.click();
    }
  } else if (task === 'trake') {
    if (trakeButton) trakeButton.classList.add('active');
    if (exportImages) exportImages.classList.remove('vqa-mode');
    if (scene1Title) scene1Title.textContent = "Sự kiện 1 (TRAKE)";
    if (scene2Title) scene2Title.textContent = "Sự kiện 2 (TRAKE)";
    if (textQuery1) textQuery1.placeholder = "Nhập Sự kiện 1 của TRAKE (VD: xe hơi rẽ trái vào ngõ)...";
  }

  // Update fullscreen mode based on active task
  const kisFullscreenButton = document.getElementById('kis-fullscreen');
  const vqaFullscreenButton = document.getElementById('vqa-fullscreen');

  if (task === 'kis' || task === 'kis-v' || task === 'trake') {
    if (kisFullscreenButton) kisFullscreenButton.classList.add('active');
    if (vqaFullscreenButton) vqaFullscreenButton.classList.remove('active');
  } else {
    if (kisFullscreenButton) kisFullscreenButton.classList.remove('active');
    if (vqaFullscreenButton) vqaFullscreenButton.classList.add('active');
  }

  updateExportArea();
  if (typeof loadExportContent === 'function') {
    loadExportContent();
  }
}

//---------------------------------------------------------------------------------------------//
// Export toggle

let exportedImages = [];

// Resets the export area by clearing the list of exported images.
function resetExportArea() {
  exportedImages = [];
  vqaInputs = {};
  updateExportArea();
}

// Handles the "drop" event to prevent default behavior (file dragging).
function allowDrop(ev) {
  ev.preventDefault();
}


// Update the export area UI with the list of exported images.
function createPinnedElement(img, index, fullscreen = false) {
  const container = document.createElement('div');
  container.className = fullscreen ? 'export-item' : 'export-image-container';
  const picture = document.createElement('img');
  // img is normalized frame metadata, not an HTMLImageElement.
  picture.src = img.src;
  picture.className = 'export-image';
  registerFrameMetadata(picture, img);
  container.appendChild(picture);
  const info = document.createElement('div');
  info.className = 'infor';
  info.textContent = img.frameInfo;
  container.appendChild(info);
  if (!fullscreen) {
    const remove = document.createElement('button');
    remove.className = 'delete-button';
    remove.textContent = '×';
    remove.onclick = () => deleteExportImage(index);
    container.appendChild(remove);
  }
  if (['qa', 'vqa', 'kis-qa'].includes(activeTask)) {
    const input = document.createElement('input');
    input.type = 'text';
    input.inputMode = 'numeric';
    input.pattern = '[0-9]+';
    input.className = 'vqa-input';
    input.dataset.frameKey = img.key;
    input.value = vqaInputs[img.key] ?? '';
    input.placeholder = 'Nhập đáp án QA (chỉ chữ số)...';
    input.setAttribute('aria-label', `Đáp án QA cho ${img.videoId}, ${img.timeSeconds} giây`);
    input.style.width = '100%';
    input.style.boxSizing = 'border-box';
    input.oninput = () => {
      updateVqaInput(img.key, input.value);
      sendVqaInputUpdate(img.key, input.value);
    };
    container.appendChild(input);
  }
  return container;
}

function updateExportArea() {
  const target = document.getElementById('export-images');
  target.replaceChildren(...exportedImages.map((img, index) => createPinnedElement(img, index)));
  if (isExportFullscreenVisible()) loadExportContent();
}

// Updates the VQA input for a specific frame ID in the export area.
function updateVqaInput(frameKey, answer) {
  if (!exportedImages.some(item => item.key === frameKey)) return;
  vqaInputs[frameKey] = String(answer);
  document.querySelectorAll('.vqa-input').forEach(input => {
    if (input.dataset.frameKey === frameKey && input.value !== String(answer)) input.value = String(answer);
  });
}

function sendVqaInputUpdate(frameKey, answer) {
  if (isExportShared && socket_share && socket_share.readyState === WebSocket.OPEN) {
    socket_share.send(JSON.stringify({type: 'vqa_input_update', frameKey, vqaInput: answer}));
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
  } else if (activeTask === 'vqa' || activeTask === 'kis-qa') {
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
  try {
    const rows = exportedImages.map(item => {
      const frame = normalizePinnedFrame(item);
      const answer = validateQaAnswer(vqaInputs[frame.key]);
      return `${frame.videoId},${frame.keyframeName},${answer}`;
    });
    downloadCSV(rows.join('\n'), fileName);
  } catch (error) { showTemporaryAlert(error.message, 'warning'); }
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
  const removed = exportedImages.splice(index, 1)[0];
  if (removed) delete vqaInputs[removed.key];
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
  if (kisButton) kisButton.addEventListener('click', () => toggleTask('kis'));
  const kisVButton = document.getElementById('kis-v');
  if (kisVButton) kisVButton.addEventListener('click', () => toggleTask('kis-v'));
  const kisQAButton = document.getElementById('kis-qa');
  if (kisQAButton) kisQAButton.addEventListener('click', () => toggleTask('kis-qa'));
  if (vqaButton) vqaButton.addEventListener('click', () => toggleTask('vqa'));
  const trakeButton = document.getElementById('trake');
  if (trakeButton) trakeButton.addEventListener('click', () => toggleTask('trake'));
  const globalTaskSelect = document.getElementById('global-task-select');
  if (globalTaskSelect) {
    globalTaskSelect.addEventListener('change', function() {
      toggleTask(this.value);
    });
  }

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
function pinImage(img) {
  try { addImageToExportArea(frameFromImage(img)); }
  catch (error) { showTemporaryAlert(error.message, 'warning'); }
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
        pinImage(imgElement);
      }
    }
  }
}


//-----------------------------------------------------------------------//
// Drag - drop image in list photo and frame container

function drag(ev) {
  ev.stopPropagation();
  try {
    const frame = frameFromImage(ev.target);
    ev.dataTransfer.setData('application/json', JSON.stringify({type: 'image', frame}));
  } catch (error) {
    ev.preventDefault();
    showTemporaryAlert(error.message, 'warning');
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
      
      if (parsedData.type === 'image') addImageToExportArea(parsedData.frame);
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
  document.getElementById('export-fullscreen-content').replaceChildren(
    ...exportedImages.map((img, index) => createPinnedElement(img, index, true)));
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
function addImageToExportArea(value, shouldBroadcast = true) {
  let frame;
  try { frame = normalizePinnedFrame(value); }
  catch (error) { showTemporaryAlert(error.message, 'warning'); return; }
  if (exportedImages.some(item => item.key === frame.key)) return;
  exportedImages.push(frame);
  openExportAreaIfClosed();
  updateExportArea();
  if (isExportShared && shouldBroadcast && socket_share && socket_share.readyState === WebSocket.OPEN) {
    socket_share.send(JSON.stringify({type: 'image_share', frame}));
  }
}

function handleSharedExportMessage(event) {
  if (!isExportShared) return;
  try {
    const message = JSON.parse(event.data);
    if (message.type === 'image_share') addImageToExportArea(message.frame, false);
    else if (message.type === 'vqa_input_update' && message.frameKey && message.vqaInput !== undefined) {
      updateVqaInput(message.frameKey, message.vqaInput);
    }
  } catch (error) { console.error('Invalid shared frame data', error); }
}

// Add this to your DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
  const openExportSocketButton = document.getElementById('open-export-socket');
  openExportSocketButton.addEventListener('click', toggleExportMode);
});


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
