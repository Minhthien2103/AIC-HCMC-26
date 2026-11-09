function performGroupSearch(imageId) {
    if (groupSearchSocket.readyState === WebSocket.OPEN) {
        requestTime = performance.now();
        toggleLoadingIndicator(true);
        groupSearchSocket.send(JSON.stringify({ imageId: imageId}));
    } else {
        console.error('Group Search WebSocket is not open');
        toggleLoadingIndicator(false);
    }
}

// function setupImageCtrlClickListeners() {
//     const images = document.querySelectorAll('.img-dis img');
    
//     images.forEach(img => {
//         img.addEventListener('click', (event) => {
//             if (event.ctrlKey) {
//                 event.preventDefault(); // Prevent default behavior
//                 performGroupSearch(img.id);
//             }
//         });
//     });
// }