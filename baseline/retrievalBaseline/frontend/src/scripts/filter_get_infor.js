//------------------------------------------------------------------------//

// Common function for handling filter functionality
async function handleFilterAction(event) {
    if (event) {
        event.preventDefault();
    }

    // Select all elements with the class 'Search_Scene'
    const scenes = document.querySelectorAll('.Search_Scene');
    const allFilters = [];
    const allTextQueries = [];
    const allImageQueries=[];
    const allOcrTexts = [];
    const allAsmTexts = [];

    // Translate queries
    for (let scene of scenes) {
            const Text = await getQueryContent(scene)
            if (Text['type'] === 'image'){
                allImageQueries.push(Text)
            }else{
                if (Text["content"==='']){
                }else{
                allTextQueries.push(Text);
            }
        }
    }

    // Extract filters and OCR text from each scene
    scenes.forEach((scene, index) => {
        const OcrTextArea = scene.querySelector("textarea[name='Ocr_Query']");
        const AsmTextArea = scene.querySelector("textarea[name='Asm_Query']");
        const filters = Array.from(scene.querySelectorAll(".object-filter")).map(section => ({
            name: section.querySelector("input[type='text']")?.value || '',
            number: section.querySelector("input[data-type='text']")?.value || ''
        }));

        allOcrTexts.push(OcrTextArea?.value || '');
        allAsmTexts.push(AsmTextArea?.value || '');
        allFilters.push(filters);
    });

    try {
        requestTime = performance.now();
        toggleLoadingIndicator(true);

        jsonString = JSON.stringify({
            ocrtext: allOcrTexts,
            asmtext: allAsmTexts,
            filters: allFilters,
            textQueries: allTextQueries,
            imageQueries: allImageQueries
        });

        console.log(jsonString);
        filterSocket.send(jsonString);

    } catch (error) {
        console.error('Filter query error:', error);
        toggleLoadingIndicator(false);
    }
}

// Event listener for filter button
document.getElementById("filter-button").addEventListener("click", handleFilterAction);
