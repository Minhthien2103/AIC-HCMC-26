//------------------------ Search Query ------------------------//


// Perform search with debounce (prevents too many requests)
const performSearch = debounce(async function (firstQuery, nextQuery) {
  const cacheKey = `${firstQuery}:${nextQuery}`;
  
  if (searchCache.has(cacheKey)) {
    updateUIWithSearchResults(searchCache.get(cacheKey));
    return;
  }

  toggleLoadingIndicator(true);

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  try {
    let translationPromise = Promise.resolve([firstQuery, nextQuery]);
    if (document.getElementById('translate-checkbox').checked) {
      translationPromise = Promise.all([translateText(firstQuery), translateText(nextQuery)]);
    }

    const [translatedFirstQuery, translatedNextQuery] = await translationPromise;
    const startTime = performance.now();
    const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
    const response = await fetch(`${backendBase}/TextQuery`, {
      method: "POST",
      headers:{
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=1000'
      },
      body: JSON.stringify({
        First_query: translatedFirstQuery,
        Next_query: translatedNextQuery,
      }),
      signal: currentAbortController.signal,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    data = await response.json();


  const endTime = performance.now();
  const duration = endTime - startTime;

  console.log(`Fetch operation took ${duration.toFixed(2)} milliseconds`);
    
    searchCache.set(cacheKey, data.kq);
    updateUIWithSearchResults(data.kq);
    document.getElementById("Text-Query-First").value = data.fquery;
    document.getElementById("Text-Query-Second").value = data.nquery;
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.error('Fetch error:', error);
    }
  } finally {
    toggleLoadingIndicator(false);
  }
}, 300);
 

// there are some way to query in UI
//----------------------------------------------//
// When start web
// Perform initial search when loading web first time
async function performInitialSearch() {
    const firstQuery = ""; // You can set a default query here if needed
    const nextQuery = "";
    performSearch(firstQuery, nextQuery);
}
  

//-----------------------------------------------//
// query backend
document.querySelectorAll('.Search_Scene textarea[name="Text_Query"]').forEach(textarea => {
  textarea.addEventListener('keydown', handleEnterKeyPress);
});


// Perform search from text area
async function performSearchFromTextareas() {
  toggleLoadingIndicator(true);
  const textareas = document.querySelectorAll('.Search_Scene textarea[name="Text_Query"]');
  const firstQuery = textareas[0].value;
  const nextQuery = textareas[1] ? textareas[1].value : "";
  performSearch(firstQuery, nextQuery);
}

document.getElementById("search-button").addEventListener("click", function(event) {
  event.preventDefault();
  performSearchFromTextareas();
});


//------------------------ Similarity search ------------------------//

// Perform similarity search

const debouncedSimilaritySearch = debounce(async (img) => {
  
  const queryid = data.kq[img.id - 1].id;
  
  try {
      const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
      const response = await fetch(`${backendBase}/similarity_search`, {
          method: "POST",
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ vector: queryid }),
      });

      data = await response.json();

      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}, message: ${data.detail}`);
      }

      // console.log("Response data:", data);

      if (data && data.kq) {
          updateUIWithSearchResults(data.kq);
      } else {
          console.error("Unexpected response structure:", data);
      }
  } catch (error) {
      console.error('Similarity search error:', error);
  } finally {
      toggleLoadingIndicator(false);
  }
}, 300);


// Initiate similarity search
async function similaritySearch(img) {
  toggleLoadingIndicator(true);
  debouncedSimilaritySearch(img);
}

