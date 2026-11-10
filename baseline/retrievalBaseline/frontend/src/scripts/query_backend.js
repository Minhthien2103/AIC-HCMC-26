//------------------------ Search Query ------------------------//

// Handle Enter keypress on query textareas
function handleEnterKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    performSearchFromTextareas();
  }
}

// Perform multi-event search with debounce (supports KIS & TRAKE with N events)
const performSearchMulti = debounce(async function (queries) {
  if (!queries || queries.length === 0) {
    toggleLoadingIndicator(false);
    return;
  }

  const cacheKey = queries.join(" || ");
  
  if (searchCache.has(cacheKey)) {
    updateUIWithSearchResults(searchCache.get(cacheKey));
    toggleLoadingIndicator(false);
    return;
  }

  toggleLoadingIndicator(true);

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  try {
    const startTime = performance.now();
    const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
    const response = await fetch(`${backendBase}/TextQuery`, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=1000'
      },
      body: JSON.stringify({
        queries: queries,
        First_query: queries[0] || "",
        Next_query: queries[1] || "",
      }),
      signal: currentAbortController.signal,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    data = await response.json();
    const duration = performance.now() - startTime;
    console.log(`Fetch operation took ${duration.toFixed(2)}ms for ${queries.length} events`);
    
    searchCache.set(cacheKey, data.kq);
    updateUIWithSearchResults(data.kq);
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.error('Fetch error:', error);
    }
  } finally {
    toggleLoadingIndicator(false);
  }
}, 300);

// Backward-compatible wrapper for single/double query
const performSearch = (firstQuery, nextQuery) => {
  const qList = [firstQuery, nextQuery].filter(q => q && q.trim());
  performSearchMulti(qList.length > 0 ? qList : [""]);
};

// Perform initial search when loading web first time
async function performInitialSearch() {
  performSearch("", "");
}

// Bind Enter keypress to all query textareas
document.querySelectorAll('.Search_Scene textarea[name="Text_Query"]').forEach(textarea => {
  textarea.addEventListener('keydown', handleEnterKeyPress);
});

// Perform search from all active text areas
async function performSearchFromTextareas() {
  toggleLoadingIndicator(true);
  const textareas = document.querySelectorAll('.Search_Scene textarea[name="Text_Query"]');
  const queries = [];
  textareas.forEach(ta => {
    queries.push(ta.value.trim());
  });

  const activeQueries = queries.filter(q => q.length > 0);
  if (activeQueries.length === 0) {
    toggleLoadingIndicator(false);
    return;
  }

  performSearchMulti(queries);
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

