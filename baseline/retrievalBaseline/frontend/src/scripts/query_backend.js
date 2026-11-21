//------------------------ Search Query ------------------------//

// Handle Enter keypress on query textareas
function handleEnterKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    performSearchFromInputs();
  }
}

// Show the exact strings encoded by the model, including fallback translations.
function showQueryPreparation(payload, error = '') {
  const errorElement = document.getElementById('query-error');
  if (errorElement) {
    errorElement.textContent = error;
    errorElement.hidden = !error;
  }
  const details = document.getElementById('query-diagnostics');
  const text = document.getElementById('query-diagnostics-text');
  const plans = payload?.query_preparation || [];
  const translation = document.getElementById('query-translation');
  if (translation) {
    const primary = plans.map((plan, index) => {
      const english = plan.english?.[0];
      if (!english) return '';
      return `${plans.length > 1 ? `Sự kiện ${index + 1}: ` : ''}${english}`;
    }).filter(Boolean);
    translation.textContent = primary.length ? `Bản dịch chính dùng để tìm:\n${primary.join('\n')}` : '';
    translation.hidden = primary.length === 0;
  }
  if (details) details.hidden = plans.length === 0;
  if (text) text.textContent = plans.map((plan, index) => {
    const source = {original: 'Giữ nguyên truy vấn', local_llm: 'Mô hình local', translator: 'Bộ dịch dự phòng'}[plan.source] || plan.source;
    return [`Mô tả ${index + 1} — ${source}`, ...(plan.warnings || []),
      ...(plan.chunks || []).map((chunk, i) => `${i + 1}. ${chunk} (${plan.token_counts?.[i] ?? '?'} / ${plan.token_limit ?? '?'} token)`)].join('\n');
  }).join('\n\n');
}

let latestTextRequest = 0;
const performSearchMulti = debounce(async function (queries, ocrText = '') {
  const requestId = ++latestTextRequest;
  if (currentAbortController) currentAbortController.abort();
  currentAbortController = null;
  if (!queries || !queries.some(query => query.trim())) {
    toggleLoadingIndicator(false);
    return;
  }
  const cacheKey = JSON.stringify([queries, ocrText]);
  showQueryPreparation(null);
  if (searchCache.has(cacheKey)) {
    data = searchCache.get(cacheKey);
    updateUIWithSearchResults(data.kq);
    showQueryPreparation(data);
    toggleLoadingIndicator(false);
    return;
  }
  toggleLoadingIndicator(true);
  const controller = new AbortController();
  currentAbortController = controller;
  try {
    const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || 'http://localhost:8006';
    const response = await fetch(`${backendBase}/TextQuery`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({queries, ocr_text: ocrText, First_query: queries[0] || '', Next_query: queries[1] || ''}),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (requestId !== latestTextRequest) return;
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : `HTTP ${response.status}`;
      throw new Error(`${detail}\nAPI: ${backendBase}/TextQuery`);
    }
    data = payload;
    searchCache.set(cacheKey, payload);
    updateUIWithSearchResults(data.kq);
    showQueryPreparation(data);
  } catch (error) {
    if (requestId === latestTextRequest && error.name !== 'AbortError') {
      data = {kq: []};
      updateUIWithSearchResults([]);
      showQueryPreparation(null, `Không thể tìm kiếm: ${error.message}`);
      console.error('Search failed:', error);
    }
  } finally {
    if (requestId === latestTextRequest) {
      toggleLoadingIndicator(false);
      currentAbortController = null;
    }
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

// Perform search from active inputs: automatically detects Text (KIS-T, KIS-QA, TRAKE) or Image (KIS-V)
async function performSearchFromInputs() {
  toggleLoadingIndicator(true);
  const ocrText = document.getElementById("ocr-text-input")?.value.trim() || "";

  // 1. Check if Scene 1 is in Image Mode (KIS-V Visual Search)
  const scene1 = document.getElementById('search-scene-1');
  const imageDropArea = scene1 ? scene1.querySelector('.image-drop-area') : null;
  const isImageTab = imageDropArea && imageDropArea.style.display !== 'none';
  const imgElement = imageDropArea ? imageDropArea.querySelector('img') : null;

  if (isImageTab && imgElement && imgElement.src) {
    try {
      console.log("[Search] Performing Visual Search (KIS-V)...");
      const backendBase = (window.CONFIG && window.CONFIG.BACKEND_BASE) || window.BACKEND_BASE || "http://localhost:8006";
      let imageB64 = imgElement.src;
      if (imageB64.includes(',')) {
        imageB64 = imageB64.split(',')[1];
      }

      const textQueryArea = scene1.querySelector('textarea[name="Text_Query"]');
      const textQuery = textQueryArea ? textQueryArea.value.trim() : "";

      const response = await fetch(`${backendBase}/mixed_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageB64, text: textQuery, ocr_text: ocrText })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const resJson = await response.json();
      data = resJson;
      updateUIWithSearchResults(resJson.kq);
    } catch (err) {
      console.error("[Visual Search Error]", err);
    } finally {
      toggleLoadingIndicator(false);
    }
    return;
  }

  // 2. Text Search (KIS-T, KIS-QA, TRAKE)
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

  performSearchMulti(queries, ocrText);
}

window.performSearchFromInputs = performSearchFromInputs;
window.performSearchFromTextareas = performSearchFromInputs;

document.getElementById("search-button").addEventListener("click", function(event) {
  event.preventDefault();
  performSearchFromInputs();
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


// Enter in the optional OCR field performs the same visual search.
document.getElementById("ocr-text-input")?.addEventListener("keydown", handleEnterKeyPress);
