// X Feed Curator - Content Script
// Posts → Embedding Model → Cosine Similarity → Ranked Filtering → Show Top K

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    batchSize: 30,
    maxMemoryMB: 200,
    apiEndpoints: [
      '/timeline',
      '/home_timeline',
      '/search/adaptive',
      '/graphql',
      '/2/timeline'
    ]
  };

  // State management
  const state = {
    filters: [], // { query: string, queryVec: Float32Array, topK: number, threshold: number, mode: 'hide'|'show' }
    processedPosts: new Map(), // postId -> { text, hidden }
    postBatch: [],
    stats: {
      processed: 0,
      hidden: 0
    },
    enabled: true
  };

  // ==================== COSINE SIMILARITY ====================
  
  function cosineSimilarity(embedding1, embedding2) {
    if (!embedding1 || !embedding2) return 0;
    if (embedding1.length !== embedding2.length) return 0;
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      norm1 += embedding1[i] * embedding1[i];
      norm2 += embedding2[i] * embedding2[i];
    }
    
    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  // ==================== ATTENTION RANKER (JS PORT) ====================
  const AttentionRanker = (() => {
    const vocabSize = 8192;
    const embedDim = 128;
    const scale = Math.sqrt(embedDim);

    function hash32(str) {
      let h = 2166136261 >>> 0;
      for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return h >>> 0;
    }

    function tokenIdFor(word) {
      return hash32(word) % vocabSize;
    }

    // Deterministic pseudo-random embedding per token id
    function embeddingForTokenId(tokenId) {
      const vec = new Float32Array(embedDim);
      // simple hash-based generator
      let x = (tokenId + 1) >>> 0;
      for (let i = 0; i < embedDim; i++) {
        // xorshift32
        x ^= x << 13; x >>>= 0;
        x ^= x >>> 17; x >>>= 0;
        x ^= x << 5; x >>>= 0;
        // map to [-0.1, 0.1]
        vec[i] = ((x & 0xffff) / 0xffff) * 0.2 - 0.1;
      }
      return vec;
    }

    function tokenize(text) {
      return text.toLowerCase().split(/\s+/).filter(Boolean);
    }

    function meanEmbeddingForText(text) {
      const words = tokenize(text);
      if (words.length === 0) return new Float32Array(embedDim);
      const acc = new Float32Array(embedDim);
      for (const w of words) {
        const id = tokenIdFor(w);
        const e = embeddingForTokenId(id);
        for (let i = 0; i < embedDim; i++) acc[i] += e[i];
      }
      for (let i = 0; i < embedDim; i++) acc[i] /= words.length;
      return acc;
    }

    function attentionScore(text, queryVec) {
      const words = tokenize(text);
      if (words.length === 0) return { score: 0, weights: [] };
      const seqLen = words.length;
      const scores = new Float32Array(seqLen);
      const embeddings = new Array(seqLen);
      for (let i = 0; i < seqLen; i++) {
        const id = tokenIdFor(words[i]);
        const e = embeddingForTokenId(id);
        embeddings[i] = e;
        // scaled dot with query
        let dp = 0;
        for (let j = 0; j < embedDim; j++) dp += e[j] * queryVec[j];
        scores[i] = dp / scale;
      }
      // softmax
      let max = -Infinity; for (let i = 0; i < seqLen; i++) if (scores[i] > max) max = scores[i];
      let sum = 0; for (let i = 0; i < seqLen; i++) { scores[i] = Math.exp(scores[i] - max); sum += scores[i]; }
      const weights = new Float32Array(seqLen);
      for (let i = 0; i < seqLen; i++) weights[i] = scores[i] / (sum || 1);
      // context vector
      const ctx = new Float32Array(embedDim);
      for (let i = 0; i < seqLen; i++) {
        const w = weights[i];
        const e = embeddings[i];
        for (let j = 0; j < embedDim; j++) ctx[j] += e[j] * w;
      }
      // cosine(context, query)
      return { score: cosineSimilarity(ctx, queryVec), weights };
    }

    function buildQueryVector(queryText) {
      return meanEmbeddingForText(queryText);
    }

    return { buildQueryVector, attentionScore };
  })();

  // Network interception layer
  function interceptNetworkRequests() {
    // Store original methods
    const originalFetch = window.fetch;
    const originalXHR = XMLHttpRequest.prototype.open;

    // Intercept fetch
    window.fetch = async function(...args) {
      const [url] = args;
      const response = await originalFetch.apply(this, args);
      
      // Check if this is a timeline request
      if (CONFIG.apiEndpoints.some(endpoint => url.includes(endpoint))) {
        const clonedResponse = response.clone();
        
        try {
          const data = await clonedResponse.json();
          processPostData(data);
        } catch (error) {
          console.error('[X Feed Curator] Failed to parse response:', error);
        }
      }
      
      return response;
    };

    // Intercept XMLHttpRequest
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
      const xhr = this;
      
      if (CONFIG.apiEndpoints.some(endpoint => url.includes(endpoint))) {
        const originalOnLoad = xhr.onload;
        xhr.onload = function() {
          try {
            const data = JSON.parse(xhr.responseText);
            processPostData(data);
          } catch (error) {
            console.error('[X Feed Curator] Failed to parse XHR response:', error);
          }
          
          if (originalOnLoad) {
            originalOnLoad.apply(xhr, arguments);
          }
        };
      }
      
      return originalXHR.apply(this, [method, url, ...rest]);
    };
  }

  // Extract posts from API response
  function extractPosts(data) {
    const posts = [];
    
    // Recursive function to find tweet objects
    function findTweets(obj) {
      if (!obj || typeof obj !== 'object') return;
      
      // Check for tweet-like objects
      if (obj.full_text || obj.text || obj.legacy?.full_text) {
        posts.push({
          id: obj.id_str || obj.rest_id || Math.random().toString(),
          text: obj.full_text || obj.text || obj.legacy?.full_text || '',
          user: obj.user?.screen_name || obj.core?.user_results?.result?.legacy?.screen_name || 'unknown'
        });
      }
      
      // Recursively search
      Object.values(obj).forEach(value => {
        if (Array.isArray(value)) {
          value.forEach(item => findTweets(item));
        } else if (typeof value === 'object') {
          findTweets(value);
        }
      });
    }
    
    findTweets(data);
    return posts;
  }

  // Process post data
  function processPostData(data) {
    const posts = extractPosts(data);
    
    if (posts.length === 0) return;
    
    // Filter out already processed posts
    const newPosts = posts.filter(post => !state.processedPosts.has(post.id));
    
    if (newPosts.length === 0) return;
    
    // Add to batch
    state.postBatch.push(...newPosts);
    
    // Process batch if it's large enough
    if (state.postBatch.length >= CONFIG.batchSize) {
      processBatch();
    } else {
      // Schedule batch processing with debounce
      clearTimeout(state.batchTimeout);
      state.batchTimeout = setTimeout(processBatch, 500);
    }
  }

  // Process batch of posts with attention-ready caching
  async function processBatch() {
    if (state.postBatch.length === 0) return;
    
    const batch = state.postBatch.splice(0, CONFIG.batchSize);
    console.log(`[X Feed Curator] Processing batch of ${batch.length} posts`);
    
    // Cache text for all posts in batch
    for (const post of batch) {
      try {
        state.processedPosts.set(post.id, { text: post.text, hidden: false });
        
        state.stats.processed++;
      } catch (error) {
        console.error('[X Feed Curator] Failed to process post:', error);
      }
    }
    
    // Apply ranking and filtering
    await applyRankedFiltering();
  }

  // ==================== RANKED FILTERING ====================
  
  async function applyRankedFiltering() {
    if (state.filters.length === 0) {
      // No filters: show all posts
      for (const [postId, postData] of state.processedPosts) {
        if (postData.hidden) {
          showPost(postId);
          postData.hidden = false;
        }
      }
      state.stats.hidden = 0;
      sendMessageToPopup({ type: 'hiddenCount', value: 0 });
      return;
    }
    
    // For each filter, compute attention-based relevance and apply Top-K
    for (const filter of state.filters) {
      const similarities = [];
      
      // Compute similarity scores for all posts
      for (const [postId, postData] of state.processedPosts) {
        const { score } = AttentionRanker.attentionScore(postData.text, filter.queryVec);
        similarities.push({ postId, similarity: score, postData });
      }
      
      // Sort by similarity (descending)
      similarities.sort((a, b) => b.similarity - a.similarity);
      
      if (filter.mode === 'hide') {
        // Hide posts matching the query (high similarity)
        let hiddenCount = 0;
        for (const item of similarities) {
          if (item.similarity >= filter.threshold) {
            if (!item.postData.hidden) {
              hidePost(item.postId);
              item.postData.hidden = true;
              hiddenCount++;
            }
          }
        }
      } else if (filter.mode === 'show') {
        // Show only top K matching posts
        const topK = filter.topK || 20;
        const topPosts = new Set(similarities.slice(0, topK).map(s => s.postId));
        
        for (const [postId, postData] of state.processedPosts) {
          if (topPosts.has(postId)) {
            if (postData.hidden) {
              showPost(postId);
              postData.hidden = false;
            }
          } else {
            if (!postData.hidden) {
              hidePost(postId);
              postData.hidden = true;
            }
          }
        }
      }
    }
    
    // Update hidden count
    let hiddenCount = 0;
    for (const [, postData] of state.processedPosts) {
      if (postData.hidden) hiddenCount++;
    }
    state.stats.hidden = hiddenCount;
    sendMessageToPopup({ type: 'hiddenCount', value: hiddenCount });
  }

  // Hide/show posts inline for immediate effect
  function hidePost(postId) {
    requestAnimationFrame(() => {
      const linkSelector = `a[href*="/status/${postId}"]`;
      document.querySelectorAll(linkSelector).forEach(link => {
        const article = link.closest('article');
        if (article) {
          if (!state.enabled) return;
          article.style.display = 'none';
          article.setAttribute('data-xfc-hidden', '1');
        }
      });
    });
  }

  function showPost(postId) {
    requestAnimationFrame(() => {
      const linkSelector = `a[href*="/status/${postId}"]`;
      document.querySelectorAll(linkSelector).forEach(link => {
        const article = link.closest('article');
        if (article) {
          article.style.display = '';
          article.removeAttribute('data-xfc-hidden');
        }
      });
    });
  }

  // ==================== COMMAND PARSING ====================
  
  async function parseCommand(command) {
    const text = command.trim();
    if (!text) return null;
    
    // Determine mode and extract query
    let mode = 'hide'; // default
    let query = text;
    let topK = 20;
    let threshold = 0.5;
    
    // Parse "hide <query>" or "remove <query>"
    if (text.toLowerCase().startsWith('hide ')) {
      mode = 'hide';
      query = text.substring(5).trim();
      threshold = 0.5; // Hide posts with similarity >= 0.5
    } else if (text.toLowerCase().startsWith('remove ')) {
      mode = 'hide';
      query = text.substring(7).trim();
      threshold = 0.5;
    }
    // Parse "show <query>" or "only <query>" - show only matching posts
    else if (text.toLowerCase().startsWith('show ') || text.toLowerCase().startsWith('only ')) {
      mode = 'show';
      const startIdx = text.toLowerCase().startsWith('show ') ? 5 : 5;
      query = text.substring(startIdx).trim();
      
      // Extract topK if specified: "show top 10 tech posts"
      const topKMatch = query.match(/top\s+(\d+)/i);
      if (topKMatch) {
        topK = parseInt(topKMatch[1]);
        query = query.replace(/top\s+\d+/i, '').trim();
      }
    }
    // Parse "only show <query>"
    else if (text.toLowerCase().includes('only show ')) {
      mode = 'show';
      query = text.toLowerCase().split('only show ')[1].trim();
    }
    
    // Build query vector for attention
    try {
      const queryVec = AttentionRanker.buildQueryVector(query);
      return { query, mode, queryVec, topK, threshold };
    } catch (error) {
      console.error('[X Feed Curator] Failed to build query vector:', error);
      throw error;
    }
  }

  // Communication with popup
  function sendMessageToPopup(message) {
    chrome.runtime.sendMessage(message).catch(() => {
      // Popup might not be open
    });
  }

  // ==================== MESSAGE HANDLING ====================
  
  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const handleAsync = async () => {
      switch (request.type) {
        case 'addFilter':
          try {
            const filter = await parseCommand(request.command);
            if (filter) {
              state.filters.push(filter);
              // Note: Can't store Float32Array in chrome.storage, only store metadata
              const filterMetadata = state.filters.map(f => ({
                query: f.query,
                mode: f.mode,
                topK: f.topK,
                threshold: f.threshold
              }));
              await chrome.storage.local.set({ filterMetadata });
              await applyRankedFiltering();
              sendResponse({ success: true, filters: filterMetadata });
            } else {
              sendResponse({ success: false, error: 'Invalid command' });
            }
          } catch (error) {
            sendResponse({ success: false, error: error.message });
          }
          break;
          
        case 'removeFilter':
          state.filters = state.filters.filter((f, i) => i !== request.index);
          const filterMetadata = state.filters.map(f => ({
            query: f.query,
            mode: f.mode,
            topK: f.topK,
            threshold: f.threshold
          }));
          await chrome.storage.local.set({ filterMetadata });
          await applyRankedFiltering();
          sendResponse({ success: true, filters: filterMetadata });
          break;
          
        case 'getStatus':
          const filterMetadataStatus = state.filters.map(f => ({
            query: f.query,
            mode: f.mode,
            topK: f.topK,
            threshold: f.threshold
          }));
          sendResponse({
            filters: filterMetadataStatus,
            stats: state.stats,
            enabled: state.enabled
          });
          break;
          
        case 'rescan':
          await applyRankedFiltering();
          sendResponse({ success: true });
          break;
          
        case 'toggle':
          state.enabled = !!request.enabled;
          if (!state.enabled) {
            // Show all posts
            for (const [postId, postData] of state.processedPosts) {
              if (postData.hidden) {
                showPost(postId);
                postData.hidden = false;
              }
            }
            state.stats.hidden = 0;
            sendMessageToPopup({ type: 'hiddenCount', value: 0 });
          } else {
            await applyRankedFiltering();
          }
          sendResponse({ success: true, enabled: state.enabled });
          break;
          
        case 'reset':
          state.filters = [];
          state.stats = { processed: 0, hidden: 0 };
          await chrome.storage.local.clear();
          // Show all posts
          for (const [postId, postData] of state.processedPosts) {
            if (postData.hidden) {
              showPost(postId);
              postData.hidden = false;
            }
          }
          sendMessageToPopup({ type: 'hiddenCount', value: 0 });
          sendResponse({ success: true });
          break;
      }
    };
    
    handleAsync();
    return true; // Keep message channel open for async response
  });

  // ==================== INITIALIZATION ====================
  
  async function init() {
    console.log('[X Feed Curator] Initializing attention-based filtering...');
    
    // Load saved filter metadata and regenerate embeddings
    const stored = await chrome.storage.local.get(['filterMetadata']);
    if (stored.filterMetadata && stored.filterMetadata.length > 0) {
      console.log('[X Feed Curator] Restoring filters...');
      // Regenerate embeddings for saved queries
      for (const meta of stored.filterMetadata) {
        try {
          const filter = await parseCommand(meta.query);
          if (filter) {
            filter.topK = meta.topK;
            filter.threshold = meta.threshold;
            state.filters.push(filter);
          }
        } catch (error) {
          console.error('[X Feed Curator] Failed to restore filter:', error);
        }
      }
    }
    
    // Start network interception
    interceptNetworkRequests();
    
    console.log('[X Feed Curator] Ready - Attention-based filtering active');
  }

  // Start extension
  init();
})();
