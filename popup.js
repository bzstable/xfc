// X Feed Curator - Popup Script
// UI for embedding-based semantic filtering

// State
let currentFilters = [];
let enabled = true;
let hiddenCount = 0;

// DOM Elements
const elements = {
  filterInput: document.getElementById('filter-input'),
  addFilterBtn: document.getElementById('add-filter'),
  filtersList: document.getElementById('filters-list'),
  resetBtn: document.getElementById('reset-filters'),
  modelStatus: document.getElementById('model-status')
};

// Initialize popup
async function init() {
  await resolveLogo();
  // Get current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  // Check if we're on X
  if (!tab.url || (!tab.url.includes('x.com') && !tab.url.includes('twitter.com'))) {
    showError('Please navigate to X to use this extension');
    return;
  }
  
  // Get current status from content script
  chrome.tabs.sendMessage(tab.id, { type: 'getStatus' }, (response) => {
    if (response) {
      updateUI(response);
    }
  });
  
  // Set up event listeners
  setupEventListeners();
}

// Set up event listeners
function setupEventListeners() {
  // Add filter button
  elements.addFilterBtn.addEventListener('click', addFilter);
  
  // Enter key on input
  elements.filterInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      addFilter();
    }
  });
  
  // Reset button
  elements.resetBtn.addEventListener('click', resetFilters);

  // Toggle
  document.getElementById('toggle-enabled').addEventListener('change', async (e) => {
    enabled = e.target.checked;
    updateStatus();
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { type: 'toggle', enabled });
  });

  // Quick chips
  document.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      elements.filterInput.value = btn.dataset.cmd;
      addFilter();
    });
  });
}

// Add filter
async function addFilter() {
  const command = elements.filterInput.value.trim();
  
  if (!command) return;
  
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  // Show loading state
  elements.addFilterBtn.disabled = true;
  elements.addFilterBtn.textContent = 'Processing...';
  
  chrome.tabs.sendMessage(tab.id, { 
    type: 'addFilter', 
    command: command 
  }, (response) => {
    elements.addFilterBtn.disabled = false;
    elements.addFilterBtn.textContent = 'Add Filter';
    
    if (response && response.success) {
      elements.filterInput.value = '';
      updateFilters(response.filters);
    } else if (response && response.error) {
      alert('Error: ' + response.error);
    }
  });
}

// Remove filter
async function removeFilter(index) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { type: 'removeFilter', index }, (response) => {
    if (response && response.success) {
      updateFilters(response.filters);
      chrome.tabs.sendMessage(tab.id, { type: 'rescan' });
    }
  });
}

// Reset all filters
async function resetFilters() {
  if (!confirm('Reset all filters and statistics?')) return;
  
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  chrome.tabs.sendMessage(tab.id, { type: 'reset' }, (response) => {
    if (response && response.success) {
      updateUI({ filters: [] });
    }
  });
}

// Update UI with current state
function updateUI(state) {
  if (state.filters) {
    updateFilters(state.filters);
  }
  if (typeof state.enabled === 'boolean') {
    enabled = state.enabled;
    document.getElementById('toggle-enabled').checked = enabled;
    updateStatus();
  }
  // Static status for attention-based local model
  updateModelStatus('loaded');
}

// Update filters display
function updateFilters(filters) {
  currentFilters = filters;
  
  if (filters.length === 0) {
    elements.filtersList.innerHTML = '<div class="empty-state">No active filters</div>';
    return;
  }
  
  elements.filtersList.innerHTML = filters.map((filter, index) => {
    const action = filter.mode === 'hide' ? 'Hide' : 'Show only';
    const query = filter.query || 'unknown';
    const details = filter.mode === 'show' ? ` (top ${filter.topK})` : ` (threshold: ${filter.threshold})`;
    
    return `
      <div class="filter-item">
        <div class="filter-text">
          <span class="filter-action">${action}</span>
          <span class="filter-categories">"${query}"${details}</span>
        </div>
        <button class="remove-filter" data-index="${index}">Remove</button>
      </div>
    `;
  }).join('');

  // event delegation for remove buttons
  elements.filtersList.onclick = (e) => {
    const btn = e.target.closest('.remove-filter');
    if (!btn) return;
    const idx = Number(btn.getAttribute('data-index'));
    if (!Number.isNaN(idx)) removeFilter(idx);
  };
}

function updateStatus() {
  const text = enabled ? 'Filtering active' : 'Filtering paused';
  document.getElementById('status-text').textContent = text;
}

// Show error
function showError(message) {
  elements.filtersList.innerHTML = `
    <div class="empty-state" style="color: #F91880;">
      ${message}
    </div>
  `;
  elements.addFilterBtn.disabled = true;
  elements.filterInput.disabled = true;
}

// Update model status indicator
function updateModelStatus(status, error = null) {
  modelStatus = status;
  
  if (!elements.modelStatus) return;
  
  elements.modelStatus.className = status;
  
  switch (status) {
    case 'loading':
      elements.modelStatus.textContent = 'Loading model...';
      break;
    case 'loaded':
      elements.modelStatus.textContent = 'Model ready';
      break;
    case 'error':
      elements.modelStatus.textContent = error ? `Error: ${error}` : 'Model error';
      break;
    default:
      elements.modelStatus.textContent = 'Initializing...';
  }
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'hiddenCount') {
    hiddenCount = message.value || 0;
    document.getElementById('hidden-count').textContent = `${hiddenCount} posts hidden`;
  }
});

// Make removeFilter available globally for onclick
// no global export needed when using delegation

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

// Try multiple logo filenames to handle typos and formats
async function resolveLogo() {
  const el = document.getElementById('logo-img');
  if (!el) return;
  const candidates = ['logo.jepg', 'logo.jpeg', 'logo.jpg', 'logo.png'];
  for (const src of candidates) {
    const ok = await checkImage(src);
    if (ok) { el.src = src; return; }
  }
  // hide the broken image icon if nothing loads
  el.style.display = 'none';
}

function checkImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = src;
  });
}
