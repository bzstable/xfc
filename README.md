# X Feed Curator Chrome Extension

**Embedding-based semantic feed filtering for X (Twitter) using cosine similarity and neural embeddings.**

## Architecture

The extension implements a state-of-the-art semantic filtering system:

```
Posts → Embedding Model → Cosine Similarity with Query Vector → Ranked Filtering → Show Top K
```

### Components

1. **Network Interception**: Patches `fetch`/`XMLHttpRequest` to intercept X timeline API calls
2. **Batch Processing**: Extracts tweet data from JSON responses and processes in batches of 30
3. **Embedding Model**: Uses `all-MiniLM-L6-v2` via Transformers.js for semantic embeddings
4. **Cosine Similarity**: Computes similarity between post embeddings and query embeddings
5. **Ranked Filtering**: 
   - **Hide mode**: Hides posts with similarity ≥ threshold (default 0.5)
   - **Show mode**: Shows only top K most similar posts (default 20)

### Technical Stack

- **Model**: Xenova/all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Framework**: Transformers.js 2.17.2 (runs in browser via WASM)
- **Similarity Metric**: Cosine similarity with normalized embeddings
- **Storage**: Chrome Storage API (filter metadata only, embeddings regenerated on load)

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked"
4. Select this extension directory
5. Navigate to X.com or twitter.com
6. Wait for the embedding model to load (~3-5 seconds first time)

## Usage

### Filter Commands

**Hide posts similar to a query:**
```
hide sports posts
hide political content
remove cryptocurrency tweets
```

**Show only posts matching a query:**
```
show technology and AI posts
only machine learning content
show top 15 startup news
```

### Examples

1. **Hide sports**: `hide sports and cricket`
2. **Tech-only feed**: `show top 20 technology and AI posts`
3. **Remove politics**: `hide political discussions and elections`
4. **AI research only**: `show top 10 machine learning research papers`

### How It Works

1. You enter a natural language query
2. The model generates an embedding vector for your query
3. Each post in your feed is also converted to an embedding
4. Cosine similarity is computed between query and all posts
5. Posts are ranked by similarity score
6. Filtering is applied based on mode (hide/show) and parameters

## Features

- **Semantic Understanding**: Uses embeddings, not keywords
- **Real-time Processing**: Filters as posts load from API
- **Batch Efficient**: Processes 30 posts at once
- **No Server Required**: Runs entirely in-browser
- **Persistent Filters**: Saves your preferences
- **Top-K Control**: Choose how many posts to show
- **Threshold Control**: Adjust sensitivity for hiding

## Performance

- **Model Loading**: ~3-5 seconds (one-time, cached)
- **Embedding Generation**: ~50-100ms per post
- **Similarity Computation**: <1ms per comparison
- **Memory Usage**: ~150-200MB (model + embeddings)
- **No scroll lag**: Processing happens asynchronously

## Architecture Details

### State Management
```javascript
state = {
  filters: [{ query, mode, embedding, topK, threshold }],
  processedPosts: Map(postId → { text, embedding, hidden }),
  model: TransformersJS pipeline,
  modelLoaded: boolean
}
```

### Filter Processing Flow
```
1. User command → parseCommand()
2. Generate query embedding
3. Store filter with embedding
4. For each post:
   - Generate post embedding (if new)
   - Compute cosine similarity
5. Apply ranked filtering
6. Hide/show posts via DOM manipulation
```

## Files

- `manifest.json` - Extension configuration (v2.0)
- `content.js` - Core logic (544 lines)
  - Model loading
  - Embedding generation
  - Cosine similarity
  - Ranked filtering
  - Network interception
- `popup.html` - User interface
- `popup.js` - UI logic (245 lines)
- `styles.css` - X-themed design

## Development

### Testing Locally
1. Load extension in Chrome
2. Navigate to X.com
3. Open DevTools → Console
4. Watch for: `[X Feed Curator] Ready - Embedding-based filtering active`
5. Add a filter and observe console logs

### Debugging
- Model status shown in popup UI
- Console logs for batch processing
- Filter metadata visible in popup

## Limitations

- First load requires model download (~25MB, cached after)
- Works only on x.com and twitter.com
- Requires active filters to process posts
- Float32Array embeddings not persisted (regenerated on load)
