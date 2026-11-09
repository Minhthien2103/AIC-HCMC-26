
# Vector Search Service

A high-performance FastAPI service for vector similarity search using the CLIP model with Milvus database.

## Features

* **CLIP Model Support**: Uses OpenAI CLIP for text and image encoding
* **Temporal Queries**: Supports sequential queries with temporal relationship scoring
* **Real-time Communication**: WebSocket support for live queries
* **Configurable**: Easy configuration via JSON file or environment variables
* **High Performance**: Async operations, caching, and vectorized computations
* **Production Ready**: Comprehensive logging, error handling, and health checks

## Installation

### Requirements

```bash
# requirements.txt
fastapi
uvicorn[standard]
torch
torchvision
ope_clip_torch
transformers
pillow
pymilvus
numpy
pydantic
python-multipart
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Method 1: Configuration File

Create a `config.json` file (see example above) and set:

```bash
export CONFIG_FILE=config.json
```

### Method 2: Environment Variables

```bash
# Model Configuration
export CLIP_MODEL_NAME="ViT-H-14-378-quickgelu"
export CLIP_PRETRAINED="dfn5b"
export DEVICE="cuda"

# Database Configuration
export MILVUS_HOST="192.168.20.156"
export MILVUS_PORT="19530"
export MILVUS_DATABASE="default"
export COLLECTION_NAME="AIC_2024_1"
export SEARCH_LIMIT="3000"

# Server Configuration
export CORS_ORIGINS="http://localhost:8007,https://localhost:8005"
export MAX_WORKERS="4"
export LOG_LEVEL="INFO"
```

## Usage

### Starting the Service

```bash
# Using the refactored code
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

#### 2. Text Query (REST)

```bash
POST /TextQuery
Content-Type: application/json

{
  "First_query": "person walking",
  "Next_query": "person sitting"  // Optional
}
```

#### 3. WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  type: 'text_query',
  firstQuery: 'person walking',
  secondQuery: 'person sitting'
}));
```

#### 4. Configuration Info

```bash
GET /config
```

## File Structure

```
project/
├── main.py              # Main application file
├── config.json          # Configuration file
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .env                # Environment variables (optional)
```

## Configuration Options

| Setting           | Description                 | Default                |
| ----------------- | --------------------------- | ---------------------- |
| `clip_model_name` | CLIP model identifier       | ViT-H-14-378-quickgelu |
| `device`          | Computing device (cuda/cpu) | cuda                   |
| `milvus_host`     | Milvus server host          | 192.168.20.156         |
| `collection_name` | Milvus collection name      | AIC\_2024\_1           |
| `search_limit`    | Maximum search results      | 3000                   |
| `max_workers`     | Thread pool size            | 4                      |

## Temporal Query Logic

The service supports temporal queries where:

1. First query finds initial matches
2. Second query finds subsequent events
3. Results are reranked based on temporal proximity
4. Frame difference threshold: 1500 frames
5. Score boosting based on temporal closeness

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --log-level debug
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test text query
curl -X POST http://localhost:8000/TextQuery \
  -H "Content-Type: application/json" \
  -d '{"First_query": "person walking"}'
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: Service automatically falls back to CPU
2. **Milvus connection failed**: Check host/port configuration
3. **Model loading errors**: Ensure sufficient disk space and memory

### Logs

Check logs for detailed error information:

```bash
# Service logs include execution times and error details
# Set LOG_LEVEL=DEBUG for verbose logging
```

## License

This project is configured for easy deployment and maintenance in production environments.
