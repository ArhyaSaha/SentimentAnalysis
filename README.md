# Sentiment Analysis Microservice

Real-time sentiment analysis powered by a fine-tuned Twitter RoBERTa model for advanced emotional tone detection.
http://electronix.arhya.codes/

## 🎨 Frontend Features

- **Clean UI**: Modern React interface with glass morphism effects and gradients
- **Real-time Analysis**: Debounced input processing with instant sentiment predictions
- **Theme Toggle**: Dark/light mode switching with smooth transitions
- **Cool Result Design**: Circular progress meter with animated confidence scores and gradient text

## ⚡ Backend Features

- **Auto Model Loading**: Automatically reloads latest fine-tuned weights from `./model/` on service startup
- **Async Batching**: Asynchronous prediction handling with thread-safe model operations for `/predict` endpoint

## 🔧 Fine-tune Script

### Features

- **Model Quantization**: 4-bit quantization using bitsandbytes for memory efficiency
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **Training Loop**: Cross-entropy loss, gradient clipping, linear LR scheduler with warmup
- **Auto Save**: Weights saved to `./model/` (automatically picked up by API on restart)

### Arguments

```bash
python finetune.py data sample_data.jsonl --epochs 3 --lr 3e-5 --use_lora --use_quantization --lora_r 8 --batch_size 16 --val_split 0.2 --max_length 256
```

**Key Parameters:**

- `--use_lora`: Enable LoRA fine-tuning
- `--use_quantization`: Enable 4-bit quantization
- `--lora_r`: LoRA rank (default: 16)
- `--batch_size`: Training batch size (default: 16)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)

## 🐳 Docker

**Optimized Multi-stage Builds:**

- Frontend: Node.js build stage → Nginx production stage
- Backend: Python builder stage → Production stage with virtual environment
- Non-root users for security
- Layer caching optimization with `.dockerignore` files

## 🚀 Setup Instructions

### 1. Manual Setup

**Backend:**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend:**

```bash
npm install
npm run dev
```

### 2. Docker Compose

```bash
docker-compose up --build
```

Access:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 💡 Design Decisions

- **CPU-First**: Optimized for CPU inference to ensure broad compatibility
- **GraphQL**: Type-safe API with Apollo Client for efficient data fetching
- **Real-time UX**: Debounced input (1s delay) prevents excessive API calls
- **Parameter Efficiency**: LoRA enables fine-tuning with minimal memory footprint
- **Container Security**: Non-root users and resource limits in production builds
- **Hot Reload Ready**: Model handler designed for automatic weight reloading
- **Binary Classification**: Simplified positive/negative sentiment for clear results
