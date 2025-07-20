# Sentiment Analysis Microservice

A production-ready microservice for binary sentiment analysis featuring a beautiful React frontend, Python GraphQL backend, and fine-tuning capabilities.

## 🚀 Features

- **Advanced AI Models**: Powered by Hugging Face Transformers
- **Real-Time Analysis**: Instant sentiment predictions with confidence scores
- **Fine-Tunable**: Customize models with your own training data
- **Modern UI**: Beautiful React frontend with gradients, glass effects, and dark mode
- **GraphQL API**: Type-safe API with comprehensive schema
- **Docker Ready**: Fully containerized with Docker Compose
- **Production Ready**: Optimized for performance and scalability

## 📋 Tech Stack

### Backend

- **Python 3.9+**
- **FastAPI** with **Strawberry GraphQL**
- **Hugging Face Transformers**
- **PyTorch** (CPU-optimized)
- **Docker**

### Frontend

- **React 18** with **JavaScript**
- **Apollo GraphQL Client**
- **Tailwind CSS**
- **Vite** build tool
- **Lucide React** icons

## 🛠️ Setup & Installation

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd project

# Build and start all services
docker-compose up --build

# The application will be available at:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - GraphQL Playground: http://localhost:8000/graphql
```

### Local Development

#### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

#### Frontend Setup (from project root)

```bash
npm install
npm run dev
```

## 🎯 API Usage

### GraphQL Endpoints

**Health Check Query:**

```graphql
query {
  health
}
```

**Sentiment Prediction Mutation:**

```graphql
mutation PredictSentiment($text: String!) {
  predictSentiment(text: $text) {
    label
    score
  }
}
```

**Example Response:**

```json
{
  "data": {
    "predictSentiment": {
      "label": "positive",
      "score": 0.9234
    }
  }
}
```

## 🎓 Fine-Tuning

### Data Format

Create a JSONL file with training data:

```jsonl
{"text": "I love this product!", "label": "positive"}
{"text": "This is terrible", "label": "negative"}
{"text": "Amazing experience", "label": "positive"}
```

### Training Command

```bash
cd backend
python finetune.py data training_data.jsonl --epochs 3 --lr 3e-5
```

### Training Parameters

- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 3e-5)
- `--batch_size`: Batch size (default: 16)
- `--val_split`: Validation split ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

## 🏗️ Architecture

```
┌─────────────────┐    GraphQL     ┌─────────────────┐
│   React Frontend│ ──────────────→│  Python Backend │
│   (Port 3000)   │                │   (Port 8000)   │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
                                   ┌─────────────────┐
                                   │ Hugging Face    │
                                   │ Transformers    │
                                   └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

- Backend runs on port 8000
- Frontend runs on port 3000
- Model weights saved to `./backend/model/`
- CPU-only inference for maximum compatibility

### Model Configuration

- Default model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Custom models automatically loaded from `./backend/model/`
- Binary classification: positive/negative

## 📊 Performance

### Typical Response Times

- **Inference**: ~100-500ms (CPU)
- **Model Loading**: ~2-5 seconds
- **Fine-tuning**: ~10-30 minutes (depends on data size)

### Resource Usage

- **Memory**: ~1-2GB RAM
- **CPU**: Optimized for multi-core processors
- **Storage**: ~1GB for model weights

## 🎨 UI Features

- **Modern Design**: Glass morphism effects and gradients
- **Dark Mode**: Automatic theme switching
- **Responsive**: Works on all device sizes
- **Animated**: Smooth transitions and micro-interactions
- **Accessible**: ARIA labels and keyboard navigation

## 🚀 Production Deployment

### Docker Compose Production

```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: unless-stopped
```

### Health Checks

Both services include health check endpoints:

- Backend: `GET /health`
- Frontend: `GET /` (nginx status)

## 🔍 Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development (from project root)

```bash
npm install
npm run dev
```

### Code Quality

- ESLint configuration for TypeScript
- Type-safe GraphQL queries
- Comprehensive error handling
- Modular component architecture

## 📝 API Documentation

### GraphQL Schema

```graphql
type Query {
  health: String!
}

type Mutation {
  predictSentiment(text: String!): PredictionResult!
}

type PredictionResult {
  label: String!
  score: Float!
}
```

### Error Handling

The API returns structured errors with:

- Error messages
- Status codes
- Debugging information (development mode)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the GitHub Issues page
2. Review the API documentation
3. Ensure Docker is running properly
4. Verify network connectivity between services

---

**Built with ❤️ for production-ready sentiment analysis**
