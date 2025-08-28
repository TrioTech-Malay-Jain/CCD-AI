# ttRag - Triotech Retrieval Augmented Generation System

A sophisticated AI-powered RAG system built with FastAPI, LangChain, and Google's Gemini LLM. The system supports multiple document formats and provides a comprehensive knowledge base management interface.

## 🚀 Features

### Core Functionality
- **Multi-format Document Support**: PDF, DOCX, TXT, and JSON files
- **Advanced RAG Pipeline**: Retrieval-Augmented Generation with LangChain
- **Vector Database**: ChromaDB for efficient document retrieval
- **Session Management**: Persistent chat history and context
- **Admin Panel**: Web-based knowledge base management
- **API Rate Limiting**: Multiple Google API key rotation
- **Real-time Processing**: Background vector database building

### AI Capabilities
- **Comprehensive Response System**: Detailed, structured answers with reasoning
- **Context-Aware Conversations**: Maintains chat history and context
- **Error Handling**: Graceful fallbacks for missing information
- **Multi-language Support**: Hindi-English (Hinglish) understanding

## 📁 Project Structure

```
ttRag/
├── 📄 main.py                     # FastAPI application (primary)
├── 🔧 build_db.py                 # Enhanced vector database builder
├── 🚀 run_fastapi.py              # FastAPI server runner
├── 🧪 create_test_files.py        # Test file generator
├── 📋 requirements.txt            # Python dependencies
├── 🔒 .env                        # Environment variables
├── 📖 README.md                   # This documentation
│
├── 📂 knowledge_base/             # Document storage
│   ├── 📄 *.txt                   # Text documents
│   ├── 📄 *.pdf                   # PDF documents
│   ├── 📄 *.docx                  # Word documents
│   └── 📄 *.json                  # JSON data files
│
├── 📂 chroma_db/                  # Vector database (auto-generated)
│   ├── 🗄️ chroma.sqlite3
│   └── 📊 vector data files
│
├── 📂 templates/                  # HTML templates
│   ├── 🏠 index.html              # Chat interface
│   └── ⚙️ admin.html              # Admin panel
│
├── 📂 static/                     # Static assets
│   ├── 🖼️ images/
│   ├── 🎨 css/
│   └── ⚡ js/
│
├── 📂 flask_structure/            # Legacy Flask implementation
│   ├── 🐍 run_flask.py            # Original Flask app
│   ├── 📋 requirements_flask.txt  # Flask dependencies
│   └── 📖 README.md               # Flask documentation
│
└── 📂 venv/                       # Virtual environment
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Google API Keys** (for Gemini LLM)
- **Virtual Environment** (recommended)

### 1. Clone Repository
```bash
git clone https://github.com/TrioTech-Malay-Jain/ttRag.git
cd ttRag
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:

```env
# Google API Keys (Multiple keys for rate limiting)
GOOGLE_API_KEY_1="your_primary_google_api_key_here"
GOOGLE_API_KEY_2="your_secondary_google_api_key_here"
GOOGLE_API_KEY_3="your_tertiary_google_api_key_here"
GOOGLE_API_KEY_4="your_quaternary_google_api_key_here"
GOOGLE_API_KEY_5="your_quinary_google_api_key_here"

# Application Security
SECRET_KEY="your_secret_key_for_sessions_here"

# Optional: Database Configuration
# CHROMA_DB_PATH="./chroma_db"
# KNOWLEDGE_BASE_DIR="./knowledge_base"
```

### 5. Setup Knowledge Base
```bash
# Create test files (optional)
python create_test_files.py

# Add your documents to knowledge_base/
# Supported formats: .txt, .pdf, .docx, .json
```

### 6. Build Vector Database
```bash
# Option 1: Command line
python build_db.py

# Option 2: Web interface (after starting server)
# Visit http://localhost:8000/admin
```

## 🚀 Deployment

### Local Development
```bash
# Start FastAPI server
python run_fastapi.py

# Or directly
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

#### Using Docker
```dockerfile
# Dockerfile (create this file)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ttrag .
docker run -p 8000:8000 --env-file .env ttrag
```

#### Cloud Deployment Options
- **Google Cloud Run**
- **AWS Lambda** (with Mangum adapter)
- **Azure Container Instances**
- **Heroku**
- **Railway**
- **Render**

## 🌐 API Endpoints

### Frontend Routes
- `GET /` - Chat interface
- `GET /admin` - Knowledge base management panel
- `GET /docs` - API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Knowledge Base Management
- `POST /api/upload` - Upload documents (.txt, .json, .pdf, .docx)
- `GET /api/files` - List uploaded files
- `DELETE /api/files/{filename}` - Delete specific file

### Vector Database
- `POST /api/build_db` - Build/rebuild vector database (async)
- `GET /api/build_status` - Check build progress

### Chat & Query
- `POST /api/chat` - Frontend chat endpoint
- `POST /api/query` - Direct API query endpoint
- `POST /ask` - Legacy compatibility endpoint

### Utilities
- `GET /api/health` - System health check
- `GET /api/chat_history/{session_id}` - Retrieve chat history
- `DELETE /api/chat_history/{session_id}` - Clear chat history

## 📝 Usage Examples

### Upload Documents via API
```python
import requests

# Upload a PDF file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f}
    )
    print(response.json())
```

### Query the AI via API
```python
import requests

response = requests.post(
    'http://localhost:8000/api/query',
    json={
        'query': 'Tell me about the event speakers',
        'session_id': 'user123',
        'history': []
    }
)
print(response.json())
```

### Build Vector Database via API
```python
import requests

# Start build process
response = requests.post('http://localhost:8000/api/build_db')
print(response.json())

# Check status
status = requests.get('http://localhost:8000/api/build_status')
print(status.json())
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY_1` | Primary Google API key | Yes |
| `GOOGLE_API_KEY_2-5` | Additional API keys for rate limiting | No |
| `SECRET_KEY` | Session security key | Yes |
| `CHROMA_DB_PATH` | Vector database path | No (default: ./chroma_db) |
| `KNOWLEDGE_BASE_DIR` | Documents directory | No (default: ./knowledge_base) |

### AI System Configuration
The AI system uses an enhanced prompt designed for comprehensive responses:

```python
system_prompt = """You are an advanced AI assistant with expertise in understanding and explaining complex information.
Your role is to answer user questions comprehensively using the provided knowledge base context.

Guidelines:
1. Always ground your answers in the provided context, but expand with reasoning, clarification, and related insights.
2. Provide clear, structured, and well-organized responses (use sections, bullet points, or lists where helpful).
3. Be detailed — explain concepts fully instead of giving short or vague replies.
4. Highlight key insights, important details, and actionable information.
5. If something is unclear in the context, infer the most likely explanation and explicitly state your assumptions.
6. If the information truly does not exist in the knowledge base, say: 
   "The available knowledge base does not provide a direct answer to this question," 
   and suggest possible directions or related knowledge.
"""
```

## 🛡️ Security Considerations

- **API Keys**: Store in environment variables, never commit to version control
- **File Uploads**: Validated file types and size limits
- **CORS**: Configure appropriately for production
- **Rate Limiting**: Implemented via multiple API key rotation
- **Input Validation**: All inputs are validated and sanitized

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **API Key Issues**
   ```bash
   # Check .env file exists and has valid keys
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY_1'))"
   ```

3. **Vector Database Issues**
   ```bash
   # Rebuild database
   python build_db.py
   ```

4. **Port Already in Use**
   ```bash
   # Use different port
   uvicorn main:app --port 8001
   ```

### Debug Mode
```bash
# Run with debug logging
uvicorn main:app --reload --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Malay Jain** - *Creator & Developer* - [TrioTech-Malay-Jain](https://github.com/TrioTech-Malay-Jain)
- SIRT College

## 📞 Support

For support, contact:
- **Phone**: 6232155888
- **Email**: Create an issue on GitHub
- **Event**: Google Cloud Community Day Bhopal

## 🎯 Roadmap

- [ ] Multi-language support enhancement
- [ ] Advanced document preprocessing
- [ ] Real-time collaborative features
- [ ] Mobile app development
- [ ] Voice interaction capabilities
- [ ] Analytics dashboard
- [ ] Custom model fine-tuning

---

## 🚀 Quick Start Commands

```bash
# Complete setup in one go
git clone https://github.com/TrioTech-Malay-Jain/ttRag.git
cd ttRag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Add your .env file
python create_test_files.py
python build_db.py
python run_fastapi.py
# Visit http://localhost:8000
```

**🎉 Happy Coding with ttRag!**

