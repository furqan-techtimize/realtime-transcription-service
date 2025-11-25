# Real-Time Transcription Service

A WebSocket-based real-time speech-to-text transcription service supporting AWS Transcribe and Deepgram with speaker diarization.

## ✨ Features

- 🎤 **Real-time audio transcription** via WebSocket
- 🔊 **Dual provider support**: AWS Transcribe & Deepgram
- 👥 **Speaker diarization** - identifies different speakers
- 📁 **Audio file upload** - transcribe pre-recorded audio
- ⚡ **Speed control** - 0.5x to 4x playback speed
- 🔬 **Provider comparison** - side-by-side AWS vs Deepgram analysis
- 📊 **Live metrics** - confidence, latency, word count, WPM
- 🌐 **Web interface** - clean HTML5 demo page

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS account (for AWS Transcribe)
- Deepgram API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/realtime-transcription-service.git
cd realtime-transcription-service

# Install dependencies
pip install -r requirements_realtime_transcription.txt

# Copy environment template
cp example.env .env

# Edit .env with your API keys
# DEEPGRAM_API_KEY=your_key_here
# AWS_ACCESS_KEY_ID=your_key_here
# AWS_SECRET_ACCESS_KEY=your_key_here
# AWS_REGION=us-east-1
```

### Running the Service

```bash
# Start the WebSocket server
python start_realtime_transcription.py

# Open the demo page
# Open realtime_transcription_demo.html in your browser
# Or visit http://localhost:8765 if you navigate there
```

## 📖 Usage

### Live Recording

1. Click **"Start Recording"**
2. Speak into your microphone
3. See real-time transcription appear
4. Click **"Stop"** when done
5. Download transcript as TXT or JSON

### File Upload

1. Click **"Upload Audio File"** or drag & drop
2. Adjust playback speed (1x-4x for faster processing)
3. Select provider (AWS or Deepgram)
4. Click **"Transcribe"**
5. Download results

### Provider Comparison

1. Upload an audio file
2. Click **"🔬 Compare AWS vs Deepgram"**
3. System runs both transcriptions
4. Downloads detailed comparison report with:
   - Processing time
   - Accuracy metrics
   - Speaker diarization analysis
   - Side-by-side transcripts

## 🔧 Configuration

### Provider Settings

**AWS Transcribe:**
- Models: Standard, Medical, Call Center
- Best for: Medical/call center content
- Limitations: Memory issues on files > 2 minutes
- Recommended speed: 1x only

**Deepgram:**
- Models: Nova-3, Nova-2, Enhanced, Base
- Best for: General transcription, long files
- Supports: Up to 4x speed for file processing

### Environment Variables

```bash
# Required for Deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key

# Required for AWS Transcribe
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

## 🌐 API Reference

### WebSocket Messages

**Start Transcription:**
```json
{
  "command": "start_transcription",
  "provider": "deepgram",
  "model": "nova-3",
  "language": "en-US",
  "diarize": true,
  "smart_format": true
}
```

**Send Audio:**
```javascript
// Send PCM 16-bit audio data
ws.send(pcmAudioBuffer);
```

**Stop Transcription:**
```json
{
  "command": "stop_transcription"
}
```

**Receive Results:**
```json
{
  "type": "transcription_result",
  "data": {
    "text": "Hello world",
    "confidence": 0.95,
    "is_final": true,
    "speaker": 0,
    "timestamp": "2025-11-25T12:00:00"
  }
}
```

## 📁 Project Structure

```
realtime-transcription-service/
├── realtime_transcription_service.py  # Main WebSocket server
├── start_realtime_transcription.py    # Startup script
├── realtime_transcription_demo.html   # Web interface
├── requirements_realtime_transcription.txt  # Python dependencies
├── example.env                        # Environment variables template
└── README.md                          # This file
```

## 🐛 Known Issues

### AWS Transcribe Memory Errors

AWS Transcribe Streaming has memory limitations with large files:

- **Symptoms**: `MemoryError` or `AWS_ERROR_OOM` during transcription
- **Cause**: Audio sent faster than AWS can process
- **Solutions**:
  1. Use Deepgram for files > 2 minutes
  2. Reduce speed to 1x when using AWS
  3. System automatically prompts to switch providers

### Deepgram Connection Timeouts

If Deepgram disconnects after ~6 minutes:

- **Cause**: WebSocket keepalive timeout
- **Solution**: System automatically reconnects with exponential backoff
- **User action**: None required (handled automatically)

## 🚢 Deployment

### Render.com (Free)

1. Create account at [render.com](https://render.com)
2. New Web Service → Connect GitHub repo
3. Settings:
   - **Build Command**: `pip install -r requirements_realtime_transcription.txt`
   - **Start Command**: `python start_realtime_transcription.py`
4. Add environment variables in dashboard
5. Deploy!

### Frontend (GitHub Pages)

```bash
git checkout -b gh-pages
cp realtime_transcription_demo.html index.html
# Update WebSocket URL in index.html to your backend
git add index.html
git commit -m "Deploy frontend"
git push origin gh-pages
```

Enable in repo: Settings → Pages → Source: gh-pages

## 📊 Performance

| Provider | Speed | Accuracy | Diarization | Max File Length |
|----------|-------|----------|-------------|-----------------|
| **AWS Transcribe** | 1x only | High | ✅ Yes | ~2 minutes |
| **Deepgram Nova-3** | Up to 4x | Very High | ✅ Yes | 1+ hours |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **API Docs**: [AWS Transcribe](https://docs.aws.amazon.com/transcribe/) | [Deepgram](https://developers.deepgram.com/)

## 🙏 Acknowledgments

- AWS Transcribe SDK
- Deepgram Python SDK
- WebSockets library

---

Built with ❤️ for real-time transcription
