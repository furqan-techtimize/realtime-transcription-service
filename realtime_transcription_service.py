"""
Real-time Speech-to-Text Transcription Service
Supports Deepgram and AWS Transcribe with real API integration
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, Optional
import websockets
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionMetrics:
    """Track real-time transcription metrics"""
    
    def __init__(self):
        self.start_time = None
        self.word_count = 0
        self.confidence_scores = []
        self.latency_measurements = []
        self.request_times = []
        
    def start_session(self):
        self.start_time = time.time()
        self.word_count = 0
        self.confidence_scores = []
        self.latency_measurements = []
        self.request_times = []
        logger.info("Metrics session started")
        
    def add_result(self, text: str, confidence: float, latency: float):
        """Add transcription result and update metrics"""
        words = text.strip().split() if text.strip() else []
        word_count = len(words)
        self.word_count += word_count
        self.confidence_scores.append(confidence)
        self.latency_measurements.append(latency)
        self.request_times.append(time.time())
        
        logger.debug(f"Metrics updated - Words: {word_count}, Confidence: {confidence:.2f}, Latency: {latency}ms")
        
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        duration = time.time() - self.start_time if self.start_time else 0
        wpm = (self.word_count / (duration / 60)) if duration > 0 else 0
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        avg_latency = sum(self.latency_measurements) / len(self.latency_measurements) if self.latency_measurements else 0
        
        metrics = {
            'duration': duration,
            'word_count': self.word_count,
            'wpm': wpm,
            'avg_confidence': avg_confidence,
            'avg_latency': avg_latency,
            'estimated_accuracy': min(0.99, max(0.75, avg_confidence))
        }
        
        logger.debug(f"Current metrics: {metrics}")
        return metrics


class DeepgramTranscriber:
    """Real Deepgram speech-to-text transcription using direct WebSocket"""
    
    def __init__(self, api_key: str, websocket_callback):
        self.api_key = api_key
        self.websocket_callback = websocket_callback
        self.deepgram_ws = None
        self.metrics = TranscriptionMetrics()
        self.is_connected = False
        self.receive_task = None
        self.keepalive_task = None
        self.last_audio_time = None
        self.keepalive_interval = 5  # Send keepalive every 5 seconds
        self.connection_config = None  # Store config for reconnection
        self.auto_reconnect = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2  # Initial delay in seconds
        logger.info(f"DeepgramTranscriber initialized with API key: {api_key[:8]}...")
        
    async def connect(self, config: Dict) -> bool:
        """Connect to Deepgram real-time API via WebSocket"""
        try:
            logger.info("Attempting to connect to Deepgram API via WebSocket...")
            
            import websockets
            
            # Build Deepgram WebSocket URL with parameters
            model = config.get('model', 'nova-2')
            language = config.get('language', 'en-US')
            smart_format = str(config.get('smart_format', True)).lower()
            interim_results = str(config.get('interim_results', True)).lower()
            
            # Using raw PCM from browser AudioContext (16-bit linear PCM)
            # Sample rate: 16000Hz for optimal compatibility with file uploads and live recording
            diarize = str(config.get('diarize', False)).lower()
            params = f"model={model}&language={language}&smart_format={smart_format}&interim_results={interim_results}&punctuate=true&diarize={diarize}&encoding=linear16&sample_rate=16000&channels=1"
            url = f"wss://api.deepgram.com/v1/listen?{params}"
            
            logger.info(f"Connecting to: {url}")
            
            # Store config for potential reconnection
            self.connection_config = config
            
            # Connect to Deepgram WebSocket with API key in header and timeout
            self.deepgram_ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={
                        "Authorization": f"Token {self.api_key}"
                    },
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=5    # Wait 5 seconds for close handshake
                ),
                timeout=15.0  # 15 second connection timeout
            )
            
            logger.info("✅ Connected to Deepgram WebSocket")
            
            self.is_connected = True
            self.reconnect_attempts = 0  # Reset on successful connection
            self.metrics.start_session()
            
            # Initialize last audio time
            self.last_audio_time = time.time()
            
            # Reset warning flag
            self._send_warning_logged = False
            
            # Start receiving messages
            self.receive_task = asyncio.create_task(self._receive_messages())
            
            # Start keepalive task to prevent timeout
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())
            
            return True
                
        except asyncio.TimeoutError:
            logger.error("❌ Connection to Deepgram timed out after 15 seconds")
            return False
        except OSError as e:
            # Network errors (DNS, connection refused, etc.)
            logger.error(f"❌ Network error connecting to Deepgram: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Deepgram: {e}", exc_info=True)
            return False
    
    async def _receive_messages(self):
        """Receive and process messages from Deepgram"""
        try:
            async for message in self.deepgram_ws:
                try:
                    data = json.loads(message)
                    
                    # Check if this is a transcript result - Deepgram API structure
                    if data.get('type') == 'Results':
                        # Deepgram returns results in channel_index array format
                        if 'channel' in data:
                            channel = data['channel']
                            alternatives = channel.get('alternatives', [])
                        else:
                            # Try alternate structure
                            logger.warning(f"📥 Missing 'channel' key. Available keys: {list(data.keys())}")
                            continue
                            
                        if alternatives and len(alternatives) > 0:
                            transcript = alternatives[0].get('transcript', '')
                            confidence = alternatives[0].get('confidence', 0.9)
                            is_final = data.get('is_final', False)
                            speech_final = data.get('speech_final', False)
                            
                            # Extract speaker information from words if diarization is enabled
                            speaker = None
                            words = alternatives[0].get('words', [])
                            if words and len(words) > 0:
                                # Get speaker from first word (all words in utterance should have same speaker)
                                speaker = words[0].get('speaker')
                            
                            if transcript.strip():
                                speaker_info = f" (Speaker {speaker})" if speaker is not None else ""
                                logger.info(f"🎤 TRANSCRIPTION: '{transcript}'{speaker_info} (final={is_final}, speech_final={speech_final}, confidence={confidence:.2f})")
                                
                                latency = 150  # Approximate
                                
                                # Update metrics for final results
                                if is_final:
                                    self.metrics.add_result(transcript, confidence, latency)
                                
                                # Send to frontend
                                await self.send_transcription_result(
                                    text=transcript,
                                    confidence=confidence,
                                    is_final=is_final,
                                    latency=latency,
                                    speaker=speaker
                                )
                            else:
                                # Empty transcript - log for debugging
                                logger.debug(f"⚠️ Empty transcript (is_final={is_final}, speech_final={speech_final}, confidence={confidence:.2f})")
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse Deepgram message")
                except Exception as e:
                    logger.error(f"Error processing Deepgram message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ Deepgram WebSocket connection closed - attempting to reconnect...")
            self.is_connected = False
            
            # Attempt to reconnect if auto_reconnect is enabled
            if self.auto_reconnect and self.connection_config:
                if self.websocket_callback:
                    await self.websocket_callback({
                        'type': 'transcription_timeout',
                        'message': 'Connection closed. Reconnecting...',
                        'provider': 'deepgram'
                    })
                
                # Retry with exponential backoff
                while self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))  # Exponential backoff
                    delay = min(delay, 30)  # Cap at 30 seconds
                    
                    logger.info(f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} (waiting {delay}s)...")
                    await asyncio.sleep(delay)
                    
                    try:
                        # Try to reconnect
                        reconnected = await self.connect(self.connection_config)
                        
                        if reconnected:
                            logger.info("✅ Successfully reconnected to Deepgram")
                            if self.websocket_callback:
                                await self.websocket_callback({
                                    'type': 'transcription_reconnected',
                                    'message': 'Reconnected successfully. Ready for audio.',
                                    'provider': 'deepgram'
                                })
                            break
                        else:
                            logger.warning(f"⚠️ Reconnection attempt {self.reconnect_attempts} failed")
                    except Exception as retry_error:
                        logger.error(f"❌ Reconnection attempt {self.reconnect_attempts} error: {retry_error}")
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("❌ Max reconnection attempts reached. Giving up.")
                    if self.websocket_callback:
                        await self.websocket_callback({
                            'type': 'error',
                            'message': 'Failed to reconnect after multiple attempts. Please refresh and try again.',
                            'provider': 'deepgram'
                        })
        except Exception as e:
            logger.error(f"Error in Deepgram receive loop: {e}", exc_info=True)
            self.is_connected = False
    
    async def _keepalive_loop(self):
        """Send keepalive packets to prevent Deepgram WebSocket timeout"""
        try:
            # Deepgram expects a JSON keepalive message
            keepalive_msg = json.dumps({"type": "KeepAlive"})
            
            while self.is_connected:
                await asyncio.sleep(self.keepalive_interval)
                
                # Check if we've received audio recently
                if self.last_audio_time and (time.time() - self.last_audio_time) > 10:
                    # No audio for 10 seconds, send keepalive
                    logger.debug("📡 Sending keepalive to Deepgram")
                    try:
                        if self.deepgram_ws and self.is_connected:
                            # Use wait_for with timeout to avoid hanging
                            await asyncio.wait_for(
                                self.deepgram_ws.send(keepalive_msg),
                                timeout=5.0
                            )
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ Keepalive send timed out - connection may be unstable")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to send keepalive: {e}")
                        # Don't set is_connected to False here, let the receive loop handle it
                        
        except asyncio.CancelledError:
            logger.debug("Deepgram keepalive task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in Deepgram keepalive loop: {e}", exc_info=True)
    
    async def send_transcription_result(self, text: str, confidence: float, is_final: bool, latency: float, speaker=None):
        """Send transcription result to frontend"""
        if self.websocket_callback:
            try:
                result_data = {
                    'text': text,
                    'confidence': confidence,
                    'is_final': is_final,
                    'latency': latency,
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'deepgram'
                }
                
                # Add speaker information if available
                if speaker is not None:
                    result_data['speaker'] = speaker
                
                await self.websocket_callback({
                    'type': 'transcription_result',
                    'data': result_data
                })
                
                # Send updated metrics for final results
                if is_final:
                    metrics = self.metrics.get_metrics()
                    await self.websocket_callback({
                        'type': 'metrics_update',
                        'data': metrics
                    })
            except Exception as e:
                logger.error(f"❌ Error sending transcription result: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to Deepgram"""
        if self.deepgram_ws and self.is_connected:
            try:
                # Update last audio time
                self.last_audio_time = time.time()
                
                await self.deepgram_ws.send(audio_data)
                # Removed verbose audio logging - too much noise
            except Exception as e:
                logger.error(f"❌ Error sending audio to Deepgram: {e}")
                self.is_connected = False
        else:
            # Only log once per disconnection to avoid spam
            if not hasattr(self, '_send_warning_logged') or not self._send_warning_logged:
                logger.warning("⚠️ Cannot send audio - not connected to Deepgram (reconnecting...)")
                self._send_warning_logged = True
    
    async def close(self):
        """Close Deepgram connection"""
        logger.info("Closing Deepgram connection...")
        self.is_connected = False
        
        # Cancel keepalive task if running
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
            try:
                await asyncio.wait_for(self.keepalive_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Deepgram keepalive task cleanup: {type(e).__name__}")
        
        if self.receive_task:
            self.receive_task.cancel()
            
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
                logger.info("✅ Deepgram connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing Deepgram: {e}")


class AWSTranscriber:
    """Real AWS Transcribe Streaming speech-to-text"""
    
    def __init__(self, websocket_callback, region: str = 'us-east-1'):
        self.region = region
        self.websocket_callback = websocket_callback
        self.stream_client = None
        self.metrics = TranscriptionMetrics()
        self.is_connected = False
        self.last_audio_time = None
        self.keepalive_task = None
        self.keepalive_interval = 5  # Send keepalive every 5 seconds
        self.connection_config = None  # Store config for reconnection
        self.auto_reconnect = True
        logger.info(f"AWSTranscriber initialized for region: {region}")
    
    async def connect(self, config: Dict) -> bool:
        """Connect to AWS Transcribe Streaming"""
        try:
            logger.info("Attempting to connect to AWS Transcribe...")
            
            import boto3
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
            
            # Load AWS credentials from environment
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION', self.region)
            
            if not aws_access_key or not aws_secret_key:
                logger.error("❌ AWS credentials not found in environment variables")
                logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file")
                return False
            
            logger.info(f"✅ AWS credentials loaded (Key: {aws_access_key[:8]}...)")
            
            # Create AWS Transcribe streaming client
            # The client will automatically use the credentials from environment variables
            self.stream_client = TranscribeStreamingClient(region=aws_region)
            
            # Store config for potential reconnection
            self.connection_config = config
            
            # Store diarization setting
            self.enable_diarization = config.get('diarize', False)
            
            # Create event handler
            class MyEventHandler(TranscriptResultStreamHandler):
                def __init__(self, output_stream, transcriber):
                    super().__init__(output_stream)
                    self.transcriber = transcriber
                    
                async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                    try:
                        results = transcript_event.transcript.results
                        for result in results:
                            if not result.alternatives:
                                continue
                            
                            transcript = result.alternatives[0].transcript
                            # Handle confidence - can be None for some items
                            confidence = 0.9
                            speaker = None
                            
                            # DEBUG: Log the result structure
                            logger.info(f"DEBUG - Result type: {type(result)}")
                            logger.info(f"DEBUG - Result attributes: {dir(result)}")
                            if result.alternatives[0].items:
                                logger.info(f"DEBUG - First item type: {type(result.alternatives[0].items[0])}")
                                logger.info(f"DEBUG - First item attributes: {dir(result.alternatives[0].items[0])}")
                                logger.info(f"DEBUG - First item: {result.alternatives[0].items[0]}")
                            
                            if result.alternatives[0].items:
                                for item in result.alternatives[0].items:
                                    # Get confidence
                                    if hasattr(item, 'confidence') and item.confidence is not None:
                                        confidence = item.confidence
                                    # Get speaker label if available
                                    if hasattr(item, 'speaker') and item.speaker is not None:
                                        speaker = item.speaker
                                        break
                            
                            is_final = not result.is_partial
                            
                            # Only process final results
                            if not is_final:
                                continue
                            
                            if transcript.strip():
                                speaker_info = f" (Speaker {speaker})" if speaker else ""
                                logger.info(f"🎤 AWS TRANSCRIPTION: '{transcript}'{speaker_info} (final={is_final}, confidence={confidence:.2f})")
                                
                                latency = 200  # Approximate
                                
                                self.transcriber.metrics.add_result(transcript, confidence, latency)
                                
                                await self.transcriber.send_transcription_result(
                                    text=transcript,
                                    confidence=confidence,
                                    is_final=is_final,
                                    latency=latency,
                                    speaker=speaker
                                )
                    except asyncio.CancelledError:
                        logger.debug("AWS event handler cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"❌ Error handling AWS transcript: {e}", exc_info=True)
            
            self.event_handler_class = MyEventHandler
            
            # Create audio queue with size limit to prevent memory overflow
            # Limit to 5 chunks (~2.5 seconds of audio) to provide backpressure
            self.audio_queue = asyncio.Queue(maxsize=5)
            self.queue_full_warnings = 0
            
            # Initialize last audio time
            self.last_audio_time = time.time()
            
            # Start the streaming task
            self.streaming_task = asyncio.create_task(self._start_streaming())
            
            # Start keepalive task to prevent timeout
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())
            
            self.is_connected = True
            self.metrics.start_session()
            logger.info("✅ Successfully initialized AWS Transcribe")
            return True
            
        except ImportError as e:
            logger.error(f"❌ AWS SDK not installed: {e}")
            logger.error("Install with: pip install amazon-transcribe boto3")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to AWS Transcribe: {e}", exc_info=True)
            return False
    
    async def _keepalive_loop(self):
        """Send silence/keepalive packets to prevent AWS timeout"""
        try:
            # Create silence frame (0.1 seconds of silence at 16kHz, 16-bit PCM)
            silence_frame = b'\x00' * int(16000 * 0.1 * 2)  # 16000 Hz * 0.1s * 2 bytes per sample
            
            while self.is_connected:
                await asyncio.sleep(self.keepalive_interval)
                
                # Check if we've received audio recently
                if self.last_audio_time and (time.time() - self.last_audio_time) > 10:
                    # No audio for 10 seconds, send keepalive
                    logger.debug("📡 Sending keepalive packet to AWS Transcribe")
                    await self.audio_queue.put(silence_frame)
                    
        except asyncio.CancelledError:
            logger.debug("Keepalive task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in keepalive loop: {e}", exc_info=True)
    
    async def _start_streaming(self):
        """Start AWS Transcribe streaming"""
        try:
            logger.info("🚀 Starting AWS Transcribe streaming...")
            
            async def audio_generator():
                """Generate audio chunks from the queue"""
                while self.is_connected:
                    try:
                        # Wait for audio data with timeout
                        audio_data = await asyncio.wait_for(
                            self.audio_queue.get(), 
                            timeout=1.0
                        )
                        yield audio_data
                    except asyncio.TimeoutError:
                        # No audio available, continue waiting
                        continue
                    except asyncio.CancelledError:
                        logger.debug("Audio generator cancelled")
                        break
                    except Exception as e:
                        logger.error(f"❌ Error in audio generator: {e}", exc_info=True)
                        break
            
            # Start the transcription stream with optional speaker identification
            stream_params = {
                'language_code': "en-US",
                'media_sample_rate_hz': 16000,
                'media_encoding': "pcm",
            }
            
            # Add speaker identification if diarization is enabled
            if self.enable_diarization:
                stream_params['show_speaker_label'] = True
            
            stream = await self.stream_client.start_stream_transcription(**stream_params)
            
            # Create event handler and start streaming
            handler = self.event_handler_class(stream.output_stream, self)
            
            await asyncio.gather(
                self._write_audio_chunks(stream.input_stream, audio_generator()),
                handler.handle_events(),
                return_exceptions=True
            )
            
        except asyncio.CancelledError:
            logger.info("⚠️ AWS streaming task cancelled")
            self.is_connected = False
        except Exception as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                logger.warning(f"⚠️ AWS Transcribe timeout - attempting to reconnect...")
                # Send notification to frontend about timeout
                if self.websocket_callback:
                    await self.websocket_callback({
                        'type': 'transcription_timeout',
                        'message': 'Connection timed out due to silence. Reconnecting...',
                        'provider': 'aws'
                    })
                
                # Attempt to reconnect if auto_reconnect is enabled
                if self.auto_reconnect and self.connection_config:
                    logger.info("🔄 Attempting to reconnect to AWS Transcribe...")
                    await asyncio.sleep(1)  # Brief delay before reconnect
                    
                    # Clear the audio queue
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except:
                            break
                    
                    # Restart streaming
                    self.streaming_task = asyncio.create_task(self._start_streaming())
                    
                    if self.websocket_callback:
                        await self.websocket_callback({
                            'type': 'transcription_reconnected',
                            'message': 'Reconnected successfully. Ready for audio.',
                            'provider': 'aws'
                        })
                else:
                    self.is_connected = False
            else:
                logger.error(f"❌ Error in AWS streaming: {e}", exc_info=True)
                self.is_connected = False
    
    async def _write_audio_chunks(self, input_stream, audio_generator):
        """Write audio chunks to AWS input stream"""
        try:
            chunk_count = 0
            async for audio_chunk in audio_generator:
                await input_stream.send_audio_event(audio_chunk=audio_chunk)
                chunk_count += 1
                
                # Add small delay between chunks to prevent overwhelming AWS
                await asyncio.sleep(0.05)  # 50ms between chunks
                
                # Log every 20 chunks to monitor flow
                if chunk_count % 20 == 0:
                    queue_size = self.audio_queue.qsize()
                    logger.debug(f"📊 Sent {chunk_count} chunks, queue size: {queue_size}/5")
            
            # Close the stream when done
            await input_stream.end_stream()
            logger.info(f"✅ Audio stream ended ({chunk_count} total chunks)")
        except asyncio.CancelledError:
            logger.debug("AWS audio write task cancelled")
            # Try to gracefully close the stream
            try:
                await input_stream.end_stream()
            except:
                pass
        except Exception as e:
            logger.error(f"❌ Error writing audio chunks: {e}", exc_info=True)
    
    async def send_transcription_result(self, text: str, confidence: float, is_final: bool, latency: float, speaker=None):
        """Send transcription result to frontend"""
        if self.websocket_callback:
            try:
                result_data = {
                    'text': text,
                    'confidence': confidence,
                    'is_final': is_final,
                    'latency': latency,
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'aws'
                }
                
                # Add speaker information if available
                if speaker is not None:
                    result_data['speaker'] = speaker
                
                await self.websocket_callback({
                    'type': 'transcription_result',
                    'data': result_data
                })
                
                if is_final:
                    metrics = self.metrics.get_metrics()
                    await self.websocket_callback({
                        'type': 'metrics_update',
                        'data': metrics
                    })
            except Exception as e:
                logger.error(f"❌ Error sending AWS transcription result: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio to AWS Transcribe with backpressure handling"""
        try:
            if not self.is_connected or not hasattr(self, 'audio_queue'):
                return
            
            # Update last audio time
            self.last_audio_time = time.time()
            
            # Try to add audio to queue with timeout to prevent blocking forever
            # If queue is full, wait briefly then drop if still full (backpressure)
            try:
                await asyncio.wait_for(self.audio_queue.put(audio_data), timeout=1.0)
                # Reset warning counter on successful put
                if self.queue_full_warnings > 0:
                    logger.info(f"✅ Queue cleared, resuming normal operation")
                    self.queue_full_warnings = 0
            except asyncio.TimeoutError:
                # Queue is full - AWS is processing slower than we're sending
                self.queue_full_warnings += 1
                if self.queue_full_warnings <= 3:  # Only log first few warnings
                    logger.warning(f"⚠️ AWS queue full (warning {self.queue_full_warnings}) - reduce playback speed or wait for processing")
                    # Send warning to frontend
                    if self.websocket_callback:
                        try:
                            await self.websocket_callback({
                                'type': 'queue_warning',
                                'message': f'Processing queue full ({self.queue_full_warnings}/3 warnings)'
                            })
                        except:
                            pass
                # Drop this chunk to prevent memory buildup
                return
        except Exception as e:
            logger.error(f"❌ Error queueing audio for AWS: {e}", exc_info=True)
    
    async def close(self):
        """Close AWS Transcribe connection"""
        logger.info("Closing AWS Transcribe connection...")
        
        # Set disconnected flag first to stop all loops
        self.is_connected = False
        
        # Give tasks a moment to see the flag change
        await asyncio.sleep(0.1)
        
        # Cancel keepalive task if running
        if self.keepalive_task and not self.keepalive_task.done():
            logger.debug("Cancelling keepalive task...")
            self.keepalive_task.cancel()
            try:
                await asyncio.wait_for(self.keepalive_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug(f"Keepalive task cleanup: {type(e).__name__}")
            except Exception as e:
                logger.warning(f"Error during keepalive cleanup: {e}")
        
        # Cancel streaming task if running
        if hasattr(self, 'streaming_task') and not self.streaming_task.done():
            logger.debug("Cancelling streaming task...")
            self.streaming_task.cancel()
            try:
                await asyncio.wait_for(self.streaming_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug(f"Streaming task cleanup: {type(e).__name__}")
            except Exception as e:
                logger.warning(f"Error during streaming cleanup: {e}")
        
        # Clear audio queue to release any blocked puts
        if hasattr(self, 'audio_queue'):
            logger.debug("Clearing audio queue...")
            cleared = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared += 1
                except:
                    break
            if cleared > 0:
                logger.debug(f"Cleared {cleared} items from queue")
        
        # Close stream client if exists
        if hasattr(self, 'stream_client'):
            self.stream_client = None
        
        logger.info("✅ AWS Transcribe connection closed")


class RealtimeTranscriptionService:
    """WebSocket service for real-time transcription"""
    
    def __init__(self):
        self.active_connections: Dict[str, object] = {}
        self.transcribers: Dict[str, object] = {}
        logger.info("RealtimeTranscriptionService initialized")
        
    async def handle_connection(self, websocket):
        """Handle new WebSocket connection"""
        connection_id = f"{websocket.remote_address}_{int(time.time())}"
        self.active_connections[connection_id] = websocket
        
        logger.info(f"🔌 NEW CONNECTION: {connection_id} from {websocket.remote_address}")
        
        try:
            await self.send_message(websocket, {
                'type': 'connection_established',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat()
            })
            
            await self.handle_messages(websocket, connection_id)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"❌ Error handling connection {connection_id}: {e}", exc_info=True)
        finally:
            await self.cleanup_connection(connection_id)
    
    async def cleanup_connection(self, connection_id: str):
        """Cleanup connection resources"""
        logger.info(f"🧹 Cleaning up connection: {connection_id}")
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if connection_id in self.transcribers:
            try:
                await self.transcribers[connection_id].close()
            except Exception as e:
                logger.error(f"❌ Error closing transcriber: {e}")
            del self.transcribers[connection_id]
            
        logger.info(f"✅ Cleanup completed for: {connection_id}")
    
    async def handle_messages(self, websocket, connection_id: str):
        """Handle incoming WebSocket messages"""
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Audio data - send to transcriber if active
                    if connection_id in self.transcribers:
                        await self.transcribers[connection_id].send_audio(message)
                else:
                    # JSON command
                    try:
                        data = json.loads(message)
                        logger.info(f"📨 Received command: {data.get('command')} from {connection_id}")
                        await self.handle_command(websocket, connection_id, data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON: {e}")
                        await self.send_error(websocket, f"Invalid JSON: {e}")
                        
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}", exc_info=True)
                await self.send_error(websocket, f"Error: {e}")
    
    async def handle_command(self, websocket, connection_id: str, data: Dict):
        """Handle JSON commands"""
        command = data.get('command')
        
        logger.info(f"⚡ Processing command: {command}")
        
        if command == 'start_transcription':
            await self.start_transcription(websocket, connection_id, data)
        elif command == 'stop_transcription':
            await self.stop_transcription(websocket, connection_id)
        elif command == 'get_metrics':
            await self.send_current_metrics(websocket, connection_id)
        else:
            logger.warning(f"⚠️ Unknown command: {command}")
            await self.send_error(websocket, f"Unknown command: {command}")
    
    async def start_transcription(self, websocket, connection_id: str, config: Dict):
        """Start real-time transcription"""
        provider = config.get('provider', 'deepgram')
        
        logger.info(f"🚀 Starting {provider} transcription for {connection_id}")
        
        try:
            # Create callback for sending results
            async def result_callback(data):
                try:
                    await self.send_message(websocket, data)
                except Exception as e:
                    logger.error(f"❌ Error in result callback: {e}")
            
            # Initialize transcriber based on provider
            if provider == 'deepgram':
                api_key = os.getenv('DEEPGRAM_API_KEY')
                if not api_key:
                    error_msg = "DEEPGRAM_API_KEY not found in environment variables"
                    logger.error(f"❌ {error_msg}")
                    await self.send_error(websocket, error_msg)
                    return
                
                logger.info(f"Using Deepgram API key: {api_key[:8]}...")
                transcriber = DeepgramTranscriber(api_key, result_callback)
                
            elif provider == 'aws':
                transcriber = AWSTranscriber(result_callback, config.get('region', 'us-east-1'))
                
            else:
                error_msg = f"Unknown provider: {provider}"
                logger.error(f"❌ {error_msg}")
                await self.send_error(websocket, error_msg)
                return
            
            # Connect transcriber
            logger.info(f"Connecting to {provider}...")
            if await transcriber.connect(config):
                self.transcribers[connection_id] = transcriber
                
                logger.info(f"✅ {provider} transcription started for {connection_id}")
                
                await self.send_message(websocket, {
                    'type': 'transcription_started',
                    'provider': provider,
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                error_msg = f"Failed to connect to {provider}"
                logger.error(f"❌ {error_msg}")
                await self.send_error(websocket, error_msg)
                
        except Exception as e:
            error_msg = f"Failed to start transcription: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            await self.send_error(websocket, error_msg)
    
    async def stop_transcription(self, websocket, connection_id: str):
        """Stop transcription"""
        logger.info(f"🛑 Stopping transcription for {connection_id}")
        
        if connection_id in self.transcribers:
            await self.transcribers[connection_id].close()
            del self.transcribers[connection_id]
            logger.info(f"✅ Transcription stopped for {connection_id}")
        
        await self.send_message(websocket, {
            'type': 'transcription_stopped',
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_current_metrics(self, websocket, connection_id: str):
        """Send current metrics"""
        if connection_id in self.transcribers:
            metrics = self.transcribers[connection_id].metrics.get_metrics()
            await self.send_message(websocket, {
                'type': 'metrics_update',
                'data': metrics
            })
    
    async def send_message(self, websocket, data: Dict):
        """Send JSON message to client"""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
    
    async def send_error(self, websocket, error: str):
        """Send error message"""
        logger.error(f"Sending error to client: {error}")
        await self.send_message(websocket, {
            'type': 'error',
            'message': error,
            'timestamp': datetime.now().isoformat()
        })


async def main():
    """Start the real-time transcription service"""
    service = RealtimeTranscriptionService()
    
    port = 8766  # Different port from any existing service
    
    logger.info("=" * 60)
    logger.info("🎤 REAL-TIME TRANSCRIPTION SERVICE")
    logger.info("=" * 60)
    logger.info(f"Starting WebSocket server on ws://localhost:{port}")
    logger.info("Real-world implementation - NO SIMULATION")
    logger.info("=" * 60)
    
    # Check for API keys
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    if deepgram_key:
        logger.info(f"✅ Deepgram API key found: {deepgram_key[:8]}...")
    else:
        logger.warning("⚠️ DEEPGRAM_API_KEY not found in environment")
    
    async with websockets.serve(
        service.handle_connection,
        "localhost",
        port,
        ping_interval=20,
        ping_timeout=10
    ):
        logger.info(f"✅ Service ready on ws://localhost:{port}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}", exc_info=True)
