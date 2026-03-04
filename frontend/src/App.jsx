import { useEffect, useRef, useState } from 'react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import styles from './App.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_ENDPOINT = `${API_BASE_URL}/api/v1/analyze`;
const STREAM_WORD_DELAY_MS = 60;
const PERMISSION_ERROR_MESSAGE = 'Camera or microphone permission is required.';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function formatAssistantResponse(data) {
  const emotion = data.fused_emotion || 'Unknown';
  const confidence = typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence ?? 'N/A';
  const responseText = data.empathetic_response || data.response_text || "I'm here with you.";

  return `Emotion detected: ${emotion} (Confidence: ${confidence})\n"${responseText}"`;
}

async function analyzePayload(payload) {
  const response = await fetch(API_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: 'frontend-session',
      ...payload,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Unable to analyze message right now.');
  }

  return response.json();
}

function stopStreamTracks(stream) {
  if (!stream) {
    return;
  }
  stream.getTracks().forEach((track) => track.stop());
}

function captureVideoFrame(videoElement) {
  if (!videoElement || !videoElement.videoWidth || !videoElement.videoHeight) {
    throw new Error('Camera is not ready yet. Please try again.');
  }

  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  const context = canvas.getContext('2d');
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL('image/png');
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hi, I'm here to listen. Tell me how you're feeling today.",
    },
  ]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [permissionError, setPermissionError] = useState('');

  const cameraVideoRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const audioStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const appendAssistantMessage = async (fullText) => {
    const messageId = Date.now() + Math.random();
    const words = fullText.split(/\s+/).filter(Boolean);

    setIsStreaming(true);
    setMessages((prev) => [...prev, { id: messageId, role: 'assistant', content: '' }]);

    let partialContent = '';
    for (const word of words) {
      partialContent = partialContent ? `${partialContent} ${word}` : word;
      setMessages((prev) =>
        prev.map((message) =>
          message.id === messageId
            ? {
                ...message,
                content: partialContent,
              }
            : message,
        ),
      );
      await delay(STREAM_WORD_DELAY_MS);
    }

    setIsStreaming(false);
  };

  const runAnalysis = async (payload) => {
    setIsAnalyzing(true);
    try {
      const data = await analyzePayload(payload);
      setIsAnalyzing(false);
      await appendAssistantMessage(formatAssistantResponse(data));
    } catch (error) {
      setIsAnalyzing(false);
      await appendAssistantMessage(`I ran into an issue while analyzing that message: ${error.message}`);
    }
  };

  const handleSend = async (text) => {
    const userMessage = { id: Date.now(), role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    await runAnalysis({ text });
  };

  const closeCamera = () => {
    stopStreamTracks(cameraStreamRef.current);
    cameraStreamRef.current = null;
    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
    setCameraOpen(false);
  };

  const handleOpenCamera = async () => {
    setPermissionError('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      closeCamera();
      cameraStreamRef.current = stream;
      setCameraOpen(true);

      requestAnimationFrame(() => {
        if (cameraVideoRef.current) {
          cameraVideoRef.current.srcObject = stream;
        }
      });
    } catch (error) {
      setPermissionError(PERMISSION_ERROR_MESSAGE);
    }
  };

  const handleCapture = async () => {
    if (!cameraVideoRef.current) {
      return;
    }

    try {
      const imageBase64 = captureVideoFrame(cameraVideoRef.current);
      const userMessage = { id: Date.now(), role: 'user', content: '[Image captured from camera]' };
      setMessages((prev) => [...prev, userMessage]);
      closeCamera();
      await runAnalysis({ image: imageBase64 });
    } catch (error) {
      await appendAssistantMessage(`I could not capture an image: ${error.message}`);
    }
  };

  const handleStartRecording = async () => {
    setPermissionError('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      audioStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        stopStreamTracks(audioStreamRef.current);
        audioStreamRef.current = null;
        mediaRecorderRef.current = null;
        audioChunksRef.current = [];

        const userMessage = { id: Date.now(), role: 'user', content: '[Voice message recorded]' };
        setMessages((prev) => [...prev, userMessage]);

        const audioBase64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = () => reject(new Error('Unable to read recorded audio.'));
          reader.readAsDataURL(audioBlob);
        });

        await runAnalysis({ audio: audioBase64 });
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      setPermissionError(PERMISSION_ERROR_MESSAGE);
    }
  };

  const handleStopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state !== 'recording') {
      return;
    }

    setIsRecording(false);
    recorder.stop();
  };

  useEffect(() => () => {
    closeCamera();
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    stopStreamTracks(audioStreamRef.current);
  }, []);

  return (
    <main className={styles.appShell}>
      <section className={styles.chatContainer}>
        <header className={styles.chatHeader}>HearMe AI Support Chat</header>
        <ChatWindow
          messages={messages}
          isAnalyzing={isAnalyzing}
          isStreaming={isStreaming}
          isRecording={isRecording}
          cameraOpen={cameraOpen}
          cameraVideoRef={cameraVideoRef}
          onCapture={handleCapture}
          onCloseCamera={closeCamera}
          permissionError={permissionError}
        />
        <InputBar
          onSend={handleSend}
          disabled={isAnalyzing || isStreaming}
          onOpenCamera={handleOpenCamera}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          isRecording={isRecording}
        />
      </section>
    </main>
  );
}
