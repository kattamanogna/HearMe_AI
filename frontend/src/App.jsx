import { useEffect, useMemo, useRef, useState } from 'react';

const API_ENDPOINT = '/api/v1/analyze';

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const base64 = result.includes(',') ? result.split(',')[1] : result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function dataUrlToBase64(dataUrl) {
  return dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
}

export default function App() {
  const [text, setText] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [faceCapture, setFaceCapture] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Waiting for input...');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const canSubmit = useMemo(() => text.trim().length > 0 && !isSubmitting, [text, isSubmitting]);

  useEffect(() => {
    let active = true;

    async function initCamera() {
      try {
        setStatusMessage('Requesting webcam access...');
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!active) return;
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatusMessage('Webcam active. Ready to capture face.');
      } catch (cameraError) {
        setStatusMessage('Webcam unavailable. You can still submit text/audio.');
      }
    }

    initCamera();

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const captureFace = () => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      setError('Camera stream is not ready yet. Try again in a moment.');
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);

    setFaceCapture(imageDataUrl);
    setError('');
    setStatusMessage('Face snapshot captured.');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;

    try {
      setIsSubmitting(true);
      setError('');
      setResult(null);

      setStatusMessage('Preparing request payload...');
      const payload = {
        session_id: 'frontend-session',
        text: text.trim(),
      };

      if (audioFile) {
        setStatusMessage('Encoding audio file...');
        payload.audio_bytes = await fileToBase64(audioFile);
      }

      if (faceCapture) {
        setStatusMessage('Encoding face capture...');
        payload.face_base64 = dataUrlToBase64(faceCapture);
      }

      setStatusMessage('Sending data to backend...');
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || 'Failed to analyze request.');
      }

      setStatusMessage('Processing results...');
      const data = await response.json();
      setResult(data);
      setStatusMessage('Done. Review the emotion analysis and response text below.');
    } catch (submitError) {
      setError(submitError.message || 'Unexpected request error.');
      setStatusMessage('Request failed. Please check inputs and backend availability.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="app-shell">
      <h1>HearMe AI</h1>
      <p className="status">Status: {statusMessage}</p>

      <form className="panel" onSubmit={handleSubmit}>
        <label htmlFor="text">Text input</label>
        <textarea
          id="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Enter how you feel..."
          rows={4}
          required
        />

        <label htmlFor="audio">Audio upload</label>
        <input
          id="audio"
          type="file"
          accept="audio/*"
          onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)}
        />

        <section className="camera-section">
          <div>
            <label>Webcam capture</label>
            <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
            <button type="button" onClick={captureFace} className="secondary-button">
              Capture Face Snapshot
            </button>
          </div>

          <div>
            <p>Captured face preview</p>
            {faceCapture ? <img src={faceCapture} alt="Captured face" className="face-preview" /> : <p>No capture yet.</p>}
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </section>

        <button type="submit" disabled={!canSubmit}>
          {isSubmitting ? 'Analyzing...' : 'Analyze Emotion'}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="panel result-panel">
          <h2>Analysis Result</h2>
          <ul>
            <li>Text Emotion: <strong>{result.text_emotion}</strong></li>
            <li>Audio Emotion: <strong>{result.audio_emotion ?? 'N/A'}</strong></li>
            <li>Face Emotion: <strong>{result.face_emotion ?? 'N/A'}</strong></li>
            <li>Fused Emotion: <strong>{result.fused_emotion}</strong></li>
            <li>Confidence: <strong>{result.confidence}</strong></li>
          </ul>

          <h3>Response Text</h3>
          <p>{result.response_text}</p>
        </section>
      ) : null}
    </main>
  );
}
