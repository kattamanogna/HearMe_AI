import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import styles from './ChatWindow.module.css';

function TypingIndicator() {
  return (
    <div className={styles.typingIndicator} aria-label="Assistant is processing emotion">
      <span className={styles.spinner} aria-hidden="true" />
      <span>Analyzing emotion...</span>
    </div>
  );
}

function CameraPreview({ isOpen, videoRef, onCapture, onClose }) {
  if (!isOpen) {
    return null;
  }

  return (
    <section className={styles.cameraPanel} aria-label="Camera preview">
      <video ref={videoRef} autoPlay playsInline muted className={styles.cameraVideo} />
      <div className={styles.cameraActions}>
        <button type="button" className={styles.cameraButton} onClick={onCapture}>
          Capture
        </button>
        <button type="button" className={`${styles.cameraButton} ${styles.closeButton}`} onClick={onClose}>
          Close
        </button>
      </div>
    </section>
  );
}

export default function ChatWindow({
  messages,
  isAnalyzing,
  isStreaming,
  isRecording,
  cameraOpen,
  cameraVideoRef,
  onCapture,
  onCloseCamera,
  permissionError,
}) {
  const chatWindowRef = useRef(null);
  const lastAssistantMessageId = [...messages].reverse().find((message) => message.role === 'assistant')?.id;

  useEffect(() => {
    const chatWindow = chatWindowRef.current;
    if (!chatWindow) {
      return;
    }

    chatWindow.scrollTo({
      top: chatWindow.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isAnalyzing, isStreaming, cameraOpen, isRecording, permissionError]);

  return (
    <section ref={chatWindowRef} className={styles.chatWindow} aria-live="polite">
      {cameraOpen ? (
        <CameraPreview isOpen={cameraOpen} videoRef={cameraVideoRef} onCapture={onCapture} onClose={onCloseCamera} />
      ) : null}
      {isRecording ? <div className={styles.recordingIndicator}>Recording...</div> : null}
      {permissionError ? <div className={styles.permissionError}>{permissionError}</div> : null}

      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          role={message.role}
          content={message.content}
          isStreaming={isStreaming && message.id === lastAssistantMessageId}
        />
      ))}
      {isAnalyzing ? <MessageBubble role="assistant" content={<TypingIndicator />} /> : null}
    </section>
  );
}
