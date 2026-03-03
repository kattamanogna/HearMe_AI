import { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import styles from './App.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_ENDPOINT = `${API_BASE_URL}/api/v1/analyze`;
const STREAM_WORD_DELAY_MS = 60;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function formatAssistantResponse(data) {
  const emotion = data.fused_emotion || 'Unknown';
  const confidence = typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence ?? 'N/A';
  const responseText = data.empathetic_response || data.response_text || "I'm here with you.";

  return `Emotion detected: ${emotion} (Confidence: ${confidence})\n"${responseText}"`;
}

async function analyzeMessage(text) {
  const response = await fetch(API_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: 'frontend-session',
      text,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Unable to analyze message right now.');
  }

  return response.json();
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

  const handleSend = async (text) => {
    const userMessage = { id: Date.now(), role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setIsAnalyzing(true);

    try {
      const data = await analyzeMessage(text);
      setIsAnalyzing(false);
      await appendAssistantMessage(formatAssistantResponse(data));
    } catch (error) {
      setIsAnalyzing(false);
      await appendAssistantMessage(`I ran into an issue while analyzing that message: ${error.message}`);
    }
  };

  return (
    <main className={styles.appShell}>
      <section className={styles.chatContainer}>
        <header className={styles.chatHeader}>HearMe AI Support Chat</header>
        <ChatWindow messages={messages} isAnalyzing={isAnalyzing} isStreaming={isStreaming} />
        <InputBar onSend={handleSend} disabled={isAnalyzing || isStreaming} />
      </section>
    </main>
  );
}
