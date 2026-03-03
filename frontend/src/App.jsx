import { useEffect, useRef, useState } from 'react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import styles from './App.module.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_ENDPOINT = `${API_BASE_URL}/api/v1/analyze`;

function formatAssistantResponse(data) {
  const emotion = data.fused_emotion || 'Unknown';
  const confidence = typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence ?? 'N/A';
  const responseText = data.empathetic_response || data.response_text || "I'm here with you.";

  return `Emotion detected: ${emotion} (Confidence: ${confidence})\n"${responseText}"`;
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hi, I'm here to listen. Tell me how you're feeling today.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async (text) => {
    const userMessage = { id: Date.now(), role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
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

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: formatAssistantResponse(data),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: `I ran into an issue while analyzing that message: ${error.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className={styles.appShell}>
      <section className={styles.chatContainer}>
        <header className={styles.chatHeader}>HearMe AI Support Chat</header>
        <ChatWindow messages={messages} isLoading={isLoading} />
        <div ref={chatEndRef} />
        <InputBar onSend={handleSend} disabled={isLoading} />
      </section>
    </main>
  );
}
