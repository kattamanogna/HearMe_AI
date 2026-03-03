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

export default function ChatWindow({ messages, isAnalyzing, isStreaming }) {
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
  }, [messages, isAnalyzing, isStreaming]);

  return (
    <section ref={chatWindowRef} className={styles.chatWindow} aria-live="polite">
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
