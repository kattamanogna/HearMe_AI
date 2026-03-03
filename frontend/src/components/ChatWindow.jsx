import MessageBubble from './MessageBubble';
import styles from './ChatWindow.module.css';

export default function ChatWindow({ messages, isLoading }) {
  return (
    <section className={styles.chatWindow} aria-live="polite">
      {messages.map((message) => (
        <MessageBubble key={message.id} role={message.role} content={message.content} />
      ))}
      {isLoading ? (
        <MessageBubble role="assistant" content="Analyzing your message..." />
      ) : null}
    </section>
  );
}
