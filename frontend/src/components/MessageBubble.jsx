import styles from './MessageBubble.module.css';

export default function MessageBubble({ role, content, isStreaming = false }) {
  const isUser = role === 'user';
  const isText = typeof content === 'string';

  return (
    <div className={`${styles.messageRow} ${isUser ? styles.userRow : styles.assistantRow}`}>
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {isText ? (
          <pre className={`${styles.messageText} ${isStreaming && !isUser ? styles.streamingText : ''}`}>{content}</pre>
        ) : (
          content
        )}
      </div>
    </div>
  );
}
