import styles from './MessageBubble.module.css';

export default function MessageBubble({ role, content }) {
  const isUser = role === 'user';

  return (
    <div className={`${styles.messageRow} ${isUser ? styles.userRow : styles.assistantRow}`}>
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        <pre className={styles.messageText}>{content}</pre>
      </div>
    </div>
  );
}
