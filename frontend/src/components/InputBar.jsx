import { useState } from 'react';
import styles from './InputBar.module.css';

export default function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    const message = value.trim();
    if (!message || disabled) {
      return;
    }
    onSend(message);
    setValue('');
  };

  return (
    <form className={styles.inputBar} onSubmit={handleSubmit}>
      <input
        type="text"
        className={styles.textInput}
        placeholder="Share how you're feeling..."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
      <button type="button" className={styles.iconButton} aria-label="Open camera" disabled={disabled}>
        📷
      </button>
      <button type="button" className={styles.iconButton} aria-label="Record audio" disabled={disabled}>
        🎤
      </button>
      <button type="submit" className={styles.sendButton} aria-label="Send message" disabled={disabled || !value.trim()}>
        +
      </button>
    </form>
  );
}
