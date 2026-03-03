import { useState } from 'react';
import styles from './InputBar.module.css';

function CameraIcon() {
  return (
    <svg className={styles.iconGlyph} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M9 4.5a1 1 0 0 0-.83.44L7.02 6.5H5.5A2.5 2.5 0 0 0 3 9v8a2.5 2.5 0 0 0 2.5 2.5h13A2.5 2.5 0 0 0 21 17V9a2.5 2.5 0 0 0-2.5-2.5h-1.52l-1.15-1.56A1 1 0 0 0 15 4.5H9Zm3 12.25A4.25 4.25 0 1 1 12 8.25a4.25 4.25 0 0 1 0 8.5Zm0-1.75A2.5 2.5 0 1 0 12 10a2.5 2.5 0 0 0 0 5Z" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg className={styles.iconGlyph} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M12 3a3.75 3.75 0 0 0-3.75 3.75v5.5a3.75 3.75 0 0 0 7.5 0v-5.5A3.75 3.75 0 0 0 12 3Zm-5.75 8.75a.75.75 0 0 0-1.5 0A7.25 7.25 0 0 0 11.25 19v2a.75.75 0 0 0 1.5 0v-2a7.25 7.25 0 0 0 6.5-7.25.75.75 0 0 0-1.5 0 5.75 5.75 0 1 1-11.5 0Z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg className={styles.sendGlyph} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M3.8 11.52a1 1 0 0 0 0 1.96l6.78 1.4 1.4 6.79a1 1 0 0 0 1.82.37L21.74 3.9a1 1 0 0 0-1.24-1.46L3.8 11.52Zm7.53 1.08L6.92 11.7l11.6-5.26-7.19 6.16Zm.97 1.05 6.16-7.2-5.26 11.61-.9-4.41Z" />
    </svg>
  );
}

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
        <CameraIcon />
      </button>
      <button type="button" className={styles.iconButton} aria-label="Record audio" disabled={disabled}>
        <MicIcon />
      </button>
      <button type="submit" className={styles.sendButton} aria-label="Send message" disabled={disabled || !value.trim()}>
        <SendIcon />
      </button>
    </form>
  );
}
