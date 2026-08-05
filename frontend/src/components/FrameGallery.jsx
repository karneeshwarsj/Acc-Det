import React, { useState } from "react";
import styles from "./FrameGallery.module.css";

/**
 * Gallery of annotated frames from backend analysis.
 * @param {string[]} frames – base64-encoded JPEG strings
 * @param {string} title – gallery header
 */
export default function FrameGallery({ frames, title = "Annotated Frames" }) {
  const [selected, setSelected] = useState(0);

  if (!frames || frames.length === 0) return null;

  return (
    <div className={`${styles.wrapper} glass fade-in`}>
      <div className={styles.header}>
        <span className={styles.title}>{title}</span>
        <span className={styles.count}>{frames.length} frame{frames.length !== 1 ? "s" : ""}</span>
      </div>

      {/* Main preview */}
      <div className={styles.mainPreview}>
        <img
          src={`data:image/jpeg;base64,${frames[selected]}`}
          alt={`Frame ${selected + 1}`}
          className={styles.mainImg}
        />
        <div className={styles.frameIdx}>Frame {selected + 1}/{frames.length}</div>
      </div>

      {/* Thumbnails */}
      {frames.length > 1 && (
        <div className={styles.thumbRow}>
          {frames.map((f, i) => (
            <button
              key={i}
              className={`${styles.thumb} ${i === selected ? styles.thumbActive : ""}`}
              onClick={() => setSelected(i)}
            >
              <img
                src={`data:image/jpeg;base64,${f}`}
                alt={`Thumb ${i + 1}`}
                className={styles.thumbImg}
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
