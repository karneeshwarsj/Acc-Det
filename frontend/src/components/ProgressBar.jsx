import React from "react";
import styles from "./ProgressBar.module.css";

const STAGES = [
  { key: "upload",    label: "Uploading",           icon: "📤" },
  { key: "extract",   label: "Extracting frames",   icon: "🎞️" },
  { key: "detect",    label: "Detecting vehicles",  icon: "🔍" },
  { key: "compute",   label: "Computing risk",      icon: "🧠" },
  { key: "done",      label: "Done",                icon: "✅" },
];

/**
 * Multi-step animated progress indicator.
 * @param {number} progress – 0-100
 * @param {boolean} active – whether processing is underway
 */
export default function ProgressBar({ progress = 0, active = false }) {
  if (!active && progress === 0) return null;

  // Map progress to active stage index
  const stageIdx = progress >= 100 ? 4
    : progress >= 75 ? 3
    : progress >= 50 ? 2
    : progress >= 25 ? 1
    : 0;

  return (
    <div className={`${styles.wrapper} glass fade-in`}>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
      <div className={styles.stages}>
        {STAGES.map((stage, i) => {
          const done    = i < stageIdx;
          const current = i === stageIdx && active;
          return (
            <div
              key={stage.key}
              className={`${styles.stage} ${done ? styles.done : ""} ${current ? styles.current : ""}`}
            >
              <div className={`${styles.stageIcon} ${current ? "spin" : ""}`}>
                {done ? "✓" : stage.icon}
              </div>
              <span className={styles.stageLabel}>{stage.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
