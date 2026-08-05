import React from "react";
import styles from "./ResultCard.module.css";

const RISK_CONFIG = {
  "Low":      { color: "green", icon: "✅", glow: "glow-green" },
  "Medium":   { color: "amber", icon: "⚠️", glow: "glow-amber" },
  "High":     { color: "amber", icon: "🔶", glow: "glow-amber" },
  "Critical": { color: "red",   icon: "🔴", glow: "glow-red"   },
  "Accident": { color: "red",   icon: "💥", glow: "glow-red"   },
  "No Accident":{ color: "green",icon: "✅",glow: "glow-green" },
};

function CircularProgress({ pct, color }) {
  const r = 52, circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  const colorMap = { red: "#ef4444", green: "#22c55e", amber: "#f59e0b", blue: "#3b82f6" };
  const stroke = colorMap[color] || "#ef4444";

  return (
    <svg className={styles.ring} viewBox="0 0 120 120">
      <circle cx="60" cy="60" r={r} className={styles.ringTrack} />
      <circle
        cx="60" cy="60" r={r}
        className={styles.ringFill}
        stroke={stroke}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        style={{ filter: `drop-shadow(0 0 8px ${stroke}88)` }}
      />
      <text x="60" y="60" textAnchor="middle" dominantBaseline="central" className={styles.ringText}>
        {Math.round(pct)}%
      </text>
    </svg>
  );
}

/* ── Classification Result (label + confidence) ───────────────────────────── */
export function ClassificationResult({ data }) {
  if (!data) return null;

  const label      = data.label || "Unknown";
  const confidence = Math.round((data.confidence || 0) * 100);
  const isAccident = label.toLowerCase().includes("accident") || label.toLowerCase() === "accident";
  const cfg        = isAccident ? RISK_CONFIG["Accident"] : RISK_CONFIG["No Accident"];
  const color      = cfg.color;

  return (
    <div className={`${styles.card} glass ${cfg.glow} fade-in`}>
      <div className={styles.header}>
        <span className={styles.headerIcon}>{cfg.icon}</span>
        <div>
          <div className={styles.resultLabel}>{label}</div>
          <div className={styles.metaRow}>
            {data.vehicle_count !== undefined && (
              <span className="badge badge-blue">🚗 {data.vehicle_count} vehicle{data.vehicle_count !== 1 ? "s" : ""}</span>
            )}
            {data.model_status && (
              <span className={`badge ${data.model_status === "fine_tuned" ? "badge-green" : "badge-purple"}`}>
                {data.model_status === "fine_tuned" ? "🧠 Fine-tuned" : "🤖 Pretrained"}
              </span>
            )}
          </div>
        </div>
        <div className={styles.ringWrap}>
          <CircularProgress pct={confidence} color={color} />
          <div className={styles.ringLabel}>Confidence</div>
        </div>
      </div>

      {data.class_scores && (
        <div className={styles.scoreBar}>
          {Object.entries(data.class_scores).map(([cls, score]) => (
            <ScoreRow key={cls} label={cls} score={score} />
          ))}
        </div>
      )}

      {data.processing_time_seconds !== undefined && (
        <div className={styles.footer}>
          ⚡ Processed in {data.processing_time_seconds}s
          {data.frame_count && <span> · {data.frame_count} frames</span>}
          {data.windows_analyzed && <span> · {data.windows_analyzed} windows</span>}
        </div>
      )}
    </div>
  );
}

/* ── Probability Result (gauge + risk level) ──────────────────────────────── */
export function ProbabilityResult({ data }) {
  if (!data) return null;

  const prob      = Math.round((data.probability || 0) * 100);
  const riskLevel = data.risk_level || "Low";
  const cfg       = RISK_CONFIG[riskLevel] || RISK_CONFIG["Low"];

  return (
    <div className={`${styles.card} glass ${cfg.glow} fade-in`}>
      <div className={styles.header}>
        <span className={styles.headerIcon}>{cfg.icon}</span>
        <div className={styles.probInfo}>
          <div className={styles.riskLevelLabel}>Risk Level</div>
          <div className={`${styles.riskLevelValue} ${styles[`risk_${riskLevel.toLowerCase()}`]}`}>
            {riskLevel}
          </div>
          <div className={styles.metaRow}>
            {data.ttc_seconds !== undefined && data.ttc_seconds < 999 && (
              <span className="badge badge-amber">⏱ TTC: {data.ttc_seconds}s</span>
            )}
            {data.vehicle_pairs_analyzed !== undefined && (
              <span className="badge badge-blue">🔀 {data.vehicle_pairs_analyzed} pair{data.vehicle_pairs_analyzed !== 1 ? "s" : ""}</span>
            )}
          </div>
        </div>
        <div className={styles.ringWrap}>
          <CircularProgress pct={prob} color={cfg.color} />
          <div className={styles.ringLabel}>Probability</div>
        </div>
      </div>

      {/* Risk over time heatmap */}
      {data.heatmap_b64 && (
        <div className={styles.heatmapWrap}>
          <div className={styles.sectionTitle}>Risk Over Time</div>
          <img
            src={`data:image/png;base64,${data.heatmap_b64}`}
            alt="Risk heatmap"
            className={styles.heatmap}
          />
        </div>
      )}

      {data.processing_time_seconds !== undefined && (
        <div className={styles.footer}>
          ⚡ Processed in {data.processing_time_seconds}s
          {data.frame_count && <span> · {data.frame_count} frames</span>}
        </div>
      )}
    </div>
  );
}

/* ── Shared ScoreRow bar ──────────────────────────────────────────────────── */
function ScoreRow({ label, score }) {
  const pct = Math.round(score * 100);
  const colorClass = pct > 65 ? styles.barRed : pct > 35 ? styles.barAmber : styles.barGreen;
  return (
    <div className={styles.scoreRow}>
      <span className={styles.scoreLabel}>{label}</span>
      <div className={styles.scoreTrack}>
        <div className={`${styles.scoreFill} ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.scorePct}>{pct}%</span>
    </div>
  );
}

/* ── Error card ───────────────────────────────────────────────────────────── */
export function ErrorCard({ message }) {
  return (
    <div className={`${styles.errorCard} glass fade-in`}>
      <span className={styles.errorIcon}>❌</span>
      <div>
        <div className={styles.errorTitle}>Analysis Failed</div>
        <div className={styles.errorMsg}>{message}</div>
      </div>
    </div>
  );
}
