import React from "react";
import styles from "./ProbabilityTable.module.css";

const RISK_COLORS = {
  Low:      { bar: "#22c55e", badge: "badge-green", bg: "rgba(34,197,94,0.06)" },
  Medium:   { bar: "#f59e0b", badge: "badge-amber", bg: "rgba(245,158,11,0.06)" },
  High:     { bar: "#f59e0b", badge: "badge-amber", bg: "rgba(245,158,11,0.08)" },
  Critical: { bar: "#ef4444", badge: "badge-red",   bg: "rgba(239,68,68,0.10)" },
};

export default function ProbabilityTable({ data }) {
  if (!data) return null;

  const { vehicle_count, estimated_distance_m, probability_table, warning_message, annotated_image_b64 } = data;

  return (
    <div className={`${styles.wrapper} fade-in`}>
      {/* Header */}
      <div className={`${styles.header} glass`}>
        <div className={styles.headerLeft}>
          <div className={styles.title}>📐 Proximity Risk Analysis</div>
          <div className={styles.metaRow}>
            {vehicle_count !== undefined && (
              <span className="badge badge-blue">🚗 {vehicle_count} vehicle{vehicle_count !== 1 ? "s" : ""} detected</span>
            )}
            {estimated_distance_m !== null && estimated_distance_m !== undefined && (
              <span className="badge badge-amber">📏 ~{estimated_distance_m}m between vehicles</span>
            )}
          </div>
        </div>
        {annotated_image_b64 && (
          <img
            src={`data:image/jpeg;base64,${annotated_image_b64}`}
            alt="Annotated"
            className={styles.previewImg}
          />
        )}
      </div>

      {/* Warning */}
      {warning_message && (
        <div className={styles.warning}>
          <span>⚠️</span>
          <span>{warning_message}</span>
        </div>
      )}

      {/* Table */}
      {probability_table && probability_table.length > 0 ? (
        <div className={`${styles.tableWrap} glass`}>
          <div className={styles.tableTitle}>Accident Probability by Speed Range</div>
          <div className={styles.tableDesc}>
            Based on stopping distance formula: <code>d = v·t_r + v²/(2µg)</code>, µ = 0.70 (dry asphalt)
          </div>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Speed Range</th>
                <th>Avg Speed</th>
                <th>Stopping Distance</th>
                <th>Risk Probability</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              {probability_table.map((row, i) => {
                const cfg = RISK_COLORS[row.risk_level] || RISK_COLORS.Low;
                return (
                  <tr key={i} className={styles.tableRow} style={{ background: cfg.bg }}>
                    <td className={styles.cellSpeed}>
                      <span className={styles.speedLabel}>{row.speed_range}</span>
                    </td>
                    <td className={styles.cellMono}>{row.avg_speed_kmh} km/h</td>
                    <td className={styles.cellMono}>{row.stopping_distance_m} m</td>
                    <td className={styles.cellProb}>
                      <div className={styles.probBar}>
                        <div className={styles.probBarTrack}>
                          <div
                            className={styles.probBarFill}
                            style={{ width: `${row.probability_pct}%`, background: cfg.bar }}
                          />
                        </div>
                        <span className={styles.probPct}>{row.probability_pct}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${cfg.badge}`}>
                        {row.risk_level === "Low" ? "🟢" : row.risk_level === "Critical" ? "🔴" : "🟡"}{" "}
                        {row.risk_level}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className={styles.tableNote}>
            ℹ️ Distance estimation is approximate. Real-world braking depends on road condition, tire quality, and driver reaction.
          </div>
        </div>
      ) : (
        <div className={`${styles.noData} glass`}>
          No probability data available. Ensure at least 2 vehicles are visible in the image.
        </div>
      )}
    </div>
  );
}
