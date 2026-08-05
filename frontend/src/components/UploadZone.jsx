import React, { useState, useCallback } from "react";
import styles from "./UploadZone.module.css";

const VIDEO_TYPES = ["video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/webm"];
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_SIZE_MB = { video: 100, image: 20 };

export default function UploadZone({ mediaType, onAnalyze, loading }) {
  const [dragging, setDragging] = useState(false);
  const [file,     setFile]     = useState(null);
  const [preview,  setPreview]  = useState(null);
  const [error,    setError]    = useState(null);

  const acceptTypes = mediaType === "video" ? VIDEO_TYPES : IMAGE_TYPES;
  const accept = mediaType === "video"
    ? ".mp4,.avi,.mov,.mkv,.webm"
    : ".jpg,.jpeg,.png,.webp,.bmp";

  const handleFile = useCallback((incoming) => {
    setError(null);
    if (!incoming) return;

    if (!acceptTypes.includes(incoming.type)) {
      setError(`Unsupported file type: ${incoming.type || "unknown"}`);
      return;
    }

    const maxMB = MAX_SIZE_MB[mediaType];
    if (incoming.size > maxMB * 1024 * 1024) {
      setError(`File too large. Maximum: ${maxMB} MB.`);
      return;
    }

    setFile(incoming);
    if (mediaType === "image") {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(incoming);
    } else {
      setPreview(URL.createObjectURL(incoming));
    }
  }, [acceptTypes, mediaType]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = () => {
    if (!file || loading) return;
    onAnalyze(file);
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  const sizeMB = file ? (file.size / (1024 * 1024)).toFixed(1) : "0";

  return (
    <div className={styles.wrapper}>

      {/* AI auto-detect notice */}
      <div className={styles.autoNotice}>
        <span className={styles.autoIcon}>🤖</span>
        <div>
          <div className={styles.autoTitle}>AI Auto-Detection</div>
          <div className={styles.autoDesc}>
            {mediaType === "video"
              ? "Upload any video — the AI will automatically determine if an accident occurred or estimate collision risk."
              : "Upload any image — the AI will automatically classify the accident or generate a proximity risk table."}
          </div>
        </div>
      </div>

      {/* Drop zone */}
      {!file ? (
        <div
          className={`${styles.dropZone} ${dragging ? styles.dragging : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById(`file-input-${mediaType}`).click()}
        >
          <input
            id={`file-input-${mediaType}`}
            type="file"
            accept={accept}
            className={styles.fileInput}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className={styles.dropIcon}>
            {mediaType === "video" ? "🎞️" : "🖼️"}
          </div>
          <p className={styles.dropTitle}>
            Drop your {mediaType} here
          </p>
          <p className={styles.dropSub}>
            or <span className={styles.browseLink}>click to browse</span>
          </p>
          <p className={styles.dropFormats}>
            {mediaType === "video"
              ? "MP4, AVI, MOV, MKV, WEBM — up to 100 MB"
              : "JPG, PNG, WEBP — up to 20 MB"}
          </p>
          {dragging && (
            <div className={styles.dropOverlay}>
              <span>✨ Drop to upload</span>
            </div>
          )}
        </div>
      ) : (
        /* Preview card */
        <div className={`${styles.previewCard} glass`}>
          <div className={styles.previewContent}>
            {mediaType === "image" && preview && (
              <img src={preview} alt="Preview" className={styles.previewImg} />
            )}
            {mediaType === "video" && preview && (
              <video src={preview} className={styles.previewVideo} muted playsInline controls />
            )}
            <div className={styles.previewMeta}>
              <div className={styles.previewFileName}>{file.name}</div>
              <div className={styles.previewFileSize}>{sizeMB} MB</div>
              <div className={styles.previewType}>
                <span className="badge badge-purple">🤖 Auto-Analysis</span>
              </div>
            </div>
          </div>
          <button
            className={`${styles.clearBtn} btn btn-ghost`}
            onClick={clearFile}
            disabled={loading}
          >
            ✕ Remove
          </button>
        </div>
      )}

      {error && (
        <div className={styles.errorMsg}>
          <span>⚠️</span> {error}
        </div>
      )}

      {/* Analyze button */}
      <button
        className={`btn btn-primary btn-lg w-full ${styles.analyzeBtn} ${file && !loading ? "pulse" : ""}`}
        onClick={handleAnalyze}
        disabled={!file || loading}
      >
        {loading ? (
          <><span className="spin">⟳</span> Analyzing…</>
        ) : (
          <><span>🔍</span> Analyze {mediaType === "video" ? "Video" : "Image"}</>
        )}
      </button>
    </div>
  );
}
