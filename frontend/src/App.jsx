import React, { useState, useCallback } from "react";
import UploadZone from "./components/UploadZone";
import { ClassificationResult, ProbabilityResult, ErrorCard } from "./components/ResultCard";
import ProbabilityTable from "./components/ProbabilityTable";
import ProgressBar from "./components/ProgressBar";
import FrameGallery from "./components/FrameGallery";
import { analyzeVideo, analyzeImage } from "./api";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab]   = useState("video");
  const [loading,   setLoading]     = useState(false);
  const [progress,  setProgress]    = useState(0);
  const [result,    setResult]      = useState(null);
  const [error,     setError]       = useState(null);

  const clearResults = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    clearResults();
  };

  const handleAnalyze = useCallback(async (file) => {
    clearResults();
    setLoading(true);
    setProgress(15);

    try {
      let data;

      const onProgress = (p) => setProgress(p);

      if (activeTab === "video") {
        setProgress(20);
        data = await analyzeVideo(file, onProgress);
      } else {
        setProgress(20);
        data = await analyzeImage(file, onProgress);
      }

      setProgress(100);
      setResult(data);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "An unexpected error occurred.";
      setError(msg);
      setProgress(0);
    } finally {
      setLoading(false);
    }
  }, [activeTab, clearResults]);

  // ── Determine result rendering based on analysis_mode from backend ──────────
  const renderResult = () => {
    if (!result) return null;

    const mode = result.analysis_mode;

    return (
      <div className="results-section fade-in">
        <div className="divider" />

        {/* Mode badge — what the AI detected */}
        <div className="mode-banner">
          <span className="mode-icon">{result.mode_icon || "🤖"}</span>
          <div>
            <div className="mode-label">AI Detected: {result.mode_label || mode}</div>
            {result.detection_info && (
              <div className="mode-reason">
                {result.detection_info.reason}
              </div>
            )}
          </div>
        </div>

        {/* Classification result (complete video or complete image) */}
        {mode === "complete_classification" && (
          <>
            <ClassificationResult data={result} />
            {result.annotated_frames_b64?.length > 0 && (
              <FrameGallery
                frames={result.annotated_frames_b64}
                title="Detection Preview"
              />
            )}
            {result.annotated_image_b64 && (
              <div className="glass annotated-image-wrap fade-in">
                <div className="annotated-title">Annotated Image</div>
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image_b64}`}
                  alt="Annotated detection"
                  className="annotated-img"
                />
              </div>
            )}
          </>
        )}

        {/* Probability result (incomplete video) */}
        {mode === "incomplete_probability" && activeTab === "video" && (
          <>
            <ProbabilityResult data={result} />
            {result.annotated_frames_b64?.length > 0 && (
              <FrameGallery
                frames={result.annotated_frames_b64}
                title="Tracked Vehicles"
              />
            )}
          </>
        )}

        {/* Proximity probability table (incomplete image) */}
        {mode === "incomplete_probability" && activeTab === "image" && (
          <ProbabilityTable data={result} />
        )}
      </div>
    );
  };

  return (
    <div className="app">
      {/* ── Navbar ──────────────────────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="container navbar-inner">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <span className="logo-text">
              Accident<span className="logo-ai">AI</span>
            </span>
          </div>
          <div className="nav-links">
            <a href="#" className="nav-link active">Analyze</a>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link"
            >
              API Docs
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="container text-center">
          <div className="hero-badge badge badge-red">AI-Powered · Auto-Detection</div>
          <h1 className="hero-title">
            Real-Time Accident<br />
            <span className="hero-gradient">Intelligence</span>
          </h1>
          <p className="hero-sub">
            Upload any video or image — the AI automatically determines the scenario
            and delivers instant accident detection, risk scoring, or collision probability.
          </p>
        </div>
      </section>

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <main className="container main-content">
        {/* Main tabs */}
        <div className="tab-bar">
          <button
            className={`tab-btn ${activeTab === "video" ? "active" : ""}`}
            onClick={() => handleTabChange("video")}
          >
            🎬 Video Analysis
          </button>
          <button
            className={`tab-btn ${activeTab === "image" ? "active" : ""}`}
            onClick={() => handleTabChange("image")}
          >
            🖼️ Image Analysis
          </button>
        </div>

        {/* Upload zone */}
        <div className="upload-section glass">
          <UploadZone
            key={activeTab}
            mediaType={activeTab}
            onAnalyze={handleAnalyze}
            loading={loading}
          />
        </div>

        {/* Progress indicator */}
        {loading && <ProgressBar progress={progress} active={loading} />}

        {/* Error */}
        {error && <ErrorCard message={error} />}

        {/* Results */}
        {renderResult()}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="container text-center">
          <p className="footer-text">
            AccidentAI · YOLOv8 + EfficientNet-B1 + LSTM · Auto-Detection Engine · FastAPI &amp; React
          </p>
        </div>
      </footer>
    </div>
  );
}
