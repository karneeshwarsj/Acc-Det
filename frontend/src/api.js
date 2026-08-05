import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min timeout for large files
});

/**
 * Auto-analyze a video — backend decides complete vs incomplete automatically.
 * @param {File} file - Video file
 * @param {function} onProgress - Optional upload progress callback (0-100)
 */
export async function analyzeVideo(file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  const { data } = await api.post("/api/video/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 40));
      }
    },
  });
  return data;
}

/**
 * Auto-analyze an image — backend decides accident scene vs near-miss automatically.
 * @param {File} file - Image file (.jpg, .png)
 * @param {function} onProgress - Optional upload progress callback (0-100)
 */
export async function analyzeImage(file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  const { data } = await api.post("/api/image/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 40));
      }
    },
  });
  return data;
}

/**
 * Check backend health.
 */
export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}
