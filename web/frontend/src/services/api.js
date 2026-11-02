import axios from "axios";

const api = axios.create({
  baseURL: "",
  // Qwen Max + CAD JSON generation can take longer than 3 minutes
  timeout: 180000
});

export const chatWithLLM = (payload) =>
  api.post("/api/chat", payload);

export const selectJsonAsset = (payload) =>
  api.post("/api/select-json", payload);

export const generateAssets = (payload) =>
  api.post("/api/generate", payload);

export const cleanupOutput = () =>
  api.post("/api/cleanup-output");

export const bootstrap = () =>
  api.get("/api/bootstrap");

export const listExistingFiles = () =>
  api.get("/api/list-files");

export default api;