import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/chat": "http://localhost:8000",
      "/reindex": "http://localhost:8000",
      "/status": "http://localhost:8000",
      "/logs": "http://localhost:8000",
    },
  },
});
