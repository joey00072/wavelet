import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // Same-origin API in development: the Python dashboard runs beside Vite.
      "/api": {
        target: process.env.WAVELET_API ?? "http://127.0.0.1:8766",
        changeOrigin: true,
      },
    },
  },
});
