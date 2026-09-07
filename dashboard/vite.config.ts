import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs so a single build works both at "/" (local
  // `python -m maads dashboard`) and at "/dashboard/" (mounted in the hosted
  // account app). The API base is resolved at runtime — see shared/api.ts.
  base: "./",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
