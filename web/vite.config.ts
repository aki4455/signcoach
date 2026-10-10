import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server (npm run dev) forwards API, media and health calls to FastAPI on port 8000.
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // fs.allow: the app imports config/i18n/en.json and config/rules/signs.json from the repo root.
  server: { proxy: { "/api": backend, "/media": backend, "/health": backend }, fs: { allow: [".."] } },
});
