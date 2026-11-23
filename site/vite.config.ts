import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project sites live under /<repo>/. Override with BASE_PATH (e.g. "/" for a user site).
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH ?? "/FLAG_Voice_Face/",
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
});
