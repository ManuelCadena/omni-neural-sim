import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "OmniNeuralSim",
      fileName: (format) => `omni-neural-sim.${format}.js`,
      formats: ["es", "umd", "iife"]
    },
    rollupOptions: {
      output: {
        globals: {}
      }
    },
    sourcemap: true,
    minify: "esbuild"
  }
});
