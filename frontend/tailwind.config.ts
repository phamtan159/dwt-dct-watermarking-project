import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 12px 40px rgba(16, 185, 129, 0.25)",
        card: "0 24px 80px rgba(15, 23, 42, 0.34)",
      },
      fontFamily: {
        body: ["'Plus Jakarta Sans'", "Segoe UI", "sans-serif"],
        display: ["'Space Grotesk'", "Segoe UI", "sans-serif"],
      },
      backgroundImage: {
        mesh:
          "radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 30%), radial-gradient(circle at top right, rgba(56,189,248,0.18), transparent 28%), radial-gradient(circle at bottom left, rgba(249,115,22,0.12), transparent 26%)",
      },
    },
  },
  plugins: [],
} satisfies Config;
