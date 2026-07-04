import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Норникель brand
        brand: {
          DEFAULT: "#00E6A6",
          dark: "#00C48F",
          soft: "#7CF3D4",
        },
        ink: {
          DEFAULT: "#2E2E48", // deep navy — заголовки/текст
          soft: "#6B6B80", // muted body
          faint: "#9A9AAD",
        },
        surface: "#F0F0F0", // фон полотна
        card: "#FFFFFF",
        line: "#E6E6EC",
        // Цвета фазовой маски (закреплены задачей)
        phase: {
          common: "#22C55E", // зелёный = обычные срастания
          thin: "#EF4444", // красный = тонкие срастания
          talc: "#3B82F6", // синий = тальк
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
      },
      borderRadius: {
        pill: "9999px",
        card: "20px",
      },
      boxShadow: {
        card: "0 8px 30px rgba(46, 46, 72, 0.08)",
        float: "0 12px 40px rgba(46, 46, 72, 0.12)",
      },
    },
  },
  plugins: [],
};

export default config;
