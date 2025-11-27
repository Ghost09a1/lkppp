/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        candy: {
          dark: "#0f0f1a",
          card: "#1a1a2e",
          accent: "#e94560",
        },
      },
    },
  },
  plugins: [],
};
