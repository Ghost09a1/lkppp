/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'candy-dark': '#0f172a',
        'candy-card': '#1e293b',
        'candy-pink': '#ec4899',
      },
    },
  },
  plugins: [],
}
