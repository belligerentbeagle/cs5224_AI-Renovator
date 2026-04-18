/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        cream: '#F5EDE0',
        'cream-dark': '#EDD9C0',
        rs: {
          amber:  '#C17A3F',
          light:  '#D4956A',
          dark:   '#8B5E3C',
          border: '#E8D5BC',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
