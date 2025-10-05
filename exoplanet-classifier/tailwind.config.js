module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html'
  ],
  theme: {
    extend: {
      colors: {
        space: '#0b1220',
        nebula: '#0f1724'
      },
      backgroundImage: {
        'space-gradient': 'radial-gradient(ellipse at 10% 10%, rgba(72,72,240,0.14), transparent), linear-gradient(180deg, rgba(0,0,0,0.6), rgba(2,6,23,0.9))'
      },
    },
  },
  plugins: [],
}
