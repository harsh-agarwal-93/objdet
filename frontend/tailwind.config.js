/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                midnight: {
                    950: '#050508',
                    900: '#0a0a0f',
                    800: '#12121a',
                    700: '#1a1a24',
                    600: '#24242f',
                },
                neon: {
                    teal: '#00f0b5',
                    green: '#00ff9f',
                    cyan: '#00e5ff',
                }
            },
            fontFamily: {
                sans: ['Outfit', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
            animation: {
                'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
            },
            keyframes: {
                glow: {
                    '0%': { boxShadow: '0 0 3px rgba(0, 240, 181, 0.15), 0 0 5px rgba(0, 240, 181, 0.1)' },
                    '100%': { boxShadow: '0 0 10px rgba(0, 240, 181, 0.2), 0 0 15px rgba(0, 240, 181, 0.15)' },
                }
            }
        },
    },
    plugins: [],
}
