import { motion } from 'framer-motion'

const variants = {
    primary: 'bg-neon-teal text-midnight-950 hover:bg-neon-green btn-glow',
    secondary: 'bg-midnight-700 text-white hover:bg-midnight-600 border border-midnight-600',
    ghost: 'text-gray-400 hover:text-white hover:bg-midnight-800',
    danger: 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30',
}

const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
}

export default function Button({
    children,
    variant = 'primary',
    size = 'md',
    className = '',
    disabled = false,
    ...props
}) {
    return (
        <motion.button
            whileHover={{ scale: disabled ? 1 : 1.02 }}
            whileTap={{ scale: disabled ? 1 : 0.98 }}
            className={`
        font-medium rounded-lg transition-colors
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `}
            disabled={disabled}
            {...props}
        >
            {children}
        </motion.button>
    )
}
