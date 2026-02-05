import { motion } from 'framer-motion'

export default function ProgressBar({
    value = 0,
    className = ''
}) {
    return (
        <div className={`h-2 bg-midnight-700 rounded-full overflow-hidden ${className}`}>
            <motion.div
                className="h-full bg-gradient-to-r from-neon-teal to-neon-cyan rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
            />
        </div>
    )
}
