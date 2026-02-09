import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown } from 'lucide-react'

import PropTypes from 'prop-types'

export default function StatCard({
    icon: Icon,
    label,
    value,
    trend,
    className = ''
}) {
    const isPositiveTrend = trend > 0

    return (
        <motion.div
            whileHover={{ y: -2 }}
            className={`bg-midnight-800/50 border border-midnight-700 rounded-xl p-4 ${className}`}
        >
            <div className="flex items-start justify-between">
                <div className="p-2 rounded-lg bg-neon-teal/10">
                    <Icon className="w-5 h-5 text-neon-teal" />
                </div>
                {trend !== undefined && (
                    <div className={`flex items-center gap-1 text-xs ${isPositiveTrend ? 'text-neon-green' : 'text-red-400'}`}>
                        {isPositiveTrend ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {Math.abs(trend)}%
                    </div>
                )}
            </div>
            <div className="mt-3">
                <p className="text-2xl font-semibold text-white">{value}</p>
                <p className="text-sm text-gray-500 mt-0.5">{label}</p>
            </div>
        </motion.div>
    )
}

StatCard.propTypes = {
    icon: PropTypes.elementType.isRequired,
    label: PropTypes.string.isRequired,
    value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    trend: PropTypes.number,
    className: PropTypes.string,
}
