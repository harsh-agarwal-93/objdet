import { CheckCircle2, Clock, AlertCircle, Loader2, XCircle } from 'lucide-react'

const statusConfig = {
    completed: {
        icon: CheckCircle2,
        color: 'text-neon-green',
        bg: 'bg-neon-green/10',
        label: 'Completed'
    },
    running: {
        icon: Loader2,
        color: 'text-neon-cyan',
        bg: 'bg-neon-cyan/10',
        label: 'Running',
        animate: true
    },
    training: {
        icon: Loader2,
        color: 'text-neon-cyan',
        bg: 'bg-neon-cyan/10',
        label: 'Training',
        animate: true
    },
    queued: {
        icon: Clock,
        color: 'text-yellow-400',
        bg: 'bg-yellow-400/10',
        label: 'Queued'
    },
    pending: {
        icon: Clock,
        color: 'text-yellow-400',
        bg: 'bg-yellow-400/10',
        label: 'Pending'
    },
    failed: {
        icon: XCircle,
        color: 'text-red-400',
        bg: 'bg-red-400/10',
        label: 'Failed'
    },
    error: {
        icon: AlertCircle,
        color: 'text-red-400',
        bg: 'bg-red-400/10',
        label: 'Error'
    },
    ready: {
        icon: CheckCircle2,
        color: 'text-neon-green',
        bg: 'bg-neon-green/10',
        label: 'Ready'
    },
    connected: {
        icon: CheckCircle2,
        color: 'text-neon-green',
        bg: 'bg-neon-green/10',
        label: 'Connected'
    },
    disconnected: {
        icon: XCircle,
        color: 'text-red-400',
        bg: 'bg-red-400/10',
        label: 'Disconnected'
    },
}

import PropTypes from 'prop-types'

export default function StatusBadge({ status }) {
    const config = statusConfig[status?.toLowerCase()] || statusConfig.pending
    const Icon = config.icon

    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.bg} ${config.color}`}>
            <Icon className={`w-3.5 h-3.5 ${config.animate ? 'animate-spin' : ''}`} />
            {config.label}
        </span>
    )
}

StatusBadge.propTypes = {
    status: PropTypes.string.isRequired,
}
