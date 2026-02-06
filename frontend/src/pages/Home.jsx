import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    Brain,
    Zap,
    Package,
    Layers,
    FolderOpen,
    ArrowRight,
    Activity,
    Clock,
    CheckCircle2,
    TrendingUp,
} from 'lucide-react'
import { Card, StatCard, Button } from '../components/ui'
import api from '../api/client'

// Quick action cards
const quickActions = [
    { id: 'models', label: 'Train Models', icon: Brain, description: 'Start or monitor training runs' },
    { id: 'effects', label: 'Manage Effects', icon: Zap, description: 'Train and validate effects' },
    { id: 'synthetic', label: 'Generate Data', icon: Package, description: 'Create synthetic datasets' },
    { id: 'sceneforge', label: 'SceneForge', icon: Layers, description: 'Compose and test scenes' },
    { id: 'loadset', label: 'Build Loadset', icon: FolderOpen, description: 'Package for deployment' },
]

// Workflow steps
const workflowSteps = [
    { step: 1, title: 'Train Model', description: 'Train detection models with MLFlow tracking' },
    { step: 2, title: 'Train Effects', description: 'Create geometric transformation effects' },
    { step: 3, title: 'Generate Data', description: 'Create synthetic training data from CAD models' },
    { step: 4, title: 'Test Scenes', description: 'Validate effects against trained models' },
    { step: 5, title: 'Deploy', description: 'Package and export deployment-ready loadsets' },
]

export default function Home({ onNavigate }) {
    const [systemStatus, setSystemStatus] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        api.getSystemStatus()
            .then(setSystemStatus)
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    const celeryStatus = systemStatus?.services?.celery?.status || 'unknown'
    const mlflowStatus = systemStatus?.services?.mlflow?.status || 'unknown'

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">
                    Welcome to <span className="text-glow text-neon-teal">ObjDet Platform</span>
                </h1>
                <p className="text-gray-400">
                    Object Detection Training and Management
                </p>
            </div>

            {/* System Status */}
            <div className="grid grid-cols-4 gap-4">
                <StatCard
                    icon={Activity}
                    label="Celery Status"
                    value={celeryStatus === 'connected' ? 'Online' : 'Offline'}
                    className={loading ? 'animate-pulse' : ''}
                />
                <a
                    href={import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block"
                >
                    <StatCard
                        icon={Brain}
                        label="MLFlow Status"
                        value={mlflowStatus === 'connected' ? 'Online' : 'Offline'}
                        className={`h-full ${loading ? 'animate-pulse' : ''}`}
                    />
                </a>
                <StatCard
                    icon={CheckCircle2}
                    label="Active Jobs"
                    value="0"
                />
                <StatCard
                    icon={TrendingUp}
                    label="Total Runs"
                    value="--"
                />
            </div>

            {/* Workflow Overview */}
            <Card className="p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Guided Workflow</h2>
                <div className="flex items-center justify-between">
                    {workflowSteps.map((item, index) => (
                        <div key={item.step} className="flex items-center">
                            <div className="text-center">
                                <div className="w-10 h-10 rounded-full bg-neon-teal/10 border border-neon-teal/30 flex items-center justify-center text-neon-teal font-semibold mb-2">
                                    {item.step}
                                </div>
                                <p className="text-sm font-medium text-white">{item.title}</p>
                                <p className="text-xs text-gray-500 max-w-[120px]">{item.description}</p>
                            </div>
                            {index < workflowSteps.length - 1 && (
                                <ArrowRight className="w-5 h-5 text-midnight-600 mx-4" />
                            )}
                        </div>
                    ))}
                </div>
            </Card>

            {/* Quick Actions */}
            <div>
                <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
                <div className="grid grid-cols-5 gap-4">
                    {quickActions.map((action) => {
                        const Icon = action.icon
                        return (
                            <motion.button
                                key={action.id}
                                onClick={() => onNavigate(action.id)}
                                whileHover={{ y: -4 }}
                                whileTap={{ scale: 0.98 }}
                                className="bg-midnight-800/50 border border-midnight-700 rounded-xl p-4 text-left hover:border-neon-teal/30 transition-colors group"
                            >
                                <div className="p-2 rounded-lg bg-neon-teal/10 w-fit mb-3 group-hover:bg-neon-teal/20 transition-colors">
                                    <Icon className="w-5 h-5 text-neon-teal" />
                                </div>
                                <p className="text-sm font-medium text-white mb-1">{action.label}</p>
                                <p className="text-xs text-gray-500">{action.description}</p>
                            </motion.button>
                        )
                    })}
                </div>
            </div>

            {/* Recent Activity */}
            <Card className="p-6">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-white">Recent Activity</h2>
                    <Button variant="ghost" size="sm">
                        View All
                    </Button>
                </div>
                <div className="space-y-3">
                    <div className="flex items-center gap-4 p-3 rounded-lg bg-midnight-900/50">
                        <Clock className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-400">No recent activity</span>
                    </div>
                </div>
            </Card>
        </div>
    )
}
