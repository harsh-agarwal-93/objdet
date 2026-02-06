import { useState } from 'react'
import { motion } from 'framer-motion'
import {
    Play,
    AlertCircle,
    Filter,
} from 'lucide-react'
import { Card, Button, Input, Select, StatusBadge, ProgressBar } from '../components/ui'

// Mock effects library (matching EffectFactory)
const mockEffects = [
    { id: 1, name: 'Circle Effect', category: 'Basic', status: 'ready', successRate: 96.2, transparency: 85.0, samples: 15000, icon: '●' },
    { id: 2, name: 'Square Effect', category: 'Basic', status: 'ready', successRate: 98.1, transparency: 88.5, samples: 12000, icon: '■' },
    { id: 3, name: 'Triangle Effect', category: 'Polygon', status: 'ready', successRate: 95.5, transparency: 80.2, samples: 11000, icon: '▲' },
    { id: 4, name: 'Hexagon Effect', category: 'Polygon', status: 'training', successRate: 88.0, transparency: 75.0, samples: 8000, icon: '⬡' },
    { id: 5, name: 'Star Effect', category: 'Complex', status: 'ready', successRate: 94.2, transparency: 90.5, samples: 14000, icon: '★' },
    { id: 6, name: 'Cross Effect', category: 'Complex', status: 'failed', successRate: 72.3, transparency: 65.9, samples: 5000, icon: '✚' },
]

const categories = ['All', 'Basic', 'Polygon', 'Complex']

const effectTypes = [
    { value: 'Basic', label: 'Basic' },
    { value: 'Polygon', label: 'Polygon' },
    { value: 'Complex', label: 'Complex' },
]

export default function Effects() {
    const [activeTab, setActiveTab] = useState('library')
    const [selectedCategory, setSelectedCategory] = useState('All')
    const [effects] = useState(mockEffects)

    const filteredEffects = selectedCategory === 'All'
        ? effects
        : effects.filter(e => e.category === selectedCategory)

    const tabs = [
        { id: 'library', label: '📚 Effect Library' },
        { id: 'train', label: '✨ Train New Effect' },
        { id: 'validation', label: '🧪 Validation Runs' },
    ]

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white">⚡ Effects Training</h1>
                <p className="text-gray-400 text-sm mt-1">Train and manage geometric transformation effects</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="p-4">
                    <p className="text-2xl font-bold text-white">{effects.length}</p>
                    <p className="text-sm text-gray-400">Total Effects</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-neon-green">{effects.filter(e => e.status === 'ready').length}</p>
                    <p className="text-sm text-gray-400">Ready</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-white">
                        {(effects.reduce((acc, e) => acc + e.successRate, 0) / effects.length).toFixed(1)}%
                    </p>
                    <p className="text-sm text-gray-400">Avg Success Rate</p>
                </Card>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-midnight-700 pb-2">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${activeTab === tab.id
                            ? 'bg-midnight-800 text-neon-teal border-b-2 border-neon-teal'
                            : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Library Tab */}
            {activeTab === 'library' && (
                <div className="space-y-4">
                    {/* Category Filter */}
                    <div className="flex gap-2">
                        <Filter className="w-5 h-5 text-gray-400" />
                        {categories.map((cat) => (
                            <button
                                key={cat}
                                onClick={() => setSelectedCategory(cat)}
                                className={`px-3 py-1 text-sm rounded-full transition-colors ${selectedCategory === cat
                                    ? 'bg-neon-teal/20 text-neon-teal'
                                    : 'bg-midnight-800 text-gray-400 hover:text-white'
                                    }`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>

                    {/* Effects Grid */}
                    <div className="grid grid-cols-3 gap-4">
                        {filteredEffects.map((effect) => (
                            <motion.div
                                key={effect.id}
                                whileHover={{ y: -4 }}
                                className="bg-midnight-800/50 border border-midnight-700 rounded-xl p-4 hover:border-neon-teal/30 transition-colors"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div className="text-3xl">{effect.icon}</div>
                                    <StatusBadge status={effect.status} />
                                </div>
                                <h3 className="font-medium text-white mb-1">{effect.name}</h3>
                                <p className="text-xs text-gray-500 mb-3">{effect.category}</p>

                                <div className="space-y-2">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-gray-400">Success Rate</span>
                                        <span className="text-neon-green">{effect.successRate}%</span>
                                    </div>
                                    <ProgressBar value={effect.successRate} />

                                    <div className="flex justify-between text-xs">
                                        <span className="text-gray-400">Transparency</span>
                                        <span className="text-white">{effect.transparency}%</span>
                                    </div>
                                    <ProgressBar value={effect.transparency} />
                                </div>

                                <div className="flex gap-2 mt-4">
                                    <Button size="sm" variant="secondary" className="flex-1">
                                        Validate
                                    </Button>
                                    <Button size="sm" variant="ghost">
                                        Export
                                    </Button>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            )}

            {/* Train Tab */}
            {activeTab === 'train' && (
                <Card className="p-6">
                    <h2 className="text-lg font-semibold text-white mb-6">Train New Effect</h2>
                    <form className="space-y-4">
                        <Input label="Effect Name" placeholder="e.g., Pentagon Effect" />
                        <Select label="Effect Type" options={effectTypes} />

                        <div className="grid grid-cols-2 gap-4">
                            <Input label="Sample Count" type="number" defaultValue={10000} />
                            <Input label="Epochs" type="number" defaultValue={50} />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <Input label="Batch Size" type="number" defaultValue={16} />
                            <Input label="Learning Rate" type="number" step="0.0001" defaultValue={0.0001} />
                        </div>

                        <div className="flex gap-4 pt-4">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input type="checkbox" className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal" defaultChecked />
                                <span className="text-sm text-gray-400">Enable Augmentation</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input type="checkbox" className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal" defaultChecked />
                                <span className="text-sm text-gray-400">Auto Validate</span>
                            </label>
                        </div>

                        <Button className="w-full">
                            <Play className="w-4 h-4 mr-2" />
                            Start Training
                        </Button>
                    </form>
                </Card>
            )}

            {/* Validation Tab */}
            {activeTab === 'validation' && (
                <Card className="p-8 text-center">
                    <AlertCircle className="w-12 h-12 text-midnight-600 mx-auto mb-3" />
                    <p className="text-gray-400">No validation runs available</p>
                    <p className="text-sm text-gray-500 mt-1">Run validation on effects to see results here</p>
                </Card>
            )}
        </div>
    )
}
