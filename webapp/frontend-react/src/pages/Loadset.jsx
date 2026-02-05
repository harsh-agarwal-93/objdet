import { useState } from 'react'
import {
    FolderOpen,
    Package,
    Plus,
    Download,
    Trash2,
    Clock,
} from 'lucide-react'
import { Card, Button, Input, Select, StatusBadge } from '../components/ui'

// Mock loadsets
const mockLoadsets = [
    {
        id: 1,
        name: 'production-v1.0',
        version: '1.0.0',
        status: 'ready',
        models: ['YOLOv8-Detection', 'RetinaNet-v2'],
        effects: ['Circle', 'Square'],
        createdAt: '2024-12-01',
    },
    {
        id: 2,
        name: 'staging-v2.0-beta',
        version: '2.0.0-beta',
        status: 'building',
        models: ['YOLOv8-Detection'],
        effects: ['Triangle', 'Hexagon'],
        createdAt: '2024-12-03',
    },
]

const modelOptions = [
    { value: 'yolov8', label: 'YOLOv8-Detection' },
    { value: 'retinanet', label: 'RetinaNet-v2' },
    { value: 'fcos', label: 'FCOS-v3' },
]

const effectOptions = [
    { value: 'circle', label: 'Circle Effect' },
    { value: 'square', label: 'Square Effect' },
    { value: 'triangle', label: 'Triangle Effect' },
]

const platformOptions = [
    { value: 'docker', label: 'Docker' },
    { value: 'edge', label: 'Edge Device' },
    { value: 'cloud', label: 'Cloud' },
]

const optimizationOptions = [
    { value: 'speed', label: 'Speed' },
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'balanced', label: 'Balanced' },
]

export default function Loadset() {
    const [activeTab, setActiveTab] = useState('list')
    const [loadsets] = useState(mockLoadsets)

    const [formData, setFormData] = useState({
        name: 'production-v1.1',
        models: [],
        effects: [],
        platform: 'docker',
        optimization: 'balanced',
    })

    const tabs = [
        { id: 'list', label: '📦 Loadsets' },
        { id: 'create', label: '➕ Create New' },
    ]

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white">📁 Loadset Builder</h1>
                <p className="text-gray-400 text-sm mt-1">Package models, effects, and configurations for deployment</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="p-4">
                    <p className="text-2xl font-bold text-white">{loadsets.length}</p>
                    <p className="text-sm text-gray-400">Total Loadsets</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-neon-green">{loadsets.filter(l => l.status === 'ready').length}</p>
                    <p className="text-sm text-gray-400">Ready</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-neon-cyan">{loadsets.filter(l => l.status === 'building').length}</p>
                    <p className="text-sm text-gray-400">Building</p>
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

            {/* Loadsets List */}
            {activeTab === 'list' && (
                <div className="space-y-4">
                    {loadsets.length === 0 ? (
                        <Card className="p-8 text-center">
                            <FolderOpen className="w-12 h-12 text-midnight-600 mx-auto mb-3" />
                            <p className="text-gray-400">No loadsets created yet</p>
                            <p className="text-sm text-gray-500 mt-1">Create a loadset to package your models for deployment</p>
                        </Card>
                    ) : (
                        loadsets.map((loadset) => (
                            <Card key={loadset.id} className="p-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="p-3 rounded-lg bg-neon-teal/10">
                                            <Package className="w-6 h-6 text-neon-teal" />
                                        </div>
                                        <div>
                                            <h3 className="font-medium text-white">{loadset.name}</h3>
                                            <p className="text-xs text-gray-500">Version {loadset.version}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <StatusBadge status={loadset.status} />
                                        <span className="text-xs text-gray-500 flex items-center gap-1">
                                            <Clock className="w-3 h-3" />
                                            {loadset.createdAt}
                                        </span>
                                    </div>
                                </div>

                                <div className="mt-4 pt-4 border-t border-midnight-700">
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <p className="text-gray-400 mb-1">Models</p>
                                            <div className="flex flex-wrap gap-1">
                                                {loadset.models.map((model) => (
                                                    <span key={model} className="px-2 py-0.5 bg-midnight-700 rounded text-xs text-white">
                                                        {model}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <p className="text-gray-400 mb-1">Effects</p>
                                            <div className="flex flex-wrap gap-1">
                                                {loadset.effects.map((effect) => (
                                                    <span key={effect} className="px-2 py-0.5 bg-midnight-700 rounded text-xs text-white">
                                                        {effect}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex gap-2 mt-4">
                                    <Button size="sm" variant="secondary" disabled={loadset.status !== 'ready'}>
                                        <Download className="w-3 h-3 mr-1" />
                                        Download
                                    </Button>
                                    <Button size="sm" variant="ghost">
                                        <Trash2 className="w-3 h-3 mr-1" />
                                        Delete
                                    </Button>
                                </div>
                            </Card>
                        ))
                    )}
                </div>
            )}

            {/* Create Tab */}
            {activeTab === 'create' && (
                <Card className="p-6">
                    <h2 className="text-lg font-semibold text-white mb-6">Create New Loadset</h2>
                    <form className="space-y-6">
                        <Input
                            label="Loadset Name"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            placeholder="e.g., production-v1.0"
                        />

                        <div className="grid grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-2">Select Models</label>
                                <div className="space-y-2">
                                    {modelOptions.map((model) => (
                                        <label key={model.value} className="flex items-center gap-2 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal"
                                            />
                                            <span className="text-sm text-gray-400">{model.label}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-2">Select Effects</label>
                                <div className="space-y-2">
                                    {effectOptions.map((effect) => (
                                        <label key={effect.value} className="flex items-center gap-2 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal"
                                            />
                                            <span className="text-sm text-gray-400">{effect.label}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-6">
                            <Select
                                label="Target Platform"
                                options={platformOptions}
                                value={formData.platform}
                                onChange={(e) => setFormData({ ...formData, platform: e.target.value })}
                            />
                            <Select
                                label="Optimization"
                                options={optimizationOptions}
                                value={formData.optimization}
                                onChange={(e) => setFormData({ ...formData, optimization: e.target.value })}
                            />
                        </div>

                        <Button className="w-full">
                            <Plus className="w-4 h-4 mr-2" />
                            Build Loadset
                        </Button>
                    </form>
                </Card>
            )}
        </div>
    )
}
