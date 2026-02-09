import { useState } from 'react'
import { motion } from 'framer-motion'
import {
    Package,
    Upload,
    Play,
    Filter,
    Grid,
    Camera,
    Settings,
} from 'lucide-react'
import { Card, Button, Input } from '../components/ui'

// Mock CAD models (matching EffectFactory)
const mockCADModels = [
    { id: 1, name: 'Boeing 737-800', category: 'Planes', vertices: '245K', textures: true, icon: '✈️' },
    { id: 2, name: 'Airbus A320', category: 'Planes', vertices: '238K', textures: true, icon: '✈️' },
    { id: 3, name: 'F-22 Raptor', category: 'Planes', vertices: '312K', textures: true, icon: '✈️' },
    { id: 4, name: 'Tesla Model S', category: 'Cars', vertices: '189K', textures: true, icon: '🚗' },
    { id: 5, name: 'Ford F-150', category: 'Trucks', vertices: '178K', textures: true, icon: '🛻' },
    { id: 6, name: 'M1 Abrams Tank', category: 'Military', vertices: '285K', textures: true, icon: '🪖' },
    { id: 7, name: 'Cargo Ship', category: 'Ships', vertices: '356K', textures: true, icon: '🚢' },
    { id: 8, name: 'Office Building', category: 'Buildings', vertices: '425K', textures: true, icon: '🏢' },
]

const categories = ['All', 'Planes', 'Cars', 'Trucks', 'Military', 'Ships', 'Buildings']

export default function SyntheticData() {
    const [activeTab, setActiveTab] = useState('cad')
    const [selectedCategory, setSelectedCategory] = useState('All')
    const [selectedModel, setSelectedModel] = useState(null)

    // Generation config
    const [config, setConfig] = useState({
        azimuth: 180,
        elevation: 45,
        rangeBins: 512,
        crossrangeBins: 512,
        pixelRes: 640,
        segMasks: true,
        bgNoise: true,
        jsonExport: true,
    })

    const filteredModels = selectedCategory === 'All'
        ? mockCADModels
        : mockCADModels.filter(m => m.category === selectedCategory)

    const tabs = [
        { id: 'cad', label: '🎨 CAD Models' },
        { id: 'config', label: '⚙️ Generation Config' },
        { id: 'jobs', label: '📊 Previous Jobs' },
    ]

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white">📦 Synthetic Data Generation</h1>
                <p className="text-gray-400 text-sm mt-1">Generate training data from CAD models</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
                <Card className="p-4">
                    <p className="text-2xl font-bold text-white">{mockCADModels.length}</p>
                    <p className="text-sm text-gray-400">Total CAD Models</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-white">{categories.length - 1}</p>
                    <p className="text-sm text-gray-400">Categories</p>
                </Card>
                <Card className="p-4">
                    <p className="text-2xl font-bold text-neon-teal">0</p>
                    <p className="text-sm text-gray-400">Active Jobs</p>
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

            {/* CAD Models Tab */}
            {activeTab === 'cad' && (
                <div className="space-y-4">
                    {/* Category Filter + Upload */}
                    <div className="flex items-center justify-between">
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
                        <Button variant="secondary">
                            <Upload className="w-4 h-4 mr-2" />
                            Upload CAD Model
                        </Button>
                    </div>

                    {/* Models Grid */}
                    <div className="grid grid-cols-4 gap-4">
                        {filteredModels.map((model) => (
                            <motion.button
                                key={model.id}
                                whileHover={{ y: -4 }}
                                onClick={() => setSelectedModel(model)}
                                className={`bg-midnight-800/50 border rounded-xl p-4 text-left transition-colors ${selectedModel?.id === model.id
                                    ? 'border-neon-teal'
                                    : 'border-midnight-700 hover:border-neon-teal/30'
                                    }`}
                            >
                                <div className="text-4xl mb-3">{model.icon}</div>
                                <h3 className="font-medium text-white text-sm mb-1">{model.name}</h3>
                                <p className="text-xs text-gray-500">{model.category}</p>
                                <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                                    <Grid className="w-3 h-3" />
                                    {model.vertices} vertices
                                </div>
                            </motion.button>
                        ))}
                    </div>

                    {selectedModel && (
                        <Card className="p-4 mt-4">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <span className="text-3xl">{selectedModel.icon}</span>
                                    <div>
                                        <h3 className="font-medium text-white">{selectedModel.name}</h3>
                                        <p className="text-sm text-gray-400">{selectedModel.vertices} vertices</p>
                                    </div>
                                </div>
                                <Button onClick={() => setActiveTab('config')}>
                                    Configure Generation
                                </Button>
                            </div>
                        </Card>
                    )}
                </div>
            )}

            {/* Config Tab */}
            {activeTab === 'config' && (
                <div className="grid grid-cols-2 gap-6">
                    <Card className="p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Camera className="w-5 h-5 text-neon-teal" />
                            Camera Angles
                        </h2>
                        <div className="space-y-4">
                            <div>
                                <label className="text-sm text-gray-400 mb-2 block">Azimuth: {config.azimuth}°</label>
                                <input
                                    type="range"
                                    min={0}
                                    max={360}
                                    value={config.azimuth}
                                    onChange={(e) => setConfig({ ...config, azimuth: Number.parseInt(e.target.value) })}
                                    className="w-full"
                                />
                            </div>
                            <div>
                                <label className="text-sm text-gray-400 mb-2 block">Elevation: {config.elevation}°</label>
                                <input
                                    type="range"
                                    min={-90}
                                    max={90}
                                    value={config.elevation}
                                    onChange={(e) => setConfig({ ...config, elevation: Number.parseInt(e.target.value) })}
                                    className="w-full"
                                />
                            </div>
                        </div>
                    </Card>

                    <Card className="p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Settings className="w-5 h-5 text-neon-teal" />
                            Simulation Parameters
                        </h2>
                        <div className="space-y-4">
                            <Input
                                label="Range Bins"
                                type="number"
                                value={config.rangeBins}
                                onChange={(e) => setConfig({ ...config, rangeBins: Number.parseInt(e.target.value) })}
                            />
                            <Input
                                label="Crossrange Bins"
                                type="number"
                                value={config.crossrangeBins}
                                onChange={(e) => setConfig({ ...config, crossrangeBins: Number.parseInt(e.target.value) })}
                            />
                            <Input
                                label="Pixel Resolution"
                                type="number"
                                value={config.pixelRes}
                                onChange={(e) => setConfig({ ...config, pixelRes: Number.parseInt(e.target.value) })}
                            />
                        </div>
                    </Card>

                    <Card className="p-6 col-span-2">
                        <h2 className="text-lg font-semibold text-white mb-4">Render Options</h2>
                        <div className="flex gap-6">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={config.segMasks}
                                    onChange={(e) => setConfig({ ...config, segMasks: e.target.checked })}
                                    className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal"
                                />
                                <span className="text-sm text-gray-400">Segmentation Masks</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={config.bgNoise}
                                    onChange={(e) => setConfig({ ...config, bgNoise: e.target.checked })}
                                    className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal"
                                />
                                <span className="text-sm text-gray-400">Background Noise</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={config.jsonExport}
                                    onChange={(e) => setConfig({ ...config, jsonExport: e.target.checked })}
                                    className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal"
                                />
                                <span className="text-sm text-gray-400">Export JSON</span>
                            </label>
                        </div>
                        <Button className="w-full mt-6">
                            <Play className="w-4 h-4 mr-2" />
                            Generate Data
                        </Button>
                    </Card>
                </div>
            )}

            {/* Jobs Tab */}
            {activeTab === 'jobs' && (
                <Card className="p-8 text-center">
                    <Package className="w-12 h-12 text-midnight-600 mx-auto mb-3" />
                    <p className="text-gray-400">No synthetic data jobs available</p>
                    <p className="text-sm text-gray-500 mt-1">Configure and generate data to see jobs here</p>
                </Card>
            )}
        </div>
    )
}
