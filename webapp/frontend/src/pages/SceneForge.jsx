import { useState } from 'react'
import { motion } from 'framer-motion'
import {
    Layers,
    Play,
    Pause,
    Eye,
    EyeOff,
    Upload,
    Settings,
    Brain,
} from 'lucide-react'
import { Card, Button, StatusBadge } from '../components/ui'

// Mock scenes
const mockScenes = [
    { id: 1, name: 'Urban Highway', description: 'Multi-lane highway with vehicles', icon: '🛣️' },
    { id: 2, name: 'Airfield', description: 'Military airbase with aircraft', icon: '🛫' },
    { id: 3, name: 'Harbor', description: 'Commercial port with ships', icon: '⚓' },
    { id: 4, name: 'Industrial Zone', description: 'Factory complex with vehicles', icon: '🏭' },
]

// Mock effects for scene
const mockSceneEffects = [
    { id: 1, name: 'Circle', active: true },
    { id: 2, name: 'Square', active: false },
    { id: 3, name: 'Triangle', active: true },
    { id: 4, name: 'Hexagon', active: false },
]

// Mock models
const mockModels = [
    { id: 1, name: 'YOLOv8-Detection', status: 'ready' },
    { id: 2, name: 'RetinaNet-v2', status: 'ready' },
    { id: 3, name: 'FCOS-v3', status: 'training' },
]

export default function SceneForge() {
    const [selectedScene, setSelectedScene] = useState(mockScenes[0])
    const [effects, setEffects] = useState(mockSceneEffects)
    const [selectedModel, setSelectedModel] = useState(mockModels[0])
    const [showInference, setShowInference] = useState(true)
    const [showEffects, setShowEffects] = useState(true)
    const [isPlaying, setIsPlaying] = useState(false)

    const toggleEffect = (effectId) => {
        setEffects(effects.map(e =>
            e.id === effectId ? { ...e, active: !e.active } : e
        ))
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white">🎬 SceneForge</h1>
                <p className="text-gray-400 text-sm mt-1">Compose scenes and test model inference</p>
            </div>

            <div className="grid grid-cols-4 gap-6">
                {/* Scene Library (Left Panel) */}
                <Card className="p-4 col-span-1">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="font-semibold text-white">Scenes</h2>
                        <Button variant="ghost" size="sm">
                            <Upload className="w-4 h-4" />
                        </Button>
                    </div>
                    <div className="space-y-2">
                        {mockScenes.map((scene) => (
                            <motion.button
                                key={scene.id}
                                onClick={() => setSelectedScene(scene)}
                                whileHover={{ x: 4 }}
                                className={`w-full text-left p-3 rounded-lg transition-colors ${selectedScene.id === scene.id
                                        ? 'bg-neon-teal/10 border border-neon-teal/30'
                                        : 'bg-midnight-800/50 hover:bg-midnight-800'
                                    }`}
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{scene.icon}</span>
                                    <div>
                                        <p className="text-sm font-medium text-white">{scene.name}</p>
                                        <p className="text-xs text-gray-500">{scene.description}</p>
                                    </div>
                                </div>
                            </motion.button>
                        ))}
                    </div>
                </Card>

                {/* Scene Preview (Center) */}
                <Card className="p-4 col-span-2">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="font-semibold text-white">Preview: {selectedScene.name}</h2>
                        <div className="flex gap-2">
                            <Button
                                variant={showInference ? 'primary' : 'secondary'}
                                size="sm"
                                onClick={() => setShowInference(!showInference)}
                            >
                                <Brain className="w-4 h-4 mr-1" />
                                Inference
                            </Button>
                            <Button
                                variant={showEffects ? 'primary' : 'secondary'}
                                size="sm"
                                onClick={() => setShowEffects(!showEffects)}
                            >
                                <Layers className="w-4 h-4 mr-1" />
                                Effects
                            </Button>
                        </div>
                    </div>

                    {/* Preview Area */}
                    <div className="aspect-video bg-midnight-900 rounded-lg flex items-center justify-center border border-midnight-700 mb-4">
                        <div className="text-center">
                            <span className="text-6xl block mb-4">{selectedScene.icon}</span>
                            <p className="text-gray-400">{selectedScene.name}</p>
                            <p className="text-xs text-gray-500 mt-1">Scene preview placeholder</p>
                        </div>
                    </div>

                    {/* Playback Controls */}
                    <div className="flex items-center justify-center gap-4">
                        <Button
                            variant="secondary"
                            onClick={() => setIsPlaying(!isPlaying)}
                        >
                            {isPlaying ? (
                                <>
                                    <Pause className="w-4 h-4 mr-2" />
                                    Pause
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4 mr-2" />
                                    Play
                                </>
                            )}
                        </Button>
                    </div>
                </Card>

                {/* Controls Panel (Right) */}
                <div className="space-y-4">
                    {/* Effects Panel */}
                    <Card className="p-4">
                        <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
                            <Layers className="w-4 h-4 text-neon-teal" />
                            Effects
                        </h2>
                        <div className="space-y-2">
                            {effects.map((effect) => (
                                <button
                                    key={effect.id}
                                    onClick={() => toggleEffect(effect.id)}
                                    className={`w-full flex items-center justify-between p-2 rounded-lg transition-colors ${effect.active
                                            ? 'bg-neon-teal/10 text-neon-teal'
                                            : 'bg-midnight-800/50 text-gray-400'
                                        }`}
                                >
                                    <span className="text-sm">{effect.name}</span>
                                    {effect.active ? (
                                        <Eye className="w-4 h-4" />
                                    ) : (
                                        <EyeOff className="w-4 h-4" />
                                    )}
                                </button>
                            ))}
                        </div>
                    </Card>

                    {/* Model Selection */}
                    <Card className="p-4">
                        <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
                            <Brain className="w-4 h-4 text-neon-teal" />
                            Model
                        </h2>
                        <div className="space-y-2">
                            {mockModels.map((model) => (
                                <button
                                    key={model.id}
                                    onClick={() => setSelectedModel(model)}
                                    disabled={model.status !== 'ready'}
                                    className={`w-full flex items-center justify-between p-2 rounded-lg transition-colors ${selectedModel.id === model.id
                                            ? 'bg-neon-teal/10 border border-neon-teal/30'
                                            : 'bg-midnight-800/50 hover:bg-midnight-800'
                                        } ${model.status !== 'ready' ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    <span className="text-sm text-white">{model.name}</span>
                                    <StatusBadge status={model.status} />
                                </button>
                            ))}
                        </div>
                    </Card>

                    {/* Settings */}
                    <Card className="p-4">
                        <Button variant="secondary" className="w-full">
                            <Settings className="w-4 h-4 mr-2" />
                            Scene Settings
                        </Button>
                    </Card>
                </div>
            </div>
        </div>
    )
}
