import { useState, useEffect } from 'react'
import {
    Play,
    Square,
    ChevronDown,
    ChevronUp,
    RefreshCw,
    Download,
    Clock,
    Cpu,
} from 'lucide-react'
import { motion } from 'framer-motion'
import { Card, Button, Input, Select, StatusBadge, ProgressBar } from '../components/ui'
import api from '../api/client'

// Mock training runs for demo (will be replaced by API data)
const mockRuns = [
    { run_id: '1', run_name: 'YOLOv8-Detection v3', status: 'FINISHED', experiment_id: '1', start_time: Date.now() - 86400000 },
    { run_id: '2', run_name: 'RetinaNet-v2', status: 'RUNNING', experiment_id: '1', start_time: Date.now() - 3600000 },
    { run_id: '3', run_name: 'FCOS-Detection', status: 'FAILED', experiment_id: '1', start_time: Date.now() - 172800000 },
]

const modelArchitectures = [
    { value: 'yolov8', label: 'YOLOv8' },
    { value: 'yolov11', label: 'YOLOv11' },
    { value: 'retinanet', label: 'RetinaNet' },
    { value: 'fcos', label: 'FCOS' },
]

const optimizers = [
    { value: 'adam', label: 'Adam' },
    { value: 'sgd', label: 'SGD' },
    { value: 'adamw', label: 'AdamW' },
]

const gpuOptions = [
    { value: 'auto', label: 'Auto' },
    { value: 'gpu0', label: 'GPU 0' },
    { value: 'gpu1', label: 'GPU 1' },
    { value: 'multi', label: 'Multi-GPU' },
]

const precisionOptions = [
    { value: 'fp16', label: 'FP16' },
    { value: 'fp32', label: 'FP32' },
    { value: 'bf16', label: 'BF16' },
]

export default function Models() {
    const [activeTab, setActiveTab] = useState('runs')
    const [runs, setRuns] = useState([])
    const [activeTasks, setActiveTasks] = useState([])
    const [loading, setLoading] = useState(true)
    const [expandedRun, setExpandedRun] = useState(null)
    const [submitting, setSubmitting] = useState(false)

    // Form state
    const [formData, setFormData] = useState({
        name: 'YOLOv8 Training Run',
        model_architecture: 'yolov8',
        dataset: 'coco2017',
        epochs: 100,
        batch_size: 32,
        learning_rate: 0.001,
        optimizer: 'adam',
        gpu: 'auto',
        mixed_precision: 'fp16',
        save_checkpoints: true,
        early_stopping: true,
        log_to_mlflow: true,
        data_augmentation: true,
    })

    useEffect(() => {
        loadRuns()
        loadActiveTasks()
    }, [])

    const loadRuns = async () => {
        setLoading(true)
        try {
            const response = await api.listRuns({ maxResults: 50 })
            setRuns(response.runs || mockRuns)
        } catch (error) {
            console.error('Failed to load runs:', error)
            setRuns(mockRuns) // Fallback to mock data
        } finally {
            setLoading(false)
        }
    }

    const loadActiveTasks = async () => {
        try {
            const response = await api.listActiveTasks()
            setActiveTasks(response.tasks || [])
        } catch (error) {
            console.error('Failed to load active tasks:', error)
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setSubmitting(true)
        try {
            const response = await api.submitTrainingJob(formData)
            alert(`Training job submitted! Task ID: ${response.task_id}`)
            setActiveTab('active')
            loadActiveTasks()
        } catch (error) {
            alert(`Failed to submit job: ${error.message}`)
        } finally {
            setSubmitting(false)
        }
    }

    const formatDate = (timestamp) => {
        if (!timestamp) return '--'
        return new Date(timestamp).toLocaleString()
    }

    const tabs = [
        { id: 'runs', label: '📊 Previous Runs' },
        { id: 'new', label: '🚀 New Training' },
        { id: 'active', label: `⚡ Active Jobs (${activeTasks.length})` },
    ]

    const renderRunsContent = () => {
        if (loading) {
            return (
                <Card className="p-8 text-center">
                    <RefreshCw className="w-8 h-8 text-neon-teal animate-spin mx-auto mb-2" />
                    <p className="text-gray-400">Loading runs...</p>
                </Card>
            )
        }

        if (runs.length === 0) {
            return (
                <Card className="p-8 text-center">
                    <p className="text-gray-400">No training runs found in MLFlow</p>
                </Card>
            )
        }

        return (
            <>
                {runs.map((run) => (
                    <Card key={run.run_id} className="p-4">
                        <button
                            type="button"
                            className="w-full flex items-center justify-between cursor-pointer text-left"
                            onClick={() => setExpandedRun(expandedRun === run.run_id ? null : run.run_id)}
                        >
                            <div className="flex items-center gap-4">
                                <div>
                                    <h3 className="font-medium text-white">{run.run_name || 'Unnamed Run'}</h3>
                                    <p className="text-xs text-gray-500 font-mono">ID: {run.run_id}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4">
                                <StatusBadge status={run.status} />
                                <span className="text-xs text-gray-500 flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {formatDate(run.start_time)}
                                </span>
                                {expandedRun === run.run_id ? (
                                    <ChevronUp className="w-5 h-5 text-gray-400" />
                                ) : (
                                    <ChevronDown className="w-5 h-5 text-gray-400" />
                                )}
                            </div>
                        </button>

                        {expandedRun === run.run_id && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="mt-4 pt-4 border-t border-midnight-700"
                            >
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <h4 className="text-sm font-medium text-gray-400 mb-2">Parameters</h4>
                                        <p className="text-xs text-gray-500">No parameters logged</p>
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-medium text-gray-400 mb-2">Metrics</h4>
                                        <p className="text-xs text-gray-500">No metrics logged</p>
                                    </div>
                                </div>
                                <div className="flex gap-2 mt-4">
                                    <Button size="sm" variant="secondary">
                                        <Download className="w-3 h-3 mr-1" />
                                        Export
                                    </Button>
                                </div>
                            </motion.div>
                        )}
                    </Card>
                ))}
            </>
        )
    }


    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">🧠 Model Training</h1>
                    <p className="text-gray-400 text-sm mt-1">Train and manage object detection models</p>
                </div>
                <Button variant="secondary" onClick={loadRuns}>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                </Button>
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

            {/* Tab Content */}
            {activeTab === 'runs' && (
                <div className="space-y-4">
                    {renderRunsContent()}
                </div>
            )}

            {activeTab === 'new' && (
                <Card className="p-6">
                    <h2 className="text-lg font-semibold text-white mb-6">Submit New Training Job</h2>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-2 gap-6">
                            {/* Left Column */}
                            <div className="space-y-4">
                                <Input
                                    label="Run Name"
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                />
                                <Select
                                    label="Model Architecture"
                                    options={modelArchitectures}
                                    value={formData.model_architecture}
                                    onChange={(e) => setFormData({ ...formData, model_architecture: e.target.value })}
                                />
                                <Input
                                    label="Dataset"
                                    value={formData.dataset}
                                    onChange={(e) => setFormData({ ...formData, dataset: e.target.value })}
                                />
                                <Input
                                    label="Epochs"
                                    type="number"
                                    min={1}
                                    max={1000}
                                    value={formData.epochs}
                                    onChange={(e) => setFormData({ ...formData, epochs: Number.parseInt(e.target.value) })}
                                />
                                <Input
                                    label="Batch Size"
                                    type="number"
                                    min={1}
                                    max={512}
                                    value={formData.batch_size}
                                    onChange={(e) => setFormData({ ...formData, batch_size: Number.parseInt(e.target.value) })}
                                />
                            </div>

                            {/* Right Column */}
                            <div className="space-y-4">
                                <Input
                                    label="Learning Rate"
                                    type="number"
                                    step="0.0001"
                                    min={0.0001}
                                    max={0.1}
                                    value={formData.learning_rate}
                                    onChange={(e) => setFormData({ ...formData, learning_rate: Number.parseFloat(e.target.value) })}
                                />
                                <Select
                                    label="Optimizer"
                                    options={optimizers}
                                    value={formData.optimizer}
                                    onChange={(e) => setFormData({ ...formData, optimizer: e.target.value })}
                                />
                                <Select
                                    label="GPU Selection"
                                    options={gpuOptions}
                                    value={formData.gpu}
                                    onChange={(e) => setFormData({ ...formData, gpu: e.target.value })}
                                />
                                <Select
                                    label="Mixed Precision"
                                    options={precisionOptions}
                                    value={formData.mixed_precision}
                                    onChange={(e) => setFormData({ ...formData, mixed_precision: e.target.value })}
                                />
                            </div>
                        </div>

                        {/* Checkboxes */}
                        <div className="grid grid-cols-4 gap-4 pt-4 border-t border-midnight-700">
                            {[
                                { key: 'save_checkpoints', label: 'Save Checkpoints' },
                                { key: 'early_stopping', label: 'Early Stopping' },
                                { key: 'log_to_mlflow', label: 'Log to MLFlow' },
                                { key: 'data_augmentation', label: 'Data Augmentation' },
                            ].map((option) => (
                                <label key={option.key} className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={formData[option.key]}
                                        onChange={(e) => setFormData({ ...formData, [option.key]: e.target.checked })}
                                        className="w-4 h-4 rounded bg-midnight-700 border-midnight-600 text-neon-teal focus:ring-neon-teal/50"
                                    />
                                    <span className="text-sm text-gray-400">{option.label}</span>
                                </label>
                            ))}
                        </div>

                        <Button type="submit" className="w-full" disabled={submitting}>
                            {submitting ? (
                                <>
                                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                    Submitting...
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4 mr-2" />
                                    Start Training
                                </>
                            )}
                        </Button>
                    </form>
                </Card>
            )}

            {activeTab === 'active' && (
                <div className="space-y-4">
                    {activeTasks.length === 0 ? (
                        <Card className="p-8 text-center">
                            <Cpu className="w-12 h-12 text-midnight-600 mx-auto mb-3" />
                            <p className="text-gray-400">No active training jobs</p>
                            <p className="text-sm text-gray-500 mt-1">Submit a new training job to get started</p>
                        </Card>
                    ) : (
                        activeTasks.map((task) => (
                            <Card key={task.task_id} className="p-4">
                                <div className="flex items-center justify-between mb-3">
                                    <div>
                                        <h3 className="font-medium text-white">{task.name || 'Training Job'}</h3>
                                        <p className="text-xs text-gray-500 font-mono">Task ID: {task.task_id}</p>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <StatusBadge status={task.status || 'running'} />
                                        <Button variant="danger" size="sm">
                                            <Square className="w-3 h-3 mr-1" />
                                            Cancel
                                        </Button>
                                    </div>
                                </div>
                                <ProgressBar value={task.progress || 0} />
                                <p className="text-xs text-gray-500 mt-2">
                                    {task.current_epoch || 0} / {task.total_epochs || 100} epochs
                                </p>
                            </Card>
                        ))
                    )}
                </div>
            )}
        </div>
    )
}

Models.propTypes = {}
