import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Brain,
    Zap,
    Package,
    Layers,
    FolderOpen,
    Home as HomeIcon,
    Activity,
} from 'lucide-react'

// Import pages
import Home from './pages/Home'
import Models from './pages/Models'
import Effects from './pages/Effects'
import SyntheticData from './pages/SyntheticData'
import SceneForge from './pages/SceneForge'
import Loadset from './pages/Loadset'

// Navigation items
const navItems = [
    { id: 'home', label: 'Home', icon: HomeIcon },
    { id: 'models', label: 'Models', icon: Brain },
    { id: 'effects', label: 'Effects', icon: Zap },
    { id: 'synthetic', label: 'Synthetic Data', icon: Package },
    { id: 'sceneforge', label: 'SceneForge', icon: Layers },
    { id: 'loadset', label: 'Loadset', icon: FolderOpen },
]

function App() {
    const [activePage, setActivePage] = useState('home')

    const renderPage = () => {
        switch (activePage) {
            case 'home':
                return <Home onNavigate={setActivePage} />
            case 'models':
                return <Models />
            case 'effects':
                return <Effects />
            case 'synthetic':
                return <SyntheticData />
            case 'sceneforge':
                return <SceneForge />
            case 'loadset':
                return <Loadset />
            default:
                return <Home onNavigate={setActivePage} />
        }
    }

    return (
        <div className="min-h-screen bg-midnight-950 grid-bg">
            {/* Sidebar */}
            <nav className="fixed left-0 top-0 h-full w-64 bg-midnight-900/80 backdrop-blur-sm border-r border-midnight-700 p-4 flex flex-col">
                {/* Logo */}
                <div className="flex items-center gap-3 mb-8 px-2">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-teal to-neon-cyan flex items-center justify-center">
                        <Activity className="w-6 h-6 text-midnight-950" />
                    </div>
                    <div>
                        <h1 className="text-lg font-semibold text-white">ObjDet</h1>
                        <p className="text-xs text-gray-500">ML Platform</p>
                    </div>
                </div>

                {/* Nav Items */}
                <div className="flex-1 space-y-1">
                    {navItems.map((item) => {
                        const Icon = item.icon
                        const isActive = activePage === item.id
                        return (
                            <motion.button
                                key={item.id}
                                onClick={() => setActivePage(item.id)}
                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${isActive
                                        ? 'bg-neon-teal/10 text-neon-teal'
                                        : 'text-gray-400 hover:text-white hover:bg-midnight-800'
                                    }`}
                                whileHover={{ x: 4 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                <Icon className="w-5 h-5" />
                                <span className="text-sm font-medium">{item.label}</span>
                                {isActive && (
                                    <motion.div
                                        layoutId="activeIndicator"
                                        className="ml-auto w-1.5 h-1.5 rounded-full bg-neon-teal"
                                    />
                                )}
                            </motion.button>
                        )
                    })}
                </div>

                {/* System Status Footer */}
                <div className="pt-4 border-t border-midnight-700">
                    <div className="flex items-center gap-2 px-3 py-2">
                        <span className="w-2 h-2 rounded-full bg-neon-green status-pulse" />
                        <span className="text-xs text-gray-500">System Online</span>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="ml-64 min-h-screen">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activePage}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.2 }}
                        className="p-8"
                    >
                        {renderPage()}
                    </motion.div>
                </AnimatePresence>
            </main>
        </div>
    )
}

export default App
