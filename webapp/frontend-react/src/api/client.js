/**
 * API Client for ObjDet Backend
 * Mirrors the Python BackendClient from the Streamlit app
 */

const BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

/**
 * Helper to handle API responses
 */
async function handleResponse(response) {
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `HTTP ${response.status}`)
    }
    return response.json()
}

/**
 * API methods
 */
export const api = {
    // ============ Training Endpoints ============

    /**
     * Submit a training job
     * @param {Object} config - Training configuration
     */
    submitTrainingJob: (config) =>
        fetch(`${BASE_URL}/api/training/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        }).then(handleResponse),

    /**
     * Get status of a training task
     * @param {string} taskId - Celery task ID
     */
    getTaskStatus: (taskId) =>
        fetch(`${BASE_URL}/api/training/status/${taskId}`).then(handleResponse),

    /**
     * Cancel a running task
     * @param {string} taskId - Celery task ID
     */
    cancelTask: (taskId) =>
        fetch(`${BASE_URL}/api/training/cancel/${taskId}`, {
            method: 'POST',
        }).then(handleResponse),

    /**
     * List all active training tasks
     */
    listActiveTasks: () =>
        fetch(`${BASE_URL}/api/training/active`).then(handleResponse),

    // ============ MLFlow Endpoints ============

    /**
     * List MLFlow experiments
     */
    listExperiments: () =>
        fetch(`${BASE_URL}/api/mlflow/experiments`).then(handleResponse),

    /**
     * List MLFlow runs with optional filtering
     * @param {Object} params - Filter parameters
     */
    listRuns: (params = {}) => {
        const searchParams = new URLSearchParams()
        if (params.experimentId) searchParams.set('experiment_id', params.experimentId)
        if (params.status) searchParams.set('status', params.status)
        if (params.maxResults) searchParams.set('max_results', String(params.maxResults))

        const queryString = searchParams.toString()
        const url = `${BASE_URL}/api/mlflow/runs${queryString ? `?${queryString}` : ''}`
        return fetch(url).then(handleResponse)
    },

    /**
     * Get detailed run information
     * @param {string} runId - MLFlow run ID
     */
    getRunDetails: (runId) =>
        fetch(`${BASE_URL}/api/mlflow/runs/${runId}`).then(handleResponse),

    /**
     * Get run metrics history
     * @param {string} runId - MLFlow run ID
     */
    getRunMetrics: (runId) =>
        fetch(`${BASE_URL}/api/mlflow/runs/${runId}/metrics`).then(handleResponse),

    /**
     * List artifacts for a run
     * @param {string} runId - MLFlow run ID
     * @param {string} path - Artifact path
     */
    listArtifacts: (runId, path = '') => {
        const url = path
            ? `${BASE_URL}/api/mlflow/runs/${runId}/artifacts?path=${encodeURIComponent(path)}`
            : `${BASE_URL}/api/mlflow/runs/${runId}/artifacts`
        return fetch(url).then(handleResponse)
    },

    // ============ System Endpoints ============

    /**
     * Get system status (Celery, MLFlow connections)
     */
    getSystemStatus: () =>
        fetch(`${BASE_URL}/api/system/status`).then(handleResponse),
}

export default api
