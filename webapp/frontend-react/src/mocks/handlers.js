/**
 * MSW Request Handlers for API Mocking
 * Mirrors the endpoints from client.js
 */
import { http, HttpResponse } from 'msw'

const BASE_URL = 'http://localhost:8000'

// Mock data
const mockExperiments = [
    { experiment_id: 'exp-1', name: 'YOLOv8 Detection', lifecycle_stage: 'active' },
    { experiment_id: 'exp-2', name: 'RetinaNet Training', lifecycle_stage: 'active' },
]

const mockRuns = [
    {
        run_id: 'run-1',
        experiment_id: 'exp-1',
        status: 'RUNNING',
        start_time: Date.now() - 3600000,
        metrics: { mAP: 0.75, loss: 0.15 },
    },
    {
        run_id: 'run-2',
        experiment_id: 'exp-1',
        status: 'FINISHED',
        start_time: Date.now() - 86400000,
        end_time: Date.now() - 82800000,
        metrics: { mAP: 0.82, loss: 0.08 },
    },
]

const mockActiveTasks = [
    { task_id: 'task-1', state: 'RUNNING', name: 'Training YOLOv8' },
]

const mockSystemStatus = {
    celery: { status: 'connected', workers: 2 },
    mlflow: { status: 'connected', tracking_uri: 'http://localhost:5000' },
    version: '0.1.0',
}

export const handlers = [
    // ============ Training Endpoints ============

    // Submit Training Job
    http.post(`${BASE_URL}/api/training/submit`, async ({ request }) => {
        const config = await request.json()
        return HttpResponse.json({
            task_id: 'task-' + Math.random().toString(36).substring(7),
            status: 'PENDING',
            config,
        })
    }),

    // Get Task Status
    http.get(`${BASE_URL}/api/training/status/:taskId`, ({ params }) => {
        const { taskId } = params
        return HttpResponse.json({
            task_id: taskId,
            state: 'RUNNING',
            progress: 45,
            current_epoch: 45,
            total_epochs: 100,
        })
    }),

    // Cancel Task
    http.post(`${BASE_URL}/api/training/cancel/:taskId`, ({ params }) => {
        const { taskId } = params
        return HttpResponse.json({
            task_id: taskId,
            state: 'REVOKED',
            message: 'Task cancelled successfully',
        })
    }),

    // List Active Tasks
    http.get(`${BASE_URL}/api/training/active`, () => {
        return HttpResponse.json(mockActiveTasks)
    }),

    // ============ MLFlow Endpoints ============

    // List Experiments
    http.get(`${BASE_URL}/api/mlflow/experiments`, () => {
        return HttpResponse.json(mockExperiments)
    }),

    // List Runs
    http.get(`${BASE_URL}/api/mlflow/runs`, ({ request }) => {
        const url = new URL(request.url)
        const experimentId = url.searchParams.get('experiment_id')
        const maxResults = parseInt(url.searchParams.get('max_results') || '50')

        let filteredRuns = mockRuns
        if (experimentId) {
            filteredRuns = mockRuns.filter((run) => run.experiment_id === experimentId)
        }
        return HttpResponse.json(filteredRuns.slice(0, maxResults))
    }),

    // Get Run Details
    http.get(`${BASE_URL}/api/mlflow/runs/:runId`, ({ params }) => {
        const { runId } = params
        const run = mockRuns.find((r) => r.run_id === runId)
        if (run) {
            return HttpResponse.json(run)
        }
        return HttpResponse.json({ detail: 'Run not found' }, { status: 404 })
    }),

    // Get Run Metrics
    http.get(`${BASE_URL}/api/mlflow/runs/:runId/metrics`, ({ params }) => {
        const { runId } = params
        return HttpResponse.json({
            run_id: runId,
            metrics: [
                { key: 'mAP', value: 0.75, step: 100 },
                { key: 'loss', value: 0.15, step: 100 },
            ],
        })
    }),

    // List Artifacts
    http.get(`${BASE_URL}/api/mlflow/runs/:runId/artifacts`, ({ params }) => {
        const { runId } = params
        return HttpResponse.json({
            run_id: runId,
            artifacts: [
                { path: 'model.pth', is_dir: false, file_size: 1024000 },
                { path: 'checkpoints/', is_dir: true },
            ],
        })
    }),

    // ============ System Endpoints ============

    // Get System Status
    http.get(`${BASE_URL}/api/system/status`, () => {
        return HttpResponse.json(mockSystemStatus)
    }),
]

export { mockExperiments, mockRuns, mockActiveTasks, mockSystemStatus }
