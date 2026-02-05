/**
 * Unit Tests for API Client
 * Uses MSW to mock backend responses
 */
import { describe, it, expect } from 'vitest'
import { api } from './client'
import { mockExperiments, mockRuns, mockActiveTasks, mockSystemStatus } from '../mocks/handlers'

describe('API Client', () => {
    describe('Training Endpoints', () => {
        it('submitTrainingJob sends config and returns task info', async () => {
            const config = {
                name: 'Test Training',
                model_architecture: 'yolov8',
                epochs: 10,
            }

            const result = await api.submitTrainingJob(config)

            expect(result).toHaveProperty('task_id')
            expect(result).toHaveProperty('status', 'PENDING')
            expect(result.config).toEqual(config)
        })

        it('getTaskStatus returns task progress', async () => {
            const result = await api.getTaskStatus('task-123')

            expect(result.task_id).toBe('task-123')
            expect(result.state).toBe('RUNNING')
            expect(result).toHaveProperty('progress')
        })

        it('cancelTask returns revoked status', async () => {
            const result = await api.cancelTask('task-456')

            expect(result.task_id).toBe('task-456')
            expect(result.state).toBe('REVOKED')
        })

        it('listActiveTasks returns active tasks array', async () => {
            const result = await api.listActiveTasks()

            expect(result).toEqual(mockActiveTasks)
        })
    })

    describe('MLFlow Endpoints', () => {
        it('listExperiments returns experiments array', async () => {
            const result = await api.listExperiments()

            expect(result).toEqual(mockExperiments)
        })

        it('listRuns returns runs array', async () => {
            const result = await api.listRuns()

            expect(result).toEqual(mockRuns)
        })

        it('listRuns filters by experiment ID', async () => {
            const result = await api.listRuns({ experimentId: 'exp-1' })

            expect(result).toHaveLength(2)
            expect(result[0].experiment_id).toBe('exp-1')
        })

        it('listRuns respects maxResults limit', async () => {
            const result = await api.listRuns({ maxResults: 1 })

            expect(result).toHaveLength(1)
        })

        it('getRunDetails returns run object', async () => {
            const result = await api.getRunDetails('run-1')

            expect(result.run_id).toBe('run-1')
            expect(result).toHaveProperty('metrics')
        })

        it('getRunMetrics returns metrics history', async () => {
            const result = await api.getRunMetrics('run-1')

            expect(result.run_id).toBe('run-1')
            expect(result.metrics).toBeInstanceOf(Array)
        })

        it('listArtifacts returns artifacts array', async () => {
            const result = await api.listArtifacts('run-1')

            expect(result.run_id).toBe('run-1')
            expect(result.artifacts).toBeInstanceOf(Array)
        })
    })

    describe('System Endpoints', () => {
        it('getSystemStatus returns system info', async () => {
            const result = await api.getSystemStatus()

            expect(result).toEqual(mockSystemStatus)
        })
    })
})
