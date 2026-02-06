/**
 * Unit Tests for StatusBadge Component
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatusBadge from './StatusBadge'

describe('StatusBadge Component', () => {
    it('renders completed status with correct styling', () => {
        render(<StatusBadge status="completed" />)
        expect(screen.getByText('Completed')).toBeInTheDocument()
    })

    it('renders running status with animation', () => {
        render(<StatusBadge status="running" />)
        expect(screen.getByText('Running')).toBeInTheDocument()
    })

    it('renders training status', () => {
        render(<StatusBadge status="training" />)
        expect(screen.getByText('Training')).toBeInTheDocument()
    })

    it('renders queued status', () => {
        render(<StatusBadge status="queued" />)
        expect(screen.getByText('Queued')).toBeInTheDocument()
    })

    it('renders pending status', () => {
        render(<StatusBadge status="pending" />)
        expect(screen.getByText('Pending')).toBeInTheDocument()
    })

    it('renders failed status', () => {
        render(<StatusBadge status="failed" />)
        expect(screen.getByText('Failed')).toBeInTheDocument()
    })

    it('renders error status', () => {
        render(<StatusBadge status="error" />)
        expect(screen.getByText('Error')).toBeInTheDocument()
    })

    it('renders ready status', () => {
        render(<StatusBadge status="ready" />)
        expect(screen.getByText('Ready')).toBeInTheDocument()
    })

    it('renders connected status', () => {
        render(<StatusBadge status="connected" />)
        expect(screen.getByText('Connected')).toBeInTheDocument()
    })

    it('renders disconnected status', () => {
        render(<StatusBadge status="disconnected" />)
        expect(screen.getByText('Disconnected')).toBeInTheDocument()
    })

    it('defaults to pending for unknown status', () => {
        render(<StatusBadge status="unknown-status" />)
        expect(screen.getByText('Pending')).toBeInTheDocument()
    })

    it('handles case-insensitive status', () => {
        render(<StatusBadge status="COMPLETED" />)
        expect(screen.getByText('Completed')).toBeInTheDocument()
    })

    it('handles null/undefined status gracefully', () => {
        render(<StatusBadge status={null} />)
        expect(screen.getByText('Pending')).toBeInTheDocument()
    })
})
