/**
 * Unit Tests for Card Component
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Card from './Card'

describe('Card Component', () => {
    it('renders with children content', () => {
        render(<Card>Card content</Card>)
        expect(screen.getByText('Card content')).toBeInTheDocument()
    })

    it('applies default styling classes', () => {
        render(<Card data-testid="card">Content</Card>)
        const card = screen.getByTestId('card')
        expect(card).toHaveClass('bg-midnight-800/50')
        expect(card).toHaveClass('border-midnight-700')
        expect(card).toHaveClass('rounded-xl')
    })

    it('applies hover classes by default', () => {
        render(<Card data-testid="card">Content</Card>)
        const card = screen.getByTestId('card')
        expect(card).toHaveClass('card-hover')
    })

    it('removes hover classes when hover is false', () => {
        render(
            <Card data-testid="card" hover={false}>
                Content
            </Card>
        )
        const card = screen.getByTestId('card')
        expect(card).not.toHaveClass('card-hover')
    })

    it('applies custom className', () => {
        render(
            <Card data-testid="card" className="custom-class">
                Content
            </Card>
        )
        const card = screen.getByTestId('card')
        expect(card).toHaveClass('custom-class')
    })

    it('passes through additional props', () => {
        render(
            <Card data-testid="card" id="test-card" aria-label="Test card">
                Content
            </Card>
        )
        const card = screen.getByTestId('card')
        expect(card).toHaveAttribute('id', 'test-card')
        expect(card).toHaveAttribute('aria-label', 'Test card')
    })
})
