/**
 * Unit Tests for Button Component
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Button from './Button'

describe('Button Component', () => {
    it('renders with children text', () => {
        render(<Button>Click me</Button>)
        expect(screen.getByText('Click me')).toBeInTheDocument()
    })

    it('applies primary variant classes by default', () => {
        render(<Button>Primary</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('bg-neon-teal')
    })

    it('applies secondary variant classes', () => {
        render(<Button variant="secondary">Secondary</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('bg-midnight-700')
    })

    it('applies danger variant classes', () => {
        render(<Button variant="danger">Danger</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('text-red-400')
    })

    it('applies ghost variant classes', () => {
        render(<Button variant="ghost">Ghost</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('text-gray-400')
    })

    it('applies small size classes', () => {
        render(<Button size="sm">Small</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('px-3', 'py-1.5', 'text-xs')
    })

    it('applies large size classes', () => {
        render(<Button size="lg">Large</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('px-6', 'py-3', 'text-base')
    })

    it('disables button when disabled prop is true', () => {
        render(<Button disabled>Disabled</Button>)
        const button = screen.getByRole('button')
        expect(button).toBeDisabled()
    })

    it('fires onClick handler when clicked', async () => {
        const handleClick = vi.fn()
        render(<Button onClick={handleClick}>Clickable</Button>)
        const button = screen.getByRole('button')

        fireEvent.click(button)

        expect(handleClick).toHaveBeenCalledTimes(1)
    })

    it('does not fire onClick when disabled', async () => {
        const handleClick = vi.fn()
        render(
            <Button disabled onClick={handleClick}>
                Disabled
            </Button>
        )
        const button = screen.getByRole('button')

        fireEvent.click(button)

        expect(handleClick).not.toHaveBeenCalled()
    })

    it('applies custom className', () => {
        render(<Button className="custom-class">Custom</Button>)
        const button = screen.getByRole('button')
        expect(button).toHaveClass('custom-class')
    })
})
