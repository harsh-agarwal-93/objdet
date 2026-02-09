import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Input from './Input'

describe('Input', () => {
    it('renders with label', () => {
        render(<Input label="Test Label" />)
        expect(screen.getByText('Test Label')).toBeDefined()
    })

    it('renders without label', () => {
        render(<Input placeholder="No Label" />)
        expect(screen.queryByText('Test Label')).toBeNull()
        expect(screen.getByPlaceholderText('No Label')).toBeDefined()
    })

    it('applies custom className', () => {
        const { container } = render(<Input className="custom-class" />)
        const input = container.querySelector('input')
        expect(input.className).toContain('custom-class')
    })
})
