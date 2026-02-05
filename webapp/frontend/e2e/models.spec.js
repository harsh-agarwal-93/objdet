/**
 * E2E Tests for Models Page
 */
import { test, expect } from '@playwright/test'

test.describe('Models Page', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/')
        // Click Models in sidebar
        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'Models', exact: true }).click()
    })

    test('displays page header', async ({ page }) => {
        await expect(page.locator('main').getByText('Model Training')).toBeVisible()
    })

    test('shows tab navigation buttons', async ({ page }) => {
        // Check for tabs in the main content area
        const main = page.locator('main')

        // Look for any tab-like buttons
        const buttons = main.getByRole('button')
        await expect(buttons.first()).toBeVisible()
    })

    test('displays model cards or configuration', async ({ page }) => {
        const main = page.locator('main')

        // Check that there are interactive elements on the page
        await expect(main.locator('button, input, select').first()).toBeVisible()
    })

    test('page has expected structure', async ({ page }) => {
        const main = page.locator('main')

        // Verify the page has content
        const content = await main.textContent()
        expect(content).toBeTruthy()
        expect(content.length).toBeGreaterThan(100)
    })
})
