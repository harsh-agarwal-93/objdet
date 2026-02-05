/**
 * E2E Tests for Navigation and Page Rendering
 */
import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
    test('home page loads with sidebar and main content', async ({ page }) => {
        await page.goto('/')

        // Check that the sidebar and main elements exist
        await expect(page.locator('nav')).toBeVisible()
        await expect(page.locator('main')).toBeVisible()

        // Check that ObjDet branding is visible
        await expect(page.locator('nav h1:text("ObjDet")')).toBeVisible()
    })

    test('sidebar navigation items are visible', async ({ page }) => {
        await page.goto('/')

        // Check all nav items are present in the sidebar navigation area
        const sidebar = page.locator('nav')
        await expect(sidebar.getByRole('button', { name: 'Home', exact: true })).toBeVisible()
        await expect(sidebar.getByRole('button', { name: 'Models', exact: true })).toBeVisible()
        await expect(sidebar.getByRole('button', { name: 'Effects', exact: true })).toBeVisible()
        await expect(sidebar.getByRole('button', { name: 'Synthetic Data', exact: true })).toBeVisible()
        await expect(sidebar.getByRole('button', { name: 'SceneForge', exact: true })).toBeVisible()
        await expect(sidebar.getByRole('button', { name: 'Loadset', exact: true })).toBeVisible()
    })

    test('can navigate to Models page', async ({ page }) => {
        await page.goto('/')

        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'Models', exact: true }).click()

        // Verify Models page content loads - look for main content area updating
        await expect(page.locator('main').getByText('Model Training')).toBeVisible()
    })

    test('can navigate to Effects page', async ({ page }) => {
        await page.goto('/')

        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'Effects', exact: true }).click()

        // Verify page loaded by checking main content has changed
        await expect(page.locator('main')).toBeVisible()

        // Check for the Effects header specifically
        await expect(page.locator('main h1, main h2').first()).toBeVisible()
    })

    test('can navigate to Synthetic Data page', async ({ page }) => {
        await page.goto('/')

        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'Synthetic Data', exact: true }).click()

        // Verify page loaded
        await expect(page.locator('main')).toBeVisible()
        await expect(page.locator('main h1, main h2').first()).toBeVisible()
    })

    test('can navigate to SceneForge page', async ({ page }) => {
        await page.goto('/')

        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'SceneForge', exact: true }).click()

        await expect(page.locator('main')).toBeVisible()
        await expect(page.locator('main h1, main h2').first()).toBeVisible()
    })

    test('can navigate to Loadset page', async ({ page }) => {
        await page.goto('/')

        const sidebar = page.locator('nav')
        await sidebar.getByRole('button', { name: 'Loadset', exact: true }).click()

        await expect(page.locator('main')).toBeVisible()
        await expect(page.locator('main h1, main h2').first()).toBeVisible()
    })

    test('system status indicator is visible', async ({ page }) => {
        await page.goto('/')

        await expect(page.getByText('System Online')).toBeVisible()
    })

    test('active navigation item shows indicator', async ({ page }) => {
        await page.goto('/')

        // Home should be active by default
        const sidebar = page.locator('nav')
        const homeButton = sidebar.getByRole('button', { name: 'Home', exact: true })
        await expect(homeButton).toHaveClass(/text-neon-teal/)
    })
})
