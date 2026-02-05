/**
 * Test Setup for Vitest
 * Configures MSW and testing-library matchers
 */
import '@testing-library/jest-dom'
import { beforeAll, afterEach, afterAll } from 'vitest'
import { server } from './mocks/server'

// Start MSW server before all tests
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))

// Reset handlers after each test to avoid test pollution
afterEach(() => server.resetHandlers())

// Clean up after all tests are done
afterAll(() => server.close())
