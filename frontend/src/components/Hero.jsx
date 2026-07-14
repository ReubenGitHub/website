import { useState, useEffect, useRef, useCallback } from 'react'
import './Hero.css'

export function Hero() {
    const [mousePos, setMousePos] = useState({ x: 0, y: 0, width: 0, height: 0 })
    const mouseRef = useRef({ x: 0, y: 0, width: 0, height: 0 })
    const rafRef = useRef(null)
    const letterRefs = useRef({})
    const letterCenters = useRef({})
    const baseWidths = useRef({})
    const offsets = useRef({})
    const containerRef = useRef(null)
    const animationRef = useRef(null)
    const timeRef = useRef(0)

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        // Check if device has a precise pointer (mouse/trackpad)
        const hasPointer = window.matchMedia('(pointer: fine)').matches

        if (!hasPointer) {
            // No real cursor — auto-animate a virtual cursor sweeping left→right→left over 4s
            const startTime = Date.now()
            let rafId = null

            const animate = () => {
                const elapsed = (Date.now() - startTime) / 1000
                const cycle = 4 // seconds per full cycle (L→R→L)
                const t = (elapsed % cycle) / cycle

                // 0→0.5 goes L→R, 0.5→1 goes R→L
                const pos = t < 0.5
                    ? 2 * t // 0→1
                    : 2 * (1 - t) // 1→0

                const rect = container.getBoundingClientRect()

                mouseRef.current = {
                    x: rect.left + pos * rect.width,
                    y: rect.top + rect.height / 2,
                    width: rect.width,
                    height: rect.height,
                }

                if (!rafId) {
                    rafId = requestAnimationFrame(() => {
                        setMousePos({ ...mouseRef.current })
                        rafId = null
                    })
                }

                animationRef.current = requestAnimationFrame(animate)
            }

            animationRef.current = requestAnimationFrame(animate)
            return () => {
                if (animationRef.current) cancelAnimationFrame(animationRef.current)
            }
        } else {
            // Real cursor — listen for mousemove
            const handleMouseMove = (e) => {
                const rect = container.getBoundingClientRect()
                mouseRef.current = {
                    x: e.clientX,
                    y: e.clientY,
                    width: rect.width,
                    height: rect.height,
                }
                if (!rafRef.current) {
                    rafRef.current = requestAnimationFrame(() => {
                        setMousePos({ ...mouseRef.current })
                        rafRef.current = null
                    })
                }
            }

            container.addEventListener('mousemove', handleMouseMove)
            return () => {
                container.removeEventListener('mousemove', handleMouseMove)
                if (rafRef.current) cancelAnimationFrame(rafRef.current)
            }
        }
    }, [])

    // Measure base widths and centers once after render
    useEffect(() => {
        letters.forEach((_, i) => {
            const el = letterRefs.current[i]
            if (el) {
                const rect = el.getBoundingClientRect()
                baseWidths.current[i] = rect.width
                letterCenters.current[i] = {
                    x: rect.left + rect.width / 2,
                    y: rect.top + rect.height / 2,
                }
            }
        })
    }, [])

    // Floating shapes animation
    useEffect(() => {
        const startTime = Date.now()
        const animate = () => {
            timeRef.current = (Date.now() - startTime) / 1000
            if (animationRef.current) {
                animationRef.current = requestAnimationFrame(animate)
            }
        }
        animationRef.current = requestAnimationFrame(animate)
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [])

    const letters = 'Hello!'.split('')

    return (
        <div ref={containerRef} className="hero">
            {/* Gradient overlay */}
            <div className="hero-gradient" />

            {/* Content */}
            <div className="hero-content">
                <div className="hero-greeting">
                    {(() => {
                        // First pass: compute all scales using stored positions
                        const scales = letters.map((_, index) => {
                            const center = letterCenters.current[index]
                            if (!center) return 1
                            const baseW = baseWidths.current[index] || 100
                            const dx = mousePos.x - center.x
                            const dy = mousePos.y - center.y
                            const distance = Math.sqrt(dx * dx + dy * dy)
                            const proximityRadius = 0.5 * baseW + 200
                            const proximity = Math.max(0, 1 - distance / proximityRadius)
                            return 1 + proximity * 0.5
                        })

                        // Second pass: compute bidirectional offsets
                        return letters.map((letter, index) => {
                            const letterEl = letterRefs.current[index]
                            const scale = scales[index]

                            const baseW = baseWidths.current[index] || 0

                            // Push from letters to the left (they swell and push this letter right)
                            let pushRight = 0
                            for (let i = 0; i < index; i++) {
                                const baseWi = baseWidths.current[i] || 0
                                pushRight += (baseWi * (scales[i] - 1)) * 0.4
                            }

                            // Push from letters to the right (they swell and push this letter left)
                            let pushLeft = 0
                            for (let i = index + 1; i < letters.length; i++) {
                                const baseWi = baseWidths.current[i] || 0
                                pushLeft += (baseWi * (scales[i] - 1)) * 0.4
                            }

                            const offset = pushRight - pushLeft

                            return (
                                <span
                                    key={index}
                                    ref={(el) => { letterRefs.current[index] = el }}
                                    className="hero-letter"
                                    style={{
                                        '--proximity': (scale - 1) / 0.5,
                                        '--scale': scale,
                                        transition: 'transform 0.15s ease-out',
                                        transform: `scale(${scale}) translateX(${offset}px)`,
                                    }}
                                >
                                    {letter}
                                </span>
                            )
                        })
                    })()}
                </div>

                <p className="hero-intro">
                    My name is Reuben. I like crafting cool things.
                </p>

                <div className="hero-cta">
                    <a href="/projects" className="hero-btn hero-btn-primary">
                        View My Work
                    </a>
                    <a href="mailto:reubenowenwilliams@outlook.com" className="hero-btn hero-btn-secondary">
                        Contact Me
                    </a>
                </div>
            </div>
        </div>
    )
}
