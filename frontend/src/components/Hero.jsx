import { useState, useEffect, useRef, useCallback } from 'react'
import './Hero.css'

export function Hero() {
    const [mousePos, setMousePos] = useState({ x: 0, y: 0, width: 0, height: 0 })
    const letterRefs = useRef({})
    const containerRef = useRef(null)
    const animationRef = useRef(null)
    const timeRef = useRef(0)

    const handleMouseMove = useCallback((e) => {
        if (!containerRef.current) return
        const rect = containerRef.current.getBoundingClientRect()
        setMousePos({
            x: e.clientX,
            y: e.clientY,
            width: rect.width,
            height: rect.height,
        })
    }, [])

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        container.addEventListener('mousemove', handleMouseMove)
        return () => container.removeEventListener('mousemove', handleMouseMove)
    }, [handleMouseMove])

    // Calculate letter bounding boxes on mount and resize
    const getLetterBounds = useCallback(() => {
        const bounds = {}
        Object.keys(letterRefs.current).forEach(key => {
            const el = letterRefs.current[key]
            if (el) {
                const rect = el.getBoundingClientRect()
                bounds[key] = {
                    centerX: rect.left + rect.width / 2,
                    centerY: rect.top + rect.height / 2,
                    width: rect.width,
                    height: rect.height,
                    radius: Math.max(rect.width, rect.height) * 0.5,
                }
            }
        })
        return bounds
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
            {/* Floating background shapes */}
            <div className="hero-bg-shapes">
                <div className="hero-shape hero-shape-1" style={{ animationDelay: '0s' }} />
                <div className="hero-shape hero-shape-2" style={{ animationDelay: '1s' }} />
                <div className="hero-shape hero-shape-3" style={{ animationDelay: '2s' }} />
                <div className="hero-shape hero-shape-4" style={{ animationDelay: '0.5s' }} />
                <div className="hero-shape hero-shape-5" style={{ animationDelay: '1.5s' }} />
            </div>

            {/* Gradient overlay */}
            <div className="hero-gradient" />

            {/* Content */}
            <div className="hero-content">
                <div className="hero-greeting">
                    {letters.map((letter, index) => {
                        const letterEl = letterRefs.current[index]
                        let proximity = 0
                        let scale = 1

                        if (letterEl) {
                            const rect = letterEl.getBoundingClientRect()
                            const letterCenterX = rect.left + rect.width / 2
                            const letterCenterY = rect.top + rect.height / 2
                            const dx = mousePos.x - letterCenterX
                            const dy = mousePos.y - letterCenterY
                            const distance = Math.sqrt(dx * dx + dy * dy)
                            const proximityRadius = Math.max(rect.width, rect.height) * 1.2
                            proximity = Math.max(0, 1 - distance / proximityRadius)
                            scale = 1 + proximity * 0.5
                        }

                        return (
                            <span
                                key={index}
                                ref={(el) => { letterRefs.current[index] = el }}
                                className="hero-letter"
                                style={{
                                    '--proximity': proximity,
                                    '--scale': scale,
                                    transition: 'transform 0.15s ease-out',
                                }}
                            >
                                {letter}
                            </span>
                        )
                    })}
                </div>

                <div className="hero-cta">
                    <a href="/machinelearner" className="hero-btn hero-btn-primary">
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
