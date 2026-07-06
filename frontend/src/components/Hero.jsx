import { useState, useEffect, useRef, useCallback } from 'react'
import './Hero.css'

export function Hero() {
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
    const heroRef = useRef(null)
    const containerRef = useRef(null)
    const animationRef = useRef(null)
    const timeRef = useRef(0)

    const handleMouseMove = useCallback((e) => {
        if (!containerRef.current) return
        const rect = containerRef.current.getBoundingClientRect()
        setMousePos({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
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
            <div ref={heroRef} className="hero-content">
                <div className="hero-greeting">
                    {letters.map((letter, index) => {
                        // Calculate distance from mouse to each letter's center
                        const letterWidth = 100 / letters.length
                        const letterCenterX = (index + 0.5) * letterWidth
                        const letterCenterY = 50
                        const dx = mousePos.x - (letterCenterX / 100 * mousePos.width)
                        const dy = mousePos.y - (letterCenterY / 100 * mousePos.height)
                        const distance = Math.sqrt(dx * dx + dy * dy)
                        const maxDistance = Math.sqrt(
                            (mousePos.width / 2) ** 2 + (mousePos.height / 2) ** 2
                        )
                        const proximity = Math.max(0, 1 - distance / (maxDistance * 0.6))
                        const scale = 1 + proximity * 0.5

                        return (
                            <span
                                key={index}
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

                <p className="hero-subtitle">
                    Developer · Builder · Explorer
                </p>

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
