import { useState, useEffect, useRef } from 'react'
import '../Hero.css'
import { BlockLetters } from './BlockLetters'

export function ProjectHero() {
    const containerRef = useRef(null)
    const mouseRef = useRef({ x: -1000, y: -1000 })
    const rafRef = useRef(null)
    const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 })

    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        const hasPointer = window.matchMedia('(pointer: fine)').matches

        if (!hasPointer) {
            // Auto-animate on touch devices
            const startTime = Date.now()
            let rafId = null
            let animId = null

            const animate = () => {
                const elapsed = (Date.now() - startTime) / 1000
                const cycle = 4
                const t = (elapsed % cycle) / cycle
                const pos = t < 0.5 ? 2 * t : 2 * (1 - t)
                const rect = container.getBoundingClientRect()

                mouseRef.current = {
                    x: rect.left + pos * rect.width,
                    y: rect.top + rect.height / 2,
                }

                if (!rafId) {
                    rafId = requestAnimationFrame(() => {
                        setMousePos({ ...mouseRef.current })
                        rafId = null
                    })
                }

                animId = requestAnimationFrame(animate)
            }

            animId = requestAnimationFrame(animate)
            return () => {
                if (animId) cancelAnimationFrame(animId)
            }
        } else {
            const handleMouseMove = (e) => {
                const rect = container.getBoundingClientRect()
                mouseRef.current = {
                    x: e.clientX,
                    y: e.clientY,
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

    return (
        <div ref={containerRef} className="hero">
            <div className="hero-content">
                <div className="hero-block-wrapper">
                    <BlockLetters
                        text="Projects"
                        mousePos={mousePos}
                        containerRef={containerRef}
                    />
                </div>
                <p className="hero-intro">
                    Things I've done.
                </p>
            </div>
        </div>
    )
}
