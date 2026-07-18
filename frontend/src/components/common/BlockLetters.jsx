import { useRef, useEffect } from 'react';
import '../Hero.css';

// 7-row block font definitions
// 1 = filled block, 0 = empty
// Trailing empty columns (all 0s across all rows) are removed
const BLOCK_FONT = {
    P: [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ],
    R: [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
    ],
    O: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    J: [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    E: [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    C: [
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
    ],
    T: [
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    S: [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
};

export function BlockLetters({ text, mousePos, containerRef }) {
    const blockSize = 14;
    const blockGap = 4;
    const letterSpacing = 10;
    const blockRefs = useRef({});
    const rafRef = useRef(null);
    const mouseRef = useRef({ x: -1000, y: -1000 });
    const textRef = useRef(text);
    const needsRecenterRef = useRef(true);

    // Track block positions for hover effect
    useEffect(() => {
        textRef.current = text;
        // Reset block refs when text changes
        blockRefs.current = {};
    }, [text]);

    // RAF loop that updates transforms directly on DOM (bypasses React re-renders)
    useEffect(() => {
        const proximityRadius = 250;
        const maxPush = 6;
        const returnSpeed = 0.08;
        const animate = () => {
            const mouse = mouseRef.current;
            const text = textRef.current;
            const normalizedText = text.toUpperCase();

            // Update centers only when necessary (scroll, resize, visibility change)
            if (needsRecenterRef.current) {
                Object.keys(blockRefs.current).forEach(key => {
                    const val = blockRefs.current[key];
                    if (val && val.nodeType === 1) {
                        const rect = val.getBoundingClientRect();
                        blockRefs.current[key] = {
                            el: val,
                            cx: rect.left + rect.width / 2,
                            cy: rect.top + rect.height / 2,
                            px: 0,
                            py: 0,
                        };
                    }
                });
                needsRecenterRef.current = false;
            }

            if (mouse.x !== -1000) {
                // Apply styles directly to DOM
                Object.keys(blockRefs.current).forEach(key => {
                    const block = blockRefs.current[key];
                    const parts = key.split('-').map(Number);
                    const [charIndex, rowIdx, colIdx] = parts;
                    const glyph = BLOCK_FONT[normalizedText[charIndex]];

                    if (!glyph || !glyph[rowIdx] || glyph[rowIdx][colIdx] !== 1) return;

                    const dx = mouse.x - block.cx;
                    const dy = mouse.y - block.cy;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const proximity = Math.max(0, 1 - distance / proximityRadius);
                    const glow = proximity;

                    const dist = Math.max(distance, 1);
                    const pushX = -(dx / dist) * proximity * maxPush;
                    const pushY = -(dy / dist) * proximity * maxPush;

                    const brightness = 0.7 + glow * 0.3;
                    const el = block.el;
                    el.style.backgroundColor = `rgba(255, 255, 255, ${brightness})`;
                    el.style.boxShadow = glow > 0
                        ? `0 0 ${glow * 16}px rgba(254, 66, 3, ${glow * 0.7})`
                        : 'none';
                    el.style.transform = `translate(${pushX}px, ${pushY}px)`;
                    block.px = pushX;
                    block.py = pushY;
                });
            } else {
                // Smoothly return letter blocks to origin
                Object.keys(blockRefs.current).forEach(key => {
                    const block = blockRefs.current[key];
                    const parts = key.split('-').map(Number);
                    const [charIndex, rowIdx, colIdx] = parts;
                    const glyph = BLOCK_FONT[normalizedText[charIndex]];

                    if (!glyph || !glyph[rowIdx] || glyph[rowIdx][colIdx] !== 1) return;

                    if (block && block.el) {
                        block.px += (0 - block.px) * returnSpeed;
                        block.py += (0 - block.py) * returnSpeed;

                        const absPx = Math.abs(block.px);
                        const absPy = Math.abs(block.py);

                        if (absPx < 0.05 && absPy < 0.05) {
                            block.px = 0;
                            block.py = 0;
                        }

                        block.el.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
                        block.el.style.boxShadow = 'none';
                        block.el.style.transform = `translate(${block.px.toFixed(2)}px, ${block.py.toFixed(2)}px)`;
                    }
                });
            }

            rafRef.current = requestAnimationFrame(animate);
        };

        rafRef.current = requestAnimationFrame(animate);
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    // Mark centers as stale on scroll/resize/visibility change
    useEffect(() => {
        const markDirty = () => { needsRecenterRef.current = true };
        window.addEventListener('scroll', markDirty, { passive: true });
        window.addEventListener('resize', markDirty, { passive: true });
        document.addEventListener('visibilitychange', markDirty, { passive: true });
        return () => {
            window.removeEventListener('scroll', markDirty);
            window.removeEventListener('resize', markDirty);
            document.removeEventListener('visibilitychange', markDirty);
        };
    }, []);

    // Update mouse position ref
    useEffect(() => {
        mouseRef.current = mousePos;
    }, [mousePos]);

    const normalizedText = text.toUpperCase();

    const getBlockStyle = (charIndex, rowIdx, colIdx) => {
        const glyph = BLOCK_FONT[normalizedText[charIndex]];
        if (!glyph || !glyph[rowIdx]) return { backgroundColor: 'transparent' };

        const isFilled = glyph[rowIdx][colIdx] === 1;
        if (!isFilled) return { backgroundColor: 'transparent' };

        // Default style (RAF loop will update these on mouse move)
        return {
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
            borderRadius: 2,
            boxShadow: 'none',
            transform: 'translate(0px, 0px)',
        };
    };

    return (
        <div className="hero-greeting hero-block-text">
            {normalizedText.split('').map((char, charIndex) => {
                const glyph = BLOCK_FONT[char];
                if (!glyph) return null;

                const glyphWidth = glyph[0].length;
                return (
                    <div
                        key={charIndex}
                        className="hero-block-letter"
                        style={{
                            display: 'grid',
                            gridTemplateColumns: `repeat(${glyphWidth}, ${blockSize}px)`,
                            gridTemplateRows: `repeat(7, ${blockSize}px)`,
                            gap: `${blockGap}px`,
                            marginRight: charIndex < normalizedText.length - 1 ? letterSpacing : 0,
                        }}
                    >
                        {glyph.map((row, rowIdx) =>
                            row.map((cell, colIdx) => {
                                const blockKey = `${charIndex}-${rowIdx}-${colIdx}`;
                                return (
                                    <div
                                        key={blockKey}
                                        ref={(el) => {
                                            if (el) {
                                                const rect = el.getBoundingClientRect();
                                                blockRefs.current[blockKey] = {
                                                    el,
                                                    cx: rect.left + rect.width / 2,
                                                    cy: rect.top + rect.height / 2,
                                                };
                                            }
                                        }}
                                        className="hero-block"
                                        style={getBlockStyle(charIndex, rowIdx, colIdx)}
                                    />
                                );
                            })
                        )}
                    </div>
                );
            })}
        </div>
    );
}
