import React, { useState, useEffect, useRef } from 'react'
import './hello.css'

const MAX_SCALE = 1.8
const MIN_SCALE = 0.7
const MIN_SCALE_CHANGE = (MAX_SCALE - MIN_SCALE) * 0.4

export const Hello = () => {
    const [letters, setLetters] = useState(initializeLetters('Hello!'))
    const letterRefs = useRef({})
    let index = -1

    useEffect(() => {
        // Set all width/height pixel values so size transitions are smooth
        for (let i = 0; i < letters.length; i++) {
            const letterElement = letterRefs.current[i]
            letters[i].width = `${letterElement.offsetWidth}px`
            letters[i].height = `${letterElement.offsetHeight}px`
        }
        setLetters([...letters])

        // Set a random scale/size on a random letter
        const interval = setInterval(() => {
            index = (index + Math.floor(Math.random() * (letters.length - 1) + 1)) % letters.length
            const letterElement = letterRefs.current[index]
            if (Math.random() < 0.5) {
                const newScale = calcCoolNewScale(letters[index].scaleX)
                letters[index].width = `${letterElement.offsetWidth * newScale}px`
                letters[index].scaleX = newScale
            } else {
                const newScale = calcCoolNewScale(letters[index].scaleY)
                letters[index].height = `${letterElement.offsetHeight * newScale}px`
                letters[index].scaleY = newScale
            }
            setLetters([...letters])
        }, 1000)

        return () => clearInterval(interval)
    }, [])

    return (
        <div className="hello">
            {letters.map((letter, index) => (
                <div
                    key={ index }
                    style={{
                        width: letter.width,
                        height: letter.height,
                    }}
                >
                    <span
                        key={ index }
                        style={{
                            transform: `scale(${letter.scaleX}, ${letter.scaleY})`,
                        }}
                        ref={(el) => (letterRefs.current[index] = el)}
                    >
                        {letter.char}
                    </span>
                </div>
            ))}
        </div>
    )
}

function initializeLetters(word) {
    return word.split('').map(char => ({
        char,
        scaleX: 1,
        scaleY: 1,
    }))
}

function calcCoolNewScale(scale) {
    const possibleLowerRange = (scale - MIN_SCALE_CHANGE) - MIN_SCALE
    const possibleHigherRange = MAX_SCALE - (scale + MIN_SCALE_CHANGE)
    const canGoLower = possibleLowerRange >= 0
    const canGoHigher = possibleHigherRange >= 0

    // Determine whether to increase or decrease scale
    let increaseScale = true
    if (canGoLower && canGoHigher) {
        if (Math.random() < 0.5) increaseScale = false
    } else if (canGoLower) {
        increaseScale = false
    }

    const newScale = (increaseScale)
        ? MAX_SCALE - Math.random() * possibleHigherRange
        : MIN_SCALE + Math.random() * possibleLowerRange

    return newScale
}
