import React from 'react'
import './VerticalTimeline.css'

export const TimelineEntry = ({ date, title, description, showLine = true }) => {
    return (
        <div className='event'>
            <div className='event-date'>{date}</div>
            {showLine && <div className='event-line'></div>}
            <div className='event-circle'></div>
            <div className='event-description'>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
    )
}
