import React from 'react'
import './VerticalTimeline.css'
import { TimelineEntry } from './TimelineEntry'

const timelineData = [
    {
        date: '29th January 2025',
        title: 'Rehosted site using AWS',
        description:
            'I used ECR, ECS with Fargate, and a load balancer to rehost my site using AWS.',
    },
    {
        date: 'December 2024',
        title: 'Refactored Python backend',
        description:
            'I refactored the Python backend to make it more maintainable and readable. While doing that, I removed the use of file storage on the server, and implemented an in-memory session-caching manager instead, for improved performance and simplified data management.',
    },
    {
        date: '28th November 2022',
        title: 'Site became unavailable',
        description:
            'Heroku removed their free tier, so the free dyno my site was hosted on was shut down and the site became unavailable.',
    },
    {
        date: '7th February 2022',
        title: 'First deployment',
        description: 'I deployed the first fully-functioning version of the site on Heroku.',
    },
    {
        date: '29th December 2021',
        title: 'Project started',
        description: 'Development of this site began.',
    },
]

export const VerticalTimeline = () => {
    return (
        <div className='vertical-timeline'>
            {timelineData.map((entry, index) => {
                const isLast = index === timelineData.length - 1
                return (
                    <TimelineEntry
                        key={index}
                        date={entry.date}
                        title={entry.title}
                        description={entry.description}
                        showLine={!isLast}
                    />
                )
            })}
        </div>
    )
}
