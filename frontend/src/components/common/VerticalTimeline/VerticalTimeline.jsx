import React from 'react'
import './VerticalTimeline.css'
import { TimelineEntry } from './TimelineEntry'

const timelineData = [
    {
        date: 'Jul 2026',
        title: 'AI-assisted redesign & new project',
        description:
            'Used a local LLM to assist with redesigning the frontend, creating a 2D physics ball-bouncing simulation project, and improving AWS deployment automation with scripts.',
    },
    {
        date: 'Jan 2025',
        title: 'Rehosted site using AWS',
        description:
            'I used ECR, ECS with Fargate, and a load balancer to rehost my site using AWS.',
    },
    {
        date: 'Dec 2024',
        title: 'Refactored Python backend',
        description:
            'I refactored the Python backend to make it more maintainable and readable. While doing that, I removed the use of file storage on the server, and implemented an in-memory session-caching manager instead, for improved performance and simplified data management.',
    },
    {
        date: 'Nov 2022',
        title: 'Site became unavailable',
        description:
            'Heroku removed their free tier, so the free dyno my site was hosted on was shut down and the site became unavailable.',
    },
    {
        date: 'Feb 2022',
        title: 'First deployment',
        description: 'I deployed the first fully-functioning version of the site on Heroku.',
    },
    {
        date: 'Dec 2021',
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
