import React from 'react'
import './VerticalTimeline.css'

export const VerticalTimeline = () => {
    return (
        <div className='vertical-timeline'>
            <div className='timeline-line'></div>
            <div className='event'>
                <div className='event-date'>29th Januray 2025</div>
                <div className='event-circle'></div>
                <div className='event-description'>
                    <h3>Rehosted site using AWS</h3>
                    <p>
                        I used ECR, ECS with Fargate, and a load balancer to rehost my
                        site using AWS.
                    </p>
                </div>
            </div>
            <div className='event'>
                <div className='event-date'>December 2024</div>
                <div className='event-circle'></div>
                <div className='event-description'>
                    <h3>Refactored Python backend</h3>
                    <p>
                        I refactored the Python backend to make it more maintainable and
                        readable. While doing that, I removed the use of file storage on
                        the server, and implemented an in-memory session-caching manager
                        instead, for improved performance and simplified data management.
                    </p>
                </div>
            </div>
            <div className='event'>
                <div className='event-date'>28th November 2022</div>
                <div className='event-circle'></div>
                <div className='event-description'>
                    <h3>Site became unavailable</h3>
                    <p>
                        Heroku removed their free tier, so the free dyno my site was
                        hosted on was shut down and the site became unavailable.
                    </p>
                </div>
            </div>
            <div className='event'>
                <div className='event-date'>7th February 2022</div>
                <div className='event-circle'></div>
                <div className='event-description'>
                    <h3>First deployment</h3>
                    <p>I deployed the first fully-functioning version of the site on Heroku.</p>
                </div>
            </div>
            <div className='event'>
                <div className='event-date'>29th December 2021</div>
                <div className='event-circle'></div>
                <div className='event-description'>
                    <h3>Project started</h3>
                    <p>Development of this site began.</p>
                </div>
            </div>
        </div>
    )
}
