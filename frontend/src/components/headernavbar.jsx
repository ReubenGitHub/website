import './headernavbar.css';
import logo_white from '../ReubenHOWLogo_White_Orange.png'
// import headerimg from '../Header_img.jpg';

export function NavBar() {
    let pages = ['', 'machinelearner', 'pathfinder']
    let url = window.location.href.split("/")
    let currentPage = url[url.length - 1]
    let index = pages.indexOf(currentPage)
    let active = []
    active[index] = "active"
    
    return (
        <div class="navbar-container">
            <div class="navbar">
            <div class="logo"> <img class="img" src={logo_white} alt="Logo"></img> </div>
            <div class="nav-links">
                <a href="/" class={active[0]}>Home</a>
                <div class="dropdown">
                    <a href="/projects" class="dropdown-btn">Projects ▾</a>
                    <div class="dropdown-content">
                        <a href="/projects">All Projects</a>
                        <a href="/machinelearner" class={active[1]}>Machine Learner</a>
                        <a href="/pathfinder" class={active[2]}>Path-finder</a>
                    </div>
                </div>
            </div>
                <a href="https://www.linkedin.com/in/reuben-owen-williams-53609a173/" target="_blank" class="right" aria-label="LinkedIn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                </a>
            </div>
        </div>
    );
}

// export function Header() {
//     return (
//         <div class="header">
//             <div class="img-header-container">
//                 <img src={headerimg} alt="Header image" height="70px"></img>
//             </div>
//             {/* <img class="img-logo" src={logo} alt="Logo" width="300px"></img> */}
//         </div>
//     );
// }