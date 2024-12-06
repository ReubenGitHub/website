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
        <div class="navbar">
            <div class="logo"> <img class="img" src={logo_white} alt="Logo" width="110px"></img> </div>
            <a href="/" class={active[0]}>Home</a>
            <a href="/machinelearner" class={active[1]}>Machine Learner</a>
            <a href="/pathfinder" class={active[2]}>Path-finder</a>
            <a href="https://www.linkedin.com/in/reuben-owen-williams-53609a173/" target="_blank" class="right">LinkedIn</a>
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