# Home Page Redesign Plan

## Phase 1: Hero Section
- [x] 1.1 Create animated "Hello!" hero section with cursor-reactive text swelling effect
- [x] 1.2 Add floating shapes background (moved to global)
- [x] 1.3 Add CTA buttons ("View My Work", "Contact")
- [x] 1.4 Ensure existing Hello component remains below hero + auto-cursor animation for touch devices
- [x] 1.5 Remove "Developer · Builder · Explorer" subtitle text
- [x] 1.6 Reduce hero height to ~1/3 of current (from 100vh to ~33vh)
- [x] 1.7 Fix cursor proximity calculation — letters should reach max scale when cursor is directly ON them, not before
- [x] 1.8 Make hero transparent to show global floating shapes background

## Phase 2: Layout & Structure
- [x] 2.1 Remove deprecated `<center>` tags, replace with CSS Flexbox/Grid
- [x] 2.2 Convert sections to semantic HTML (`<section>`, `<article>`)
- [x] 2.3 Implement card-based glassmorphism layout for content sections
- [x] 2.4 Add CSS custom properties for consistent theming

## Phase 3: Global Background
- [x] 3.1 Move floating shapes from hero to global page background
- [x] 3.2 Replace light geometric background with dark gradient
- [x] 3.3 Make hero transparent to show global background

## Phase 4: Home Page Content Restructuring
- [ ] 4.1 Move intro text ("My name is Reuben...") — consider hero subtitle or dedicated about page
- [ ] 4.2 Create dedicated project/work pages (Machine Learner, Path-finder, DotNet Demo, Speedy PV)
- [ ] 4.3 Replace home page project descriptions with highlight cards linking to detail pages
- [ ] 4.4 Move contact details (email, social links) to dedicated contact/about page or footer
- [ ] 4.5 Reduce home page content section size — keep it concise
- [ ] 4.6 Add "About" link to navbar

## Phase 5: Scroll Animations
- [ ] 5.1 Implement Intersection Observer for fade-in/slide-up on scroll
- [ ] 5.2 Add staggered animations for sequential elements
- [ ] 5.3 Smooth scroll behavior

## Phase 6: Dark/Light Mode
- [ ] 6.1 Set up CSS custom properties for both themes
- [ ] 6.2 Add theme toggle button in navbar
- [ ] 6.3 Persist theme preference in localStorage

## Phase 7: Typography
- [ ] 7.1 Add Google Font (Inter or Plus Jakarta Sans)
- [ ] 7.2 Implement responsive font sizes with `clamp()`
- [ ] 7.3 Improve heading/body contrast with font weights
- [ ] 7.4 Better line-height and spacing

## Phase 8: Timeline Enhancement
- [ ] 8.1 Color-code events by category
- [ ] 8.2 Add icons per event type
- [ ] 8.3 Make cards expandable on click
- [ ] 8.4 Add scroll-triggered staggered animations

## Phase 9: Micro-Interactions & Polish
- [ ] 9.1 Hover effects on cards (lift + shadow)
- [ ] 9.2 Link underline animations
- [ ] 9.3 Gradient accent on headings/links
- [ ] 9.4 Replace float-based navbar with Flexbox
- [ ] 9.5 Consistent spacing with `clamp()`
