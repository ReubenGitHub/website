# Home Page Redesign Plan

## Phase 1: Hero Section
- [x] 1.1 Create animated "Hello!" hero section with cursor-reactive text swelling effect
- [x] 1.2 Add subtle gradient background or floating shapes animation (skipped for now)
- [x] 1.3 Add CTA buttons ("View My Work", "Contact")
- [x] 1.4 Ensure existing Hello component remains below hero + auto-cursor animation for touch devices
- [x] 1.5 Remove "Developer · Builder · Explorer" subtitle text
- [x] 1.6 Reduce hero height to ~1/3 of current (from 100vh to ~33vh)
- [x] 1.7 Fix cursor proximity calculation — letters should reach max scale when cursor is directly ON them, not before

## Phase 2: Layout & Structure
- [x] 2.1 Remove deprecated `<center>` tags, replace with CSS Flexbox/Grid
- [x] 2.2 Convert sections to semantic HTML (`<section>`, `<article>`)
- [ ] 2.3 Implement card-based glassmorphism layout for content sections
- [ ] 2.4 Add CSS custom properties for consistent theming

## Phase 3: Scroll Animations
- [ ] 3.1 Implement Intersection Observer for fade-in/slide-up on scroll
- [ ] 3.2 Add staggered animations for sequential elements
- [ ] 3.3 Smooth scroll behavior

## Phase 4: Dark/Light Mode
- [ ] 4.1 Set up CSS custom properties for both themes
- [ ] 4.2 Add theme toggle button in navbar
- [ ] 4.3 Persist theme preference in localStorage

## Phase 5: Typography
- [ ] 5.1 Add Google Font (Inter or Plus Jakarta Sans)
- [ ] 5.2 Implement responsive font sizes with `clamp()`
- [ ] 5.3 Improve heading/body contrast with font weights
- [ ] 5.4 Better line-height and spacing

## Phase 6: Timeline Enhancement
- [ ] 6.1 Color-code events by category
- [ ] 6.2 Add icons per event type
- [ ] 6.3 Make cards expandable on click
- [ ] 6.4 Add scroll-triggered staggered animations

## Phase 7: Micro-Interactions & Polish
- [ ] 7.1 Hover effects on cards (lift + shadow)
- [ ] 7.2 Link underline animations
- [ ] 7.3 Gradient accent on headings/links
- [ ] 7.4 Replace float-based navbar with Flexbox
- [ ] 7.5 Consistent spacing with `clamp()`
