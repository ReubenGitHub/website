import './SimulationControls.css';

const SimulationControls = ({ isRunning, isPaused, onPlay, onPause, onReset }) => {
  const isPlaying = isRunning && !isPaused;
  
  return (
    <div className="simulation-controls">
      <button
        className={`control-btn toggle-btn ${isPlaying ? 'playing' : ''}`}
        onClick={isPlaying ? onPause : onPlay}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="2" width="3.5" height="12" rx="1"/>
            <rect x="9.5" y="2" width="3.5" height="12" rx="1"/>
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 2.5v11l9-5.5z"/>
          </svg>
        )}
      </button>
      <button
        className="control-btn reset-btn"
        onClick={onReset}
        title="Reset"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <path d="M2 8a6 6 0 0 1 10.47-4M14 8a6 6 0 0 1-10.47 4"/>
          <path d="M12 1v3h-3M4 15v-3h3"/>
        </svg>
      </button>
    </div>
  );
};

export default SimulationControls;
