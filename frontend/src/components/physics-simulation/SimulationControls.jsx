import './SimulationControls.css';

const SimulationControls = ({ isRunning, isPaused, onPlay, onPause, onReset }) => {
  return (
    <div className="simulation-controls">
      <div className="controls-group">
        <button
          className={`control-btn ${!isRunning ? 'active' : ''}`}
          onClick={onPlay}
          disabled={isRunning}
          title={isPaused ? 'Resume' : 'Play'}
        >
          {isPaused ? '▶ Resume' : '▶ Play'}
        </button>
        <button
          className={`control-btn ${isRunning ? 'paused' : 'active'}`}
          onClick={onPause}
          disabled={!isRunning}
          title="Pause"
        >
          ⏸ Pause
        </button>
        <button
          className="control-btn reset"
          onClick={onReset}
          title="Reset"
        >
          ↻ Reset
        </button>
      </div>
      <div className="controls-info">
        <span className={`status-indicator ${isRunning ? 'running' : isPaused ? 'paused' : 'stopped'}`}>
          {isRunning ? '● Running' : isPaused ? '◌ Paused' : '○ Stopped'}
        </span>
      </div>
    </div>
  );
};

export default SimulationControls;
