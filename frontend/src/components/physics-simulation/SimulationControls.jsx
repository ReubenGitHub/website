import './SimulationControls.css';

const SimulationControls = ({ isRunning, onPause, onResume, onReset }) => {
  return (
    <div className="simulation-controls">
      <div className="controls-group">
        <button
          className={`control-btn ${isRunning ? 'paused' : 'active'}`}
          onClick={onPause}
          disabled={!isRunning}
          title="Pause"
        >
          ⏸ Pause
        </button>
        <button
          className={`control-btn ${!isRunning ? 'active' : ''}`}
          onClick={onResume}
          disabled={isRunning}
          title="Resume"
        >
          ▶ Resume
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
        <span className={`status-indicator ${isRunning ? 'running' : 'stopped'}`}>
          {isRunning ? '● Running' : '○ Stopped'}
        </span>
      </div>
    </div>
  );
};

export default SimulationControls;
