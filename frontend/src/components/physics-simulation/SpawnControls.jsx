import './SpawnControls.css';

const SpawnControls = ({ ballCount, setBallCount }) => {
  return (
    <div className="spawn-controls">
      <div className="spawn-control-group">
        <label className="spawn-label">
          Ball Count: <span className="spawn-value">{ballCount}</span>
        </label>
        <input
          type="range"
          min="10"
          max="10000"
          step="10"
          value={ballCount}
          onChange={(e) => setBallCount(Number(e.target.value))}
          className="spawn-slider"
        />
        <div className="spawn-slider-labels">
          <span>10</span>
          <span>5,000</span>
          <span>10,000</span>
        </div>
      </div>
    </div>
  );
};

export default SpawnControls;
