import { useState, useEffect } from 'react';
import './SpawnControls.css';

const SpawnControls = ({ ballCount, setBallCount, restitution, onRestitutionChange, controlsDisabled }) => {
  const [displayRestitution, setDisplayRestitution] = useState(restitution);

  useEffect(() => {
    setDisplayRestitution(restitution);
  }, [restitution]);

  const handleRestitutionRelease = (e) => {
    const newValue = Number(e.target.value);
    setDisplayRestitution(newValue);
    onRestitutionChange(newValue);
  };

  return (
    <div className="spawn-controls">
      <div className={`spawn-control-group ${controlsDisabled ? 'disabled' : ''}`}>
        <label className="spawn-label">
          Ball Count: <span className="spawn-value">{ballCount}</span>
        </label>
        <input
          type="range"
          min="10"
          max="5000"
          step="10"
          value={ballCount}
          onChange={(e) => setBallCount(Number(e.target.value))}
          className="spawn-slider"
          disabled={controlsDisabled}
        />
        <div className="spawn-slider-labels">
          <span>10</span>
          <span>2,500</span>
          <span>5,000</span>
        </div>
      </div>
      
      <div className="spawn-control-group">
        <label className="spawn-label">
          Restitution (Bounciness): <span className="spawn-value">{displayRestitution.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={displayRestitution}
          onChange={(e) => setDisplayRestitution(Number(e.target.value))}
          onPointerUp={handleRestitutionRelease}
          onTouchEnd={handleRestitutionRelease}
          className="spawn-slider restitution-slider"
        />
        <div className="spawn-slider-labels">
          <span>0 (None)</span>
          <span>0.5</span>
          <span>1 (Perfect)</span>
        </div>
      </div>
    </div>
  );
};

export default SpawnControls;
