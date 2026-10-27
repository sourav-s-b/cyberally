export default function RewardChart({ history }) {
  if (!history?.length) {
    return (
      <div className="grid min-h-40 place-items-center px-6 py-10 text-sm text-dim">
        Launch a local CAGE4 episode to plot its reward.
      </div>
    );
  }

  const width = 720;
  const height = 180;
  const pad = { left: 48, right: 16, top: 14, bottom: 26 };
  const maxTick = Math.max(...history.map((point) => point.tick), 1);
  const values = history.map((point) => point.cumulative);
  const minValue = Math.min(...values, 0);
  let maxValue = Math.max(...values, 0);
  if (minValue === maxValue) maxValue = minValue + 1;

  const x = (tick) => pad.left + (tick / maxTick) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - (value - minValue) / (maxValue - minValue)) * (height - pad.top - pad.bottom);
  const path = history
    .map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.tick).toFixed(1)},${y(point.cumulative).toFixed(1)}`)
    .join(' ');
  const last = history[history.length - 1];

  return (
    <div className="px-3 pb-3 pt-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Cumulative CAGE4 team reward over simulator steps"
        className="block h-auto w-full"
      >
        {[0, 1, 2, 3, 4].map((index) => {
          const value = minValue + ((maxValue - minValue) * index) / 4;
          const gridY = y(value);
          return (
            <g key={index}>
              <line x1={pad.left} y1={gridY} x2={width - pad.right} y2={gridY} className="reward-grid-line" />
              <text x={pad.left - 8} y={gridY + 3} textAnchor="end" className="reward-axis-label">
                {value.toFixed(1)}
              </text>
            </g>
          );
        })}
        <text x={pad.left} y={height - 8} textAnchor="start" className="reward-axis-label">step 0</text>
        <text x={width - pad.right} y={height - 8} textAnchor="end" className="reward-axis-label">
          {`step ${maxTick}`}
        </text>
        <path d={path} className="reward-line" />
        <circle cx={x(last.tick)} cy={y(last.cumulative)} r={4} className="reward-last-point" />
      </svg>
    </div>
  );
}
