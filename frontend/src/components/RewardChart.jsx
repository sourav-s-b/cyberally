import { useRef, useState } from 'react';

function formatReward(value) {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
}

export default function RewardChart({ history }) {
  const chartRef = useRef(null);
  const [hovered, setHovered] = useState(null);

  if (!history?.length) {
    return (
      <div className="grid min-h-40 place-items-center px-6 py-10 text-sm text-dim">
        Launch a local CAGE4 episode to plot its reward.
      </div>
    );
  }

  const width = 720;
  const height = 190;
  const pad = { left: 58, right: 68, top: 16, bottom: 26 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const stepPenalties = history.map((point) => Math.min(0, Number(point.reward) || 0));
  const cumulativePenalties = history.map((point) => Math.min(0, Number(point.cumulative) || 0));
  const maxStepPenalty = Math.max(...stepPenalties.map(Math.abs)) || 1;
  const maxCumulativePenalty = Math.max(...cumulativePenalties.map(Math.abs)) || 1;
  const yStep = (penalty) => pad.top + (Math.abs(penalty) / maxStepPenalty) * plotHeight;
  const yCumulative = (penalty) => pad.top + (Math.abs(penalty) / maxCumulativePenalty) * plotHeight;
  const stepWidth = plotWidth / history.length;
  const barWidth = Math.max(1, stepWidth * 0.76);
  const firstTick = history[0].tick;
  const lastTick = history[history.length - 1].tick;
  const cumulativePath = history
    .map((point, index) => {
      const x = pad.left + (index + 0.5) * stepWidth;
      const y = yCumulative(cumulativePenalties[index]);
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  const lastX = pad.left + (history.length - 0.5) * stepWidth;
  const lastY = yCumulative(cumulativePenalties[cumulativePenalties.length - 1]);
  const handlePointerMove = (event) => {
    const svgBounds = event.currentTarget.getBoundingClientRect();
    const localX = ((event.clientX - svgBounds.left) / svgBounds.width) * width;
    const localY = ((event.clientY - svgBounds.top) / svgBounds.height) * height;
    if (localX < pad.left || localX > width - pad.right
      || localY < pad.top || localY > height - pad.bottom) {
      setHovered(null);
      return;
    }
    const index = Math.min(history.length - 1, Math.floor((localX - pad.left) / stepWidth));
    const wrapperBounds = chartRef.current?.getBoundingClientRect();
    if (!wrapperBounds) return;
    const tooltipWidth = 220;
    setHovered({
      index,
      left: Math.max(tooltipWidth / 2, Math.min(event.clientX - wrapperBounds.left, wrapperBounds.width - tooltipWidth / 2)),
      top: Math.max(6, event.clientY - wrapperBounds.top - 56),
    });
  };

  return (
    <div ref={chartRef} className="relative px-3 pb-3 pt-3">
      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[10px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-brand-red" /> Per-step penalty
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded-full bg-brand-purple" /> Cumulative penalty
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Per-step CAGE4 penalty bars overlaid with cumulative penalty line; both scales start at zero"
        className="block h-auto w-full"
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHovered(null)}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((fraction, index) => {
          const gridY = pad.top + fraction * plotHeight;
          return (
            <g key={index}>
              <line
                x1={pad.left}
                y1={gridY}
                x2={width - pad.right}
                y2={gridY}
                className={fraction === 0 ? 'reward-zero-line' : 'reward-grid-line'}
              />
              <text x={pad.left - 8} y={gridY + 3} textAnchor="end" className="reward-axis-label">
                {formatReward(-maxStepPenalty * fraction)}
              </text>
              <text x={width - pad.right + 8} y={gridY + 3} textAnchor="start" className="reward-axis-label">
                {formatReward(-maxCumulativePenalty * fraction)}
              </text>
            </g>
          );
        })}
        {history.map((point, index) => {
          const penaltyY = yStep(stepPenalties[index]);
          const barHeight = penaltyY - pad.top;
          return (
            <rect
              key={`${point.tick}-${index}`}
              x={pad.left + index * stepWidth + (stepWidth - barWidth) / 2}
              y={pad.top}
              width={barWidth}
              height={barHeight}
              className="reward-negative-bar"
            >
              <title>{`Step ${point.tick}: ${formatReward(stepPenalties[index])} penalty`}</title>
            </rect>
          );
        })}
        <path d={cumulativePath} className="reward-cumulative-line" />
        <circle cx={lastX} cy={lastY} r={3.5} className="reward-cumulative-point" />
        <rect
          x={pad.left}
          y={pad.top}
          width={plotWidth}
          height={plotHeight}
          fill="transparent"
          pointerEvents="all"
        />
        <text x={pad.left} y={height - 7} textAnchor="start" className="reward-axis-label">
          {`step ${firstTick}`}
        </text>
        <text x={width - pad.right} y={height - 7} textAnchor="end" className="reward-axis-label">
          {`step ${lastTick}`}
        </text>
      </svg>
      {hovered ? (
        <div
          className="reward-tooltip"
          style={{ left: `${hovered.left}px`, top: `${hovered.top}px` }}
          role="status"
          aria-live="polite"
        >
          <span className="font-semibold text-ink">Step {history[hovered.index].tick}</span>
          <span className="text-brand-red">Penalty {formatReward(stepPenalties[hovered.index])}</span>
          <span className="text-brand-purple">Cumulative {formatReward(cumulativePenalties[hovered.index])}</span>
        </div>
      ) : null}
    </div>
  );
}
