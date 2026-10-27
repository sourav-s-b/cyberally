import { useMemo, useState } from 'react';
import {
  Briefcase,
  Building2,
  Cloud,
  Globe,
  Network,
  Server,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react';

const KNOWN_EVIDENCE = new Set(['clean', 'suspicious', 'confirmed', 'verify', 'unknown']);

const EVIDENCE_LABEL = {
  clean: 'Clean',
  suspicious: 'Suspicious',
  confirmed: 'Confirmed compromise',
  verify: 'Needs verification',
  unknown: 'Unknown',
};

const EVIDENCE_BADGE = {
  clean: 'border-brand-green/40 bg-brand-green/10 text-brand-green',
  suspicious: 'border-brand-yellow/40 bg-brand-yellow/10 text-brand-yellow',
  confirmed: 'border-brand-red/40 bg-brand-red/10 text-brand-red',
  verify: 'border-brand-orange/40 bg-brand-orange/10 text-brand-orange',
  unknown: 'border-line-strong bg-surface-2 text-muted',
};

const ZONE_KIND = {
  restricted: {
    label: 'Restricted',
    icon: ShieldAlert,
    card: 'border-brand-red/35 bg-brand-red/10',
    iconBox: 'bg-brand-red/15 text-brand-red',
    text: 'text-brand-red',
  },
  operational: {
    label: 'Operational',
    icon: Server,
    card: 'border-brand-green/35 bg-brand-green/10',
    iconBox: 'bg-brand-green/15 text-brand-green',
    text: 'text-brand-green',
  },
  admin: {
    label: 'Admin',
    icon: ShieldCheck,
    card: 'border-brand-purple/35 bg-brand-purple/10',
    iconBox: 'bg-brand-purple/15 text-brand-purple',
    text: 'text-brand-purple',
  },
  public: {
    label: 'Public',
    icon: Globe,
    card: 'border-brand-blue/35 bg-brand-blue/10',
    iconBox: 'bg-brand-blue/15 text-brand-blue',
    text: 'text-brand-blue',
  },
  office: {
    label: 'Office',
    icon: Building2,
    card: 'border-brand-indigo/35 bg-brand-indigo/10',
    iconBox: 'bg-brand-indigo/15 text-brand-indigo',
    text: 'text-brand-indigo',
  },
  contractor: {
    label: 'Contractor',
    icon: Briefcase,
    card: 'border-brand-orange/35 bg-brand-orange/10',
    iconBox: 'bg-brand-orange/15 text-brand-orange',
    text: 'text-brand-orange',
  },
  shared: {
    label: 'Shared',
    icon: Network,
    card: 'border-line-strong bg-surface-2/90',
    iconBox: 'bg-surface-hover text-muted',
    text: 'text-muted',
  },
};

const ZONE_POSITIONS = {
  restricted_zone_a: { angle: -138, radius: 0.58 },
  operational_zone_a: { angle: -88, radius: 0.52 },
  restricted_zone_b: { angle: -38, radius: 0.64 },
  operational_zone_b: { angle: 18, radius: 0.54 },
  public_access_zone: { angle: 62, radius: 0.62 },
  admin_network: { angle: 116, radius: 0.5 },
  office_network: { angle: 168, radius: 0.7 },
};

const FALLBACK_ANGLES = [-138, -88, -38, 18, 62, 116, 168];
const FALLBACK_RADII = [0.58, 0.52, 0.64, 0.54, 0.62, 0.5, 0.7];
const VIEW = { width: 1000, height: 800, cx: 500, cy: 400, rx: 420, ry: 300 };

function hostEvidence(host) {
  const value = String(host.blue_evidence || host.evidence || 'unknown').toLowerCase();
  return KNOWN_EVIDENCE.has(value) ? value : 'unknown';
}

function zoneFor(host) {
  if (host.zone_id) {
    return {
      id: host.zone_id,
      label: host.zone_label || host.zone_id,
      kind: host.zone_kind || 'shared',
    };
  }
  const zoneId = String(host.name || '').split('_subnet')[0];
  const kind = zoneId.includes('restricted') ? 'restricted'
    : zoneId.includes('operational') ? 'operational'
      : zoneId.includes('admin') ? 'admin'
        : zoneId.includes('public_access') ? 'public'
          : zoneId.includes('office') ? 'office'
            : zoneId.includes('contractor') ? 'contractor'
              : 'shared';
  const label = zoneId.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
  return { id: zoneId || 'unknown', label: label || 'Unknown zone', kind };
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function curvePath(from, to, bend) {
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  const offset = bend * 34;
  return `M ${from.x} ${from.y} Q ${midX - (dy / length) * offset} ${midY + (dx / length) * offset} ${to.x} ${to.y}`;
}

function tooltipStyle(x, y) {
  const style = {};
  if (x < 24) {
    style.left = `calc(${x}% + 14px)`;
  } else if (x > 76) {
    style.left = `calc(${x}% - 14px)`;
    style.transform = 'translateX(-100%)';
  } else {
    style.left = `${x}%`;
    style.transform = 'translateX(-50%)';
  }
  if (y < 30) {
    style.top = `calc(${y}% + 16px)`;
  } else {
    style.top = `calc(${y}% - 16px)`;
    style.transform = `${style.transform || ''} translateY(-100%)`.trim();
  }
  return style;
}

export default function Topology({ hosts }) {
  const [hovered, setHovered] = useState(null);

  const zones = useMemo(() => {
    const grouped = new Map();
    for (const host of hosts || []) {
      const zone = zoneFor(host);
      if (!grouped.has(zone.id)) grouped.set(zone.id, { ...zone, hosts: [] });
      grouped.get(zone.id).hosts.push(host);
    }
    return [...grouped.values()].map((zone, zoneIndex) => {
      const layout = ZONE_POSITIONS[zone.id] || {
        angle: FALLBACK_ANGLES[zoneIndex % FALLBACK_ANGLES.length],
        radius: FALLBACK_RADII[zoneIndex % FALLBACK_RADII.length],
      };
      const angle = (layout.angle * Math.PI) / 180;
      const x = VIEW.cx + Math.cos(angle) * VIEW.rx * layout.radius;
      const y = VIEW.cy + Math.sin(angle) * VIEW.ry * layout.radius;
      const baseAngle = hashString(zone.id) * Math.PI * 2;
      const hostPoints = zone.hosts.map((host, hostIndex) => {
        const jitter = (hashString(`${host.name}:${host.agent}`) - 0.5) * 7;
        const hostAngle = baseAngle + hostIndex * Math.PI * (3 - Math.sqrt(5));
        const radiusX = 104 + Math.sqrt(hostIndex) * 2.6 + jitter;
        const radiusY = 60 + Math.sqrt(hostIndex) * 2.2 + jitter * 0.5;
        const hostX = x + Math.cos(hostAngle) * radiusX;
        const hostY = y + Math.sin(hostAngle) * radiusY;
        return {
          host,
          hostIndex,
          x: hostX,
          y: hostY,
          xPct: (hostX / VIEW.width) * 100,
          yPct: (hostY / VIEW.height) * 100,
        };
      });
      return {
        ...zone,
        x,
        y,
        xPct: (x / VIEW.width) * 100,
        yPct: (y / VIEW.height) * 100,
        hostPoints,
      };
    });
  }, [hosts]);

  if (!hosts?.length) {
    return (
      <div className="grid min-h-64 place-items-center px-6 py-10 text-sm text-dim">
        Launch a local CAGE4 episode to load its host list.
      </div>
    );
  }

  const setHostHover = (point, zone) => {
    setHovered({
      key: `${point.host.agent}-${point.host.name}-${point.hostIndex}`,
      host: point.host,
      zoneLabel: zone.label,
      x: point.xPct,
      y: point.yPct,
    });
  };

  const clearHostHover = (key) => {
    setHovered((current) => (current?.key === key ? null : current));
  };

  return (
    <div>
      <div className="relative mx-auto aspect-[5/4] w-full max-w-[1200px]">
        <svg
          viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
          role="img"
          aria-label="CAGE4 zones, hosts and Blue-visible evidence status"
          className="absolute inset-0 h-full w-full"
        >
          <defs>
            <radialGradient id="topology-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(99, 102, 241, 0.16)" />
              <stop offset="55%" stopColor="rgba(56, 189, 248, 0.05)" />
              <stop offset="100%" stopColor="rgba(6, 9, 20, 0)" />
            </radialGradient>
            <pattern id="topology-grid" width="42" height="42" patternUnits="userSpaceOnUse">
              <path d="M 42 0 L 0 0 0 42" fill="none" stroke="rgba(99, 137, 245, 0.07)" strokeWidth="1" />
            </pattern>
          </defs>

          <rect width={VIEW.width} height={VIEW.height} fill="url(#topology-grid)" />
          <circle cx={VIEW.cx} cy={VIEW.cy} r={330} fill="url(#topology-glow)" />

          {zones.map((zone, index) => (
            <path
              key={`backbone-${zone.id}`}
              d={curvePath(
                { x: VIEW.cx, y: VIEW.cy },
                { x: zone.x, y: zone.y },
                (index % 2 === 0 ? 1 : -1) * (0.8 + (index % 3) * 0.35),
              )}
              className="topology-backbone-edge"
            />
          ))}

          {zones.map((zone) => zone.hostPoints.map((point) => (
            <line
              key={`host-edge-${zone.id}-${point.host.agent}-${point.host.name}-${point.hostIndex}`}
              x1={zone.x}
              y1={zone.y}
              x2={point.x}
              y2={point.y}
              className="topology-host-edge"
            />
          )))}

          {zones.map((zone) => {
            const meta = ZONE_KIND[zone.kind] || ZONE_KIND.shared;
            const Icon = meta.icon;
            return (
              <foreignObject
                key={`zone-${zone.id}`}
                x={zone.x - 88}
                y={zone.y - 38}
                width={176}
                height={76}
                style={{ pointerEvents: 'none' }}
              >
                <div
                  xmlns="http://www.w3.org/1999/xhtml"
                  className="grid h-full w-full place-items-center"
                >
                  <div className={`flex h-14 w-[156px] items-center gap-2 rounded-xl border px-2 py-1.5 shadow-xl shadow-black/20 backdrop-blur ${meta.card}`}>
                    <span className={`grid h-7 w-7 shrink-0 place-items-center rounded-lg ${meta.iconBox}`}>
                      <Icon size={14} />
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-[10px] font-semibold leading-tight text-ink">
                        {zone.label}
                      </span>
                      <span className={`mt-0.5 block truncate text-[8px] font-medium uppercase tracking-wide ${meta.text}`}>
                        {meta.label} · {zone.hosts.length} host{zone.hosts.length === 1 ? '' : 's'}
                      </span>
                    </span>
                  </div>
                </div>
              </foreignObject>
            );
          })}

          <foreignObject
            x={VIEW.cx - 70}
            y={VIEW.cy - 70}
            width={140}
            height={140}
            style={{ pointerEvents: 'none' }}
          >
            <div
              xmlns="http://www.w3.org/1999/xhtml"
              className="grid h-full w-full place-items-center"
            >
              <div className="relative grid h-[104px] w-[104px] place-items-center rounded-full border border-brand-indigo/40 bg-surface/95 text-brand-blue shadow-[0_0_70px_rgba(99,102,241,0.35)] backdrop-blur">
                <span className="absolute inset-2 rounded-full border border-brand-blue/15" />
                <Cloud size={40} strokeWidth={1.5} />
              </div>
            </div>
          </foreignObject>

          {zones.map((zone) => zone.hostPoints.map((point) => {
            const evidence = hostEvidence(point.host);
            const key = `${point.host.agent}-${point.host.name}-${point.hostIndex}`;
            const isHovered = hovered?.key === key;
            const isRouter = point.host.name.includes('_router');
            return (
              <g
                key={`host-${key}`}
                transform={`translate(${point.x}, ${point.y})`}
                className="topology-host-group"
                tabIndex={0}
                role="button"
                aria-label={`${point.host.name}, ${EVIDENCE_LABEL[evidence]}${point.host.pending ? ', action pending' : ''}`}
                onMouseEnter={() => setHostHover(point, zone)}
                onMouseLeave={() => clearHostHover(key)}
                onFocus={() => setHostHover(point, zone)}
                onBlur={() => clearHostHover(key)}
                onClick={() => (isHovered ? setHovered(null) : setHostHover(point, zone))}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  if (isHovered) setHovered(null);
                  else setHostHover(point, zone);
                }}
              >
                <circle r={13} className="topology-host-hit" />
                <g
                  className="topology-host-scale"
                  style={{ transform: isHovered ? 'scale(1.65)' : 'scale(1)' }}
                >
                  {isRouter ? (
                    <rect
                      x={-6}
                      y={-6}
                      width={12}
                      height={12}
                      rx={2}
                      className={`topology-host-dot evidence-${evidence}`}
                    />
                  ) : (
                    <circle r={6.5} className={`topology-host-dot evidence-${evidence}`} />
                  )}
                  {point.host.pending ? <circle r={11} className="topology-host-pending" /> : null}
                </g>
              </g>
            );
          }))}
        </svg>

        {hovered ? (
          <div className="pointer-events-none absolute z-30 w-60" style={tooltipStyle(hovered.x, hovered.y)}>
            <div className="rounded-xl border border-line-strong bg-surface/95 p-3 text-left shadow-2xl shadow-black/40 backdrop-blur">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate font-mono text-[11px] font-semibold text-ink">{hovered.host.name}</p>
                  <p className="mt-0.5 truncate text-[10px] text-dim">
                    {hovered.zoneLabel} · {hovered.host.agent}
                  </p>
                </div>
                <span className={`shrink-0 rounded-md border px-1.5 py-0.5 text-[9px] font-semibold ${EVIDENCE_BADGE[hostEvidence(hovered.host)]}`}>
                  {EVIDENCE_LABEL[hostEvidence(hovered.host)]}
                </span>
              </div>
              <div className="mt-2 space-y-1 text-[10px] text-muted">
                <p className="flex justify-between gap-3">
                  <span className="text-dim">Subnet</span>
                  <span className="truncate font-mono">{hovered.host.subnet || 'unknown'}</span>
                </p>
                <p className="flex justify-between gap-3">
                  <span className="text-dim">Status</span>
                  <span className={hovered.host.pending ? 'text-brand-orange' : 'text-muted'}>
                    {hovered.host.pending ? 'Action pending' : 'No action pending'}
                  </span>
                </p>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1 pb-3 text-[10px] text-dim">
        <span className="font-semibold uppercase tracking-wider">Zones</span>
        <span className="inline-flex items-center gap-1.5">
          <ShieldAlert size={12} className="text-brand-red" /> Restricted
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Server size={12} className="text-brand-green" /> Operational
        </span>
        <span className="inline-flex items-center gap-1.5">
          <ShieldCheck size={12} className="text-brand-purple" /> Admin
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border border-white/70 bg-brand-blue" /> Host
        </span>
      </div>
    </div>
  );
}
