import React from "react";

interface SkeletonProps {
  className?: string;
  style?: React.CSSProperties;
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = "",
  style = {},
  width,
  height,
  borderRadius,
}) => {
  return (
    <div
      className={`skeleton-pulse ${className}`}
      style={{
        width: width ?? "100%",
        height: height ?? "1rem",
        borderRadius: borderRadius ?? "var(--radius-sm)",
        ...style,
      }}
      aria-hidden="true"
    />
  );
};

export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({
  rows = 8,
  columns = 10,
}) => {
  return (
    <div className="table-container" style={{ padding: "0.5rem" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.625rem" }}>
        {/* Header row */}
        <div style={{ display: "flex", gap: "1rem", padding: "0.75rem 1rem", borderBottom: "1px solid var(--border-subtle)" }}>
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton key={`h-${i}`} height="14px" width={i === 0 ? "40px" : i === 1 ? "140px" : "80px"} />
          ))}
        </div>
        {/* Rows */}
        {Array.from({ length: rows }).map((_, r) => (
          <div
            key={`r-${r}`}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              padding: "0.75rem 1rem",
              borderBottom: r < rows - 1 ? "1px solid var(--border-subtle)" : "none",
            }}
          >
            {Array.from({ length: columns }).map((_, c) => (
              <Skeleton
                key={`c-${r}-${c}`}
                height="16px"
                width={c === 0 ? "35px" : c === 1 ? `${100 + (r % 3) * 25}px` : `${60 + (c % 3) * 15}px`}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export const PlayerDetailSkeleton: React.FC = () => {
  return (
    <div className="card" style={{ padding: "1.5rem" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.25rem" }}>
        <Skeleton width="48px" height="48px" borderRadius="var(--radius-md)" />
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", flex: 1 }}>
          <Skeleton width="200px" height="22px" />
          <Skeleton width="320px" height="14px" />
        </div>
      </div>
      {/* Badges and summary */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>
        <Skeleton height="70px" borderRadius="var(--radius-md)" />
        <Skeleton height="70px" borderRadius="var(--radius-md)" />
      </div>
      {/* Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "0.625rem" }}>
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} height="42px" borderRadius="var(--radius-sm)" />
        ))}
      </div>
    </div>
  );
};
