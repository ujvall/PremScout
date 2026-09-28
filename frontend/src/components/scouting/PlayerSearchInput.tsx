import React, { useState, useEffect, useRef } from "react";
import type { PlayerSearchItem } from "../../types/api";
import { searchPlayers } from "../../api";
import { Search, User, X } from "lucide-react";

interface PlayerSearchInputProps {
  selectedPlayer: PlayerSearchItem | null;
  onSelectPlayer: (player: PlayerSearchItem | null) => void;
  placeholder?: string;
}

export const PlayerSearchInput: React.FC<PlayerSearchInputProps> = ({
  selectedPlayer,
  onSelectPlayer,
  placeholder = "Search player name (e.g. Declan Rice, Bukayo Saka, Rodri)...",
}) => {
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<PlayerSearchItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounced live backend search with race condition prevention
  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      setIsOpen(false);
      setHighlightedIndex(-1);
      return;
    }

    let isCurrent = true;
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const players = await searchPlayers(trimmed, 20);
        if (isCurrent) {
          setResults(players);
          setIsOpen(true);
          setHighlightedIndex(-1);
        }
      } catch (err) {
        if (isCurrent) {
          console.error("Failed to query players:", err);
          setResults([]);
        }
      } finally {
        if (isCurrent) {
          setLoading(false);
        }
      }
    }, 250);

    return () => {
      isCurrent = false;
      clearTimeout(timer);
    };
  }, [query]);

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
        setHighlightedIndex(-1);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (player: PlayerSearchItem) => {
    onSelectPlayer(player);
    setQuery("");
    setIsOpen(false);
    setHighlightedIndex(-1);
  };

  const handleClear = () => {
    onSelectPlayer(null);
    setQuery("");
    setHighlightedIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen || results.length === 0) {
      if (e.key === "ArrowDown" && results.length > 0) {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev < results.length - 1 ? prev + 1 : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : results.length - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (highlightedIndex >= 0 && highlightedIndex < results.length) {
        handleSelect(results[highlightedIndex]);
      }
    } else if (e.key === "Escape") {
      setIsOpen(false);
      setHighlightedIndex(-1);
    }
  };

  return (
    <div ref={containerRef} className="autocomplete-container">
      {/* If a player is selected, show compact summary card */}
      {selectedPlayer ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.75rem 1rem",
            backgroundColor: "rgba(16, 185, 129, 0.08)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            borderRadius: "var(--radius-md)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "50%",
                backgroundColor: "var(--bg-elevated)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent-emerald)",
              }}
            >
              <User size={18} />
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                <span style={{ fontWeight: 700, fontSize: "1rem", color: "#ffffff" }}>
                  {selectedPlayer.player_name}
                </span>
                <span className="badge badge-emerald">
                  {selectedPlayer.position || selectedPlayer.normalized_position}
                </span>
                <span
                  style={{
                    fontSize: "0.6875rem",
                    color: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.12)",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    borderRadius: "9999px",
                    padding: "0.1rem 0.45rem",
                    fontWeight: 600,
                  }}
                >
                  Data as of: {selectedPlayer.data_as_of || "2026-09-20"}
                </span>
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0.2rem 0 0" }}>
                Club: <strong style={{ color: "var(--text-secondary)" }}>{selectedPlayer.club}</strong> |
                Season: {selectedPlayer.season}
                {selectedPlayer.age ? ` | Age: ${selectedPlayer.age}` : ""}
                {selectedPlayer.market_value_eur
                  ? ` | Val: €${(selectedPlayer.market_value_eur / 1e6).toFixed(1)}M`
                  : ""}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={handleClear}
            className="btn btn-ghost"
            style={{ padding: "0.35rem 0.5rem", fontSize: "0.75rem", gap: "0.25rem" }}
            title="Choose a different player"
            aria-label="Change selected player"
          >
            <X size={14} />
            <span>Change</span>
          </button>
        </div>
      ) : (
        /* Search Input */
        <div style={{ position: "relative" }}>
          <div
            style={{
              position: "absolute",
              left: "0.875rem",
              top: "50%",
              transform: "translateY(-50%)",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              pointerEvents: "none",
            }}
          >
            <Search size={16} />
          </div>

          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: "2.35rem", paddingRight: loading ? "2.5rem" : "1rem" }}
            placeholder={placeholder}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              if (results.length > 0) setIsOpen(true);
            }}
            role="combobox"
            aria-expanded={isOpen}
            aria-autocomplete="list"
            aria-label="Search football players"
          />

          {loading && (
            <div
              style={{
                position: "absolute",
                right: "0.875rem",
                top: "50%",
                transform: "translateY(-50%)",
              }}
            >
              <div
                className="animate-spin"
                style={{
                  width: "14px",
                  height: "14px",
                  border: "2px solid var(--accent-emerald)",
                  borderTopColor: "transparent",
                  borderRadius: "50%",
                }}
              />
            </div>
          )}

          {/* Autocomplete Dropdown */}
          {isOpen && results.length > 0 && (
            <div className="autocomplete-dropdown animate-fade-in" role="listbox">
              {results.map((item, idx) => {
                const isHighlighted = idx === highlightedIndex;
                return (
                  <div
                    key={`${item.player_name}-${item.season}`}
                    role="option"
                    aria-selected={isHighlighted}
                    className="autocomplete-item"
                    style={{
                      backgroundColor: isHighlighted ? "var(--bg-card-hover)" : undefined,
                    }}
                    onClick={() => handleSelect(item)}
                    onMouseEnter={() => setHighlightedIndex(idx)}
                  >
                    <div>
                      <span style={{ fontWeight: 600, fontSize: "0.875rem", color: "#ffffff" }}>
                        {item.player_name}
                      </span>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                        {item.club} • {item.season}
                        {item.age ? ` • Age ${item.age}` : ""}
                        <span style={{ marginLeft: "0.5rem", color: "#38bdf8", fontSize: "0.6875rem", opacity: 0.85 }}>
                          (Data as of: {item.data_as_of || "2026-09-20"})
                        </span>
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span className="badge badge-blue">
                        {item.position || item.normalized_position}
                      </span>
                      {item.market_value_eur ? (
                        <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                          €{(item.market_value_eur / 1e6).toFixed(1)}M
                        </span>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {isOpen && query.trim().length >= 2 && results.length === 0 && !loading && (
            <div
              className="autocomplete-dropdown"
              style={{ padding: "1rem", textAlign: "center", color: "var(--text-muted)", fontSize: "0.8125rem" }}
            >
              No players found matching "{query}". Try a different spelling or surname.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
