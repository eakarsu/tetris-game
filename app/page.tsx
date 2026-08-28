"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ *
 * CHROMA STACK — a falling-block game built with React + browser APIs.
 * Pure logic lives in module scope; the component wires it to state,
 * refs and effects so input never reads stale values.
 * ------------------------------------------------------------------ */

/* ----------------------------- Types ------------------------------ */

type Cell = number | null; // 0..6 -> piece color index, null -> empty
type Board = Cell[][]; // rows x cols, row 0 is top
type PieceType = 0 | 1 | 2 | 3 | 4 | 5 | 6;
type Rotation = 0 | 1 | 2 | 3;
type Phase = "ready" | "playing" | "paused" | "over";
type Dir = "cw" | "ccw";

interface ActivePiece {
  type: PieceType;
  rot: Rotation;
  x: number;
  y: number;
}

interface GameState {
  board: Board;
  active: ActivePiece | null;
  queue: PieceType[];
  score: number;
  lines: number;
  level: number;
  phase: Phase;
  lastClear: number;
  lockTimer: number;
  dropAccum: number;
  spawnFlash: boolean;
  resetCount: number;
}

/* --------------------------- Constants ---------------------------- */

const COLS = 10;
const ROWS = 20;
const LOCK_DELAY = 500; // ms before a grounded piece locks
const MAX_RESETS = 15; // soft-drop / move resets of the lock timer
const BEST_KEY = "chroma-stack-best-v1";

// Per-piece base offsets (relative to a 4x4 box) for each rotation.
const SHAPES: Record<PieceType, number[][][]> = {
  0: [[0, 1], [1, 1], [2, 1], [3, 1]], // I
  1: [[1, 1], [2, 1], [1, 2], [2, 2]], // O
  2: [[1, 1], [0, 2], [1, 2], [2, 2]], // T
  3: [[1, 1], [1, 2], [1, 3], [2, 3]], // L
  4: [[1, 1], [1, 2], [1, 3], [0, 3]], // J
  5: [[1, 1], [2, 1], [0, 2], [1, 2]], // S
  6: [[0, 1], [1, 1], [1, 2], [2, 2]], // Z
};

// Wall-kick tables (offsets are [dx, dy] with +y downward).
const KICKS_JLSTZ: Record<string, number[][]> = {
  "0>1": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "1>0": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "1>2": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "2>1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "2>3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "3>2": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "3>0": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "0>3": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
};

const KICKS_I: Record<string, number[][]> = {
  "0>1": [[0, 0], [-2, 0], [1, 0], [-2, 1], [1, -2]],
  "1>0": [[0, 0], [2, 0], [-1, 0], [2, -1], [-1, 2]],
  "1>2": [[0, 0], [-1, 0], [2, 0], [-1, -2], [2, 1]],
  "2>1": [[0, 0], [1, 0], [-2, 0], [1, 2], [-2, -1]],
  "2>3": [[0, 0], [2, 0], [-1, 0], [2, -1], [-1, 2]],
  "3>2": [[0, 0], [-2, 0], [1, 0], [-2, 1], [1, -2]],
  "3>0": [[0, 0], [1, 0], [-2, 0], [1, 2], [-2, -1]],
  "0>3": [[0, 0], [-1, 0], [2, 0], [-1, -2], [2, 1]],
};

const CLEAR_SCORE = [0, 100, 300, 500, 800];

/* ------------------------- Pure helpers --------------------------- */

function rotateOffsets(type: PieceType, rot: Rotation): number[][] {
  const base = SHAPES[type];
  if (rot === 0) return base.map(([r, c]) => [r, c]);
  let pts = base;
  for (let i = 0; i < rot; i++) {
    pts = pts.map(([r, c]) => [c, 3 - r]);
  }
  return pts;
}

function createBoard(): Board {
  return Array.from({ length: ROWS }, () => Array<Cell>(COLS).fill(null));
}

function cloneBoard(b: Board): Board {
  return b.map((row) => row.slice());
}

function shuffle<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function makeBag(): PieceType[] {
  return shuffle([0, 1, 2, 3, 4, 5, 6]);
}

function refillQueue(q: PieceType[]): PieceType[] {
  let out = q.slice();
  while (out.length < 7) out = out.concat(makeBag());
  return out;
}

function collides(board: Board, type: PieceType, rot: Rotation, px: number, py: number): boolean {
  for (const [r, c] of rotateOffsets(type, rot)) {
    const x = px + c;
    const y = py + r;
    if (x < 0 || x >= COLS || y >= ROWS) return true;
    if (y >= 0 && board[y][x] !== null) return true;
  }
  return false;
}

function ghostY(board: Board, p: ActivePiece): number {
  let gy = p.y;
  while (!collides(board, p.type, p.rot, p.x, gy + 1)) gy++;
  return gy;
}

function hardDropDistance(board: Board, p: ActivePiece): number {
  return ghostY(board, p) - p.y;
}

function clearLines(board: Board): { board: Board; cleared: number } {
  const kept = board.filter((row) => row.some((cell) => cell === null));
  const cleared = ROWS - kept.length;
  const empties = Array.from({ length: cleared }, () => Array<Cell>(COLS).fill(null));
  return { board: [...empties, ...kept], cleared };
}

function computeLevel(lines: number): number {
  return Math.floor(lines / 10) + 1;
}

function gravityInterval(level: number): number {
  return Math.max(60, 800 - (level - 1) * 70);
}

function spawnPiece(type: PieceType): ActivePiece {
  return { type, rot: 0, x: 3, y: 0 };
}

function tryRotate(state: GameState, dir: Dir): GameState {
  const p = state.active;
  if (!p) return state;
  const table = p.type === 0 ? KICKS_I : KICKS_JLSTZ;
  const key = `${p.rot}>${dir === "cw" ? (p.rot + 1) % 4 : (p.rot + 3) % 4}` as string;
  const kicks = table[key] ?? [[0, 0]];
  for (const [dx, dy] of kicks) {
    const nx = p.x + dx;
    const ny = p.y + dy;
    if (!collides(state.board, p.type, p.rot, nx, ny)) {
      return {
        ...state,
        active: { ...p, x: nx, y: ny },
        lockTimer: 0,
        resetCount: 0,
      };
    }
  }
  return state;
}

function tryMove(state: GameState, dx: number): GameState {
  const p = state.active;
  if (!p) return state;
  const nx = p.x + dx;
  if (!collides(state.board, p.type, p.rot, nx, p.y)) {
    return { ...state, active: { ...p, x: nx }, lockTimer: 0, resetCount: 0 };
  }
  return state;
}

function lockAndSpawn(state: GameState): GameState {
  const p = state.active;
  if (!p) return state;
  const board = cloneBoard(state.board);
  for (const [r, c] of rotateOffsets(p.type, p.rot)) {
    const x = p.x + c;
    const y = p.y + r;
    if (y >= 0 && y < ROWS && x >= 0 && x < COLS) board[y][x] = p.type;
  }
  const { board: cleared, cleared: n } = clearLines(board);
  const lines = state.lines + n;
  const level = computeLevel(lines);
  const gained = CLEAR_SCORE[n] * state.level;
  const score = state.score + gained;
  const queue = refillQueue(state.queue.slice(1));
  const next = spawnPiece(queue[0]);
  const over = collides(cleared, next.type, next.rot, next.x, next.y);
  return {
    ...state,
    board: cleared,
    active: over ? null : next,
    queue,
    score,
    lines,
    level,
    phase: over ? "over" : "playing",
    lastClear: n,
    lockTimer: 0,
    dropAccum: 0,
    resetCount: 0,
    spawnFlash: true,
  };
}

function stepDown(state: GameState): GameState {
  const p = state.active;
  if (!p) return state;
  if (!collides(state.board, p.type, p.rot, p.x, p.y + 1)) {
    return { ...state, active: { ...p, y: p.y + 1 } };
  }
  return lockAndSpawn(state);
}

function doHardDrop(state: GameState): GameState {
  const p = state.active;
  if (!p) return state;
  const dist = hardDropDistance(state.board, p);
  const moved: GameState = { ...state, active: { ...p, y: p.y + dist } };
  const scored: GameState = { ...moved, score: moved.score + dist * 2 };
  return lockAndSpawn(scored);
}

function tick(state: GameState, dt: number): GameState {
  if (state.phase !== "playing" || !state.active) return state;
  const interval = gravityInterval(state.level);
  let dropAccum = state.dropAccum + dt;
  let s = state;
  let guard = 0;
  while (dropAccum >= interval && guard < 24) {
    dropAccum -= interval;
    guard++;
    s = stepDown(s);
    if (s.phase !== "playing") break;
  }
  return { ...s, dropAccum };
}

function updateLock(state: GameState, dt: number): GameState {
  const p = state.active;
  if (!p || state.phase !== "playing") return state;
  const grounded = collides(state.board, p.type, p.rot, p.x, p.y + 1);
  if (!grounded) return { ...state, lockTimer: 0 };
  const lockTimer = state.lockTimer + dt;
  if (lockTimer >= LOCK_DELAY) return lockAndSpawn(state);
  return { ...state, lockTimer };
}

function initialState(): GameState {
  const queue = refillQueue([]);
  const first = queue[0];
  const rest = refillQueue(queue.slice(1));
  return {
    board: createBoard(),
    active: spawnPiece(first),
    queue: rest,
    score: 0,
    lines: 0,
    level: 1,
    phase: "ready",
    lastClear: 0,
    lockTimer: 0,
    dropAccum: 0,
    resetCount: 0,
    spawnFlash: false,
  };
}

function freshPlaying(): GameState {
  return { ...initialState(), phase: "playing" };
}

/* --------------------------- Component ---------------------------- */

export default function Page() {
  const [state, setState] = useState<GameState>(() => initialState());
  const [best, setBest] = useState<number>(() => {
    if (typeof window === "undefined") return 0;
    try {
      const raw = window.localStorage.getItem(BEST_KEY);
      if (raw != null) {
        const v = parseInt(raw, 10);
        if (!Number.isNaN(v)) return v;
      }
    } catch {
      /* storage unavailable */
    }
    return 0;
  });

  // Refs mirror the latest state so callbacks/effects never read stale data.
  const stateRef = useRef(state);
  const bestRef = useRef(best);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);
  const holdDirRef = useRef<"left" | "right" | null>(null);
  const holdTimerRef = useRef<number | null>(null);
  const holdRepeatRef = useRef<number | null>(null);
  const touchActiveRef = useRef(false);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);
  useEffect(() => {
    bestRef.current = best;
  }, [best]);

  /* ---- Persist best whenever it grows ---- */
  useEffect(() => {
    if (state.score > bestRef.current) {
      bestRef.current = state.score;
      setBest(state.score);
      try {
        window.localStorage.setItem(BEST_KEY, String(state.score));
      } catch {
        /* ignore */
      }
    }
  }, [state.score]);

  /* ---- Derive aria live text without effects ---- */
  const announce = useMemo(() => {
    if (state.phase === "over") return "Game over";
    if (state.phase === "paused") return "Paused";
    if (state.lastClear > 0) {
      const names = ["", "Single", "Double", "Triple", "Tetris!"];
      return `${names[state.lastClear]} cleared`;
    }
    if (state.spawnFlash) return "New piece";
    if (state.phase === "playing") return "Playing";
    return "";
  }, [state.phase, state.lastClear, state.spawnFlash]);

  /* ---- Game loop via requestAnimationFrame ---- */
  useEffect(() => {
    const loop = (ts: number) => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      let dt = ts - lastTsRef.current;
      lastTsRef.current = ts;
      if (dt > 100) dt = 100; // clamp after tab was hidden

      const cur = stateRef.current;
      if (cur.phase === "playing") {
        let next = tick(cur, dt);
        next = updateLock(next, dt);
        if (next !== cur) setState(next);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTsRef.current = null;
    };
  }, []);

  /* ---- Actions (stable, ref-based) ---- */
  const start = useCallback(() => {
    setState(freshPlaying());
  }, []);

  const restart = useCallback(() => {
    setState(freshPlaying());
  }, []);

  const togglePause = useCallback(() => {
    setState((s) => {
      if (s.phase === "playing") return { ...s, phase: "paused" };
      if (s.phase === "paused") return { ...s, phase: "playing" };
      return s;
    });
  }, []);

  const move = useCallback((dx: number) => {
    setState((s) => (s.phase === "playing" ? tryMove(s, dx) : s));
  }, []);

  const rotate = useCallback((dir: Dir) => {
    setState((s) => (s.phase === "playing" ? tryRotate(s, dir) : s));
  }, []);

  const softDrop = useCallback(() => {
    setState((s) => {
      if (s.phase !== "playing" || !s.active) return s;
      const p = s.active;
      if (!collides(s.board, p.type, p.rot, p.x, p.y + 1)) {
        return {
          ...s,
          active: { ...p, y: p.y + 1 },
          score: s.score + 1,
          lockTimer: 0,
          resetCount: Math.min(MAX_RESETS, (s.resetCount ?? 0) + 1),
        };
      }
      return s;
    });
  }, []);

  const handleHardDrop = useCallback(() => {
    setState((s) => (s.phase === "playing" ? doHardDrop(s) : s));
  }, []);

  /* ---- Hold-to-repeat for horizontal movement ---- */
  const endHold = useCallback(() => {
    if (holdTimerRef.current != null) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    if (holdRepeatRef.current != null) {
      clearInterval(holdRepeatRef.current);
      holdRepeatRef.current = null;
    }
    holdDirRef.current = null;
  }, []);

  const beginHold = useCallback(
    (dir: "left" | "right") => {
      if (holdDirRef.current === dir) return;
      endHold();
      holdDirRef.current = dir;
      move(dir === "left" ? -1 : 1);
      holdTimerRef.current = window.setTimeout(() => {
        holdRepeatRef.current = window.setInterval(() => {
          if (stateRef.current.phase === "playing") move(dir === "left" ? -1 : 1);
        }, 60);
      }, 180);
    },
    [move, endHold]
  );

  useEffect(() => () => endHold(), [endHold]);

  /* ---- Keyboard handling ---- */
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Ignore typing controls if present
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
        return;
      }

      const k = e.key.toLowerCase();
      const code = e.code;
      const playing = stateRef.current.phase === "playing";

      // Determine if this is a rotation key
      const isRotationKey =
        code === "ArrowUp" ||
        code === "KeyW" ||
        code === "KeyX" ||
        code === "KeyZ";

      // Prevent page scroll for game keys.
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", " ", "spacebar"].includes(k) || code === "Space") {
        e.preventDefault();
      }

      // Handle rotation keys separately to manage repeat behavior
      if (isRotationKey) {
        if (e.repeat) {
          // Ignore repeats for rotation keys so each press rotates once
          return;
        }
        
        if (code === "ArrowUp" || code === "KeyW" || code === "KeyX") {
          if (playing) rotate("cw");
        } else if (code === "KeyZ") {
          if (playing) rotate("ccw");
        }
        return;
      }

      switch (k) {
        case "arrowleft":
        case "a":
          if (playing) move(-1);
          break;
        case "arrowright":
        case "d":
          if (playing) move(1);
          break;
        case "arrowdown":
        case "s":
          if (playing) softDrop();
          break;
        case " ":
        case "enter":
          if (code === "Space") {
            if (playing) handleHardDrop();
            else if (stateRef.current.phase === "ready" || stateRef.current.phase === "over") start();
          } else if (k === "enter") {
            if (stateRef.current.phase === "ready" || stateRef.current.phase === "over") start();
          }
          break;
        case "p":
        case "escape":
          togglePause();
          break;
        case "r":
          restart();
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [move, rotate, softDrop, handleHardDrop, togglePause, restart, start]);

  /* ---- Touch controls: prevent scroll on the deck ---- */
  useEffect(() => {
    const el = document.getElementById("cs-deck");
    if (!el) return;
    const prevent = (e: TouchEvent) => {
      if (touchActiveRef.current) e.preventDefault();
    };
    el.addEventListener("touchmove", prevent, { passive: false });
    return () => el.removeEventListener("touchmove", prevent);
  }, []);

  /* ---- Derived render data ---- */
  const display = useMemo(() => {
    const grid: Cell[][] = state.board.map((row) => row.slice());
    const ghostGrid: boolean[][] = Array.from({ length: ROWS }, () => Array<boolean>(COLS).fill(false));
    const p = state.active;
    if (p) {
      const gy = ghostY(state.board, p);
      for (const [r, c] of rotateOffsets(p.type, p.rot)) {
        const x = p.x + c;
        const y = gy + r;
        if (y >= 0 && y < ROWS && x >= 0 && x < COLS && grid[y][x] === null) ghostGrid[y][x] = true;
      }
      for (const [r, c] of rotateOffsets(p.type, p.rot)) {
        const x = p.x + c;
        const y = p.y + r;
        if (y >= 0 && y < ROWS && x >= 0 && x < COLS) grid[y][x] = p.type;
      }
    }
    return { grid, ghostGrid };
  }, [state.board, state.active]);

  const nextPieces = useMemo(() => state.queue.slice(0, 5), [state.queue]);

  const statusText =
    state.phase === "ready"
      ? "Press Start or Space"
      : state.phase === "paused"
      ? "Paused"
      : state.phase === "over"
      ? "Game Over"
      : `Level ${state.level} · ${gravityInterval(state.level)}ms`;

  const isOver = state.phase === "over";
  const isNewBest = isOver && state.score > 0 && state.score >= best;

  /* ------------------------------- Render ------------------------------- */

  return (
    <main className="cs-app">
      <div className="cs-shell">
        {/* Header */}
        <header className="cs-header">
          <h1 className="cs-logo">
            <span className="cs-logo-chroma">CHROMA</span>
            <span className="cs-logo-stack">STACK</span>
          </h1>
          <p className="cs-tagline">Stack the spectrum. Clear the line.</p>
        </header>

        <div className="cs-layout">
          {/* Left stats column */}
          <aside className="cs-panel cs-stats" aria-label="Score panel">
            <Stat label="Score" value={state.score.toLocaleString()} />
            <Stat label="Lines" value={String(state.lines)} />
            <Stat label="Level" value={String(state.level)} />
            <Stat label="Best" value={best.toLocaleString()} highlight />
            <div className="cs-status" role="status" aria-live="polite">
              <span className="cs-status-dot" data-phase={state.phase} />
              <span>{statusText}</span>
            </div>
          </aside>

          {/* Board */}
          <section className="cs-board-wrap" aria-label="Game board">
            <div
              className="cs-board"
              role="grid"
              aria-label={`Chroma Stack board, ${ROWS} by ${COLS}, ${state.phase}`}
            >
              {display.grid.map((row, y) =>
                row.map((cell, x) => {
                  const ghost = display.ghostGrid[y][x];
                  const cls =
                    cell !== null
                      ? `cs-cell cs-block cs-block-${cell}`
                      : ghost
                      ? "cs-cell cs-ghost"
                      : "cs-cell cs-empty";
                  return (
                    <div
                      key={`${y}-${x}`}
                      className={cls}
                      role="gridcell"
                      aria-label={
                        cell !== null
                          ? `Filled block at row ${y + 1}, column ${x + 1}`
                          : ghost
                          ? `Ghost preview at row ${y + 1}, column ${x + 1}`
                          : `Empty at row ${y + 1}, column ${x + 1}`
                      }
                    />
                  );
                })
              )}

              {/* Overlays */}
              {state.phase === "ready" && (
                <Overlay title="Ready?" subtitle="Line up the colors and drop.">
                  <button className="cs-btn cs-btn-primary" onClick={start} autoFocus>
                    Start Game
                  </button>
                </Overlay>
              )}
              {state.phase === "paused" && (
                <Overlay title="Paused" subtitle="Take a breath.">
                  <button className="cs-btn cs-btn-primary" onClick={togglePause} autoFocus>
                    Resume
                  </button>
                  <button className="cs-btn" onClick={restart}>
                    Restart
                  </button>
                </Overlay>
              )}
              {isOver && (
                <Overlay title="Game Over" subtitle={isNewBest ? "New personal best!" : `You scored ${state.score.toLocaleString()}`}>
                  <button className="cs-btn cs-btn-primary" onClick={restart} autoFocus>
                    Play Again
                  </button>
                </Overlay>
              )}
            </div>
          </section>

          {/* Right column: next + guide */}
          <aside className="cs-side">
            <div className="cs-panel cs-next" aria-label="Next pieces">
              <h2 className="cs-panel-title">Next</h2>
              <ul className="cs-next-list">
                {nextPieces.map((t, i) => (
                  <li key={i} className={i === 0 ? "cs-next-item cs-next-first" : "cs-next-item"}>
                    <MiniPiece type={t} label={i === 0 ? "Upcoming piece" : `Preview ${i + 1}`} />
                  </li>
                ))}
              </ul>
            </div>

            <div className="cs-panel cs-guide" aria-label="Keyboard controls">
              <h2 className="cs-panel-title">Controls</h2>
              <dl className="cs-keys">
                <KeyRow keys={["←", "→"]} action="Move" />
                <KeyRow keys={["↓"]} action="Soft drop" />
                <KeyRow keys={["↑", "X"]} action="Rotate CW" />
                <KeyRow keys={["Z"]} action="Rotate CCW" />
                <KeyRow keys={["Space"]} action="Hard drop" />
                <KeyRow keys={["P", "Esc"]} action="Pause" />
                <KeyRow keys={["R"]} action="Restart" />
              </dl>
            </div>
          </aside>
        </div>

        {/* Touch control deck */}
        <nav id="cs-deck" className="cs-deck" aria-label="Touch controls" style={{ touchAction: "none" }}>
          <DeckButton label="Move left" onPress={() => beginHold("left")} onRelease={endHold} touchRef={touchActiveRef}>
            ◀
          </DeckButton>
          <DeckButton label="Rotate counter-clockwise" onPress={() => rotate("ccw")} touchRef={touchActiveRef}>
            ↺
          </DeckButton>
          <DeckButton label="Hard drop" onPress={handleHardDrop} touchRef={touchActiveRef}>
            ⤓
          </DeckButton>
          <DeckButton label="Rotate clockwise" onPress={() => rotate("cw")} touchRef={touchActiveRef}>
            ↻
          </DeckButton>
          <DeckButton label="Move right" onPress={() => beginHold("right")} onRelease={endHold} touchRef={touchActiveRef}>
            ▶
          </DeckButton>
          <div className="cs-deck-row">
            <DeckButton label="Soft drop" onPress={softDrop} touchRef={touchActiveRef}>
              ▼
            </DeckButton>
            <DeckButton label="Pause" onPress={togglePause} touchRef={touchActiveRef}>
              ❚❚
            </DeckButton>
            <DeckButton label="Restart" onPress={restart} touchRef={touchActiveRef}>
              ⟳
            </DeckButton>
          </div>
        </nav>

        <footer className="cs-footer">
          <span>CHROMA STACK</span>
          <span className="cs-sep">·</span>
          <span>Built with React &amp; browser APIs</span>
        </footer>
      </div>

      {/* Live region for announcements */}
      <div className="cs-sr-only" role="status" aria-live="polite">
        {announce}
      </div>
    </main>
  );
}

/* --------------------------- Sub-components --------------------------- */

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={highlight ? "cs-stat cs-stat-highlight" : "cs-stat"}>
      <span className="cs-stat-label">{label}</span>
      <span className="cs-stat-value">{value}</span>
    </div>
  );
}

function KeyRow({ keys, action }: { keys: string[]; action: string }) {
  return (
    <div className="cs-keyrow">
      <dt className="cs-keyset">
        {keys.map((kk) => (
          <kbd key={kk} className="cs-kbd">
            {kk}
          </kbd>
        ))}
      </dt>
      <dd className="cs-keyaction">{action}</dd>
    </div>
  );
}

function MiniPiece({ type, label }: { type: PieceType; label: string }) {
  const cells = rotateOffsets(type, 0);
  const minR = Math.min(...cells.map(([r]) => r));
  const maxR = Math.max(...cells.map(([r]) => r));
  const minC = Math.min(...cells.map(([, c]) => c));
  const maxC = Math.max(...cells.map(([, c]) => c));
  const h = maxR - minR + 1;
  const w = maxC - minC + 1;
  const filled = new Set(cells.map(([r, c]) => `${r - minR},${c - minC}`));
  const size = 4;
  const grid: Cell[][] = [];
  for (let r = 0; r < size; r++) {
    const row: Cell[] = [];
    for (let c = 0; c < size; c++) {
      const rr = r - Math.floor((size - h) / 2);
      const cc = c - Math.floor((size - w) / 2);
      row.push(filled.has(`${rr},${cc}`) ? type : null);
    }
    grid.push(row);
  }
  return (
    <div className="cs-mini" role="img" aria-label={label}>
      {grid.map((row, r) =>
        row.map((cell, c) => (
          <span
            key={`${r}-${c}`}
            className={cell !== null ? `cs-mini-cell cs-mini-fill cs-mini-${cell}` : "cs-mini-cell cs-mini-blank"}
          />
        ))
      )}
    </div>
  );
}

function Overlay({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="cs-overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="cs-overlay-card">
        <h2 className="cs-overlay-title">{title}</h2>
        <p className="cs-overlay-sub">{subtitle}</p>
        <div className="cs-overlay-actions">{children}</div>
      </div>
    </div>
  );
}

function DeckButton({
  label,
  onPress,
  onRelease,
  children,
  touchRef,
}: {
  label: string;
  onPress: () => void;
  onRelease?: () => void;
  children: React.ReactNode;
  touchRef: React.MutableRefObject<boolean>;
}) {
  const handlePointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    touchRef.current = true;
    (e.currentTarget as HTMLButtonElement).focus?.();
    onPress();
  };
  const handlePointerUp = (e: React.PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    touchRef.current = false;
    onRelease?.();
  };
  return (
    <button
      type="button"
      className="cs-deck-btn"
      aria-label={label}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={onRelease ? handlePointerUp : undefined}
      onPointerCancel={onRelease ? handlePointerUp : undefined}
      onContextMenu={(e) => e.preventDefault()}
    >
      {children}
    </button>
  );
}

