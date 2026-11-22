// Sokoban solver: room-decomposed push-optimal A*.
//
// A level is a hallway with one to four rooms off it. A box pushed out of a
// room into the hallway can never come back, so each room is its own little
// Sokoban, solved on its own from its door, and the rooms' optima add up to
// the level's. The player walks from room to room through the hallway in
// between, so the rooms are solved in turn and the moves concatenated.
//
// Each room is searched over pushes, not steps: a state is the room's boxes
// plus the player's reachable region (named by its lowest square), and A*
// orders states by pushes made plus a matching heuristic -- the cheapest way
// to pair every box with a distinct goal, where the price of a pair is the
// fewest pushes that box needs to reach that goal with the other boxes out
// of the way. That price comes from a pull-back pass from the goals, which
// also marks every square a box can never leave (a dead square). A second
// deadlock is pruned too: a box caught in a frozen 2 x 2 block of walls and
// boxes, one of them off its goal.
//
// The search is capped by states generated, never by the clock, so a run
// always answers the same way twice. If a room's A* gives up, a plain
// breadth-first search over the same states is tried before the instance is
// left unanswered; a room that fails both leaves its instance empty rather
// than risking the whole run.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four steps: up, down, left, right. The order is fixed everywhere, so
// every tie in the search and every walk comes out the same on every run.
const DIRS: { dx: number; dy: number; letter: string }[] = [
  { dx: 0, dy: -1, letter: 'u' },
  { dx: 0, dy: 1, letter: 'd' },
  { dx: -1, dy: 0, letter: 'l' },
  { dx: 1, dy: 0, letter: 'r' },
];

// Work budgets, in states generated: never the clock, so the answers are
// deterministic. A room the generator keeps is solvable inside 250,000
// states by its own breadth-first search, and A* looks at fewer states
// than that; the fallback gets more room, and the whole run has a ceiling
// so one bad room can never eat the set's 60 seconds.
const ASTAR_CAP = 250_000;
const BFS_CAP = 400_000;
const RUN_BUDGET = 2_000_000;
let workSpent = 0;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

type Room = {
  boxes: number[];
  goals: number[];
  done: boolean;
};

/** A level in XSB text: its width and height, walls, goals, boxes, player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < rows.length; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // Nothing leaves the board: the outermost frame is wall whatever it shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For each goal, the fewest pushes a lone box needs to reach it from every
 * square, other boxes and the player ignored. Found by pulling back from
 * the goal: a box at t could have come from b = t - d by a push in direction
 * d, which needs the player to have stood at t - 2d. A square with no such
 * distance to any goal is dead: a box there can never be pushed home.
 */
function pushDistances(wall: Uint8Array, goals: number[], w: number, h: number): Int32Array[] {
  const size = w * h;
  const byGoal: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const t = queue[qi];
      const tx = t % w;
      const ty = Math.floor(t / w);
      // Push up: the box was below, the player two below.
      if (ty + 2 < h) {
        const b = t + w;
        if (!wall[b] && !wall[t + 2 * w] && dist[b] < 0) { dist[b] = dist[t] + 1; queue.push(b); }
      }
      // Push down: the box was above, the player two above.
      if (ty - 2 >= 0) {
        const b = t - w;
        if (!wall[b] && !wall[t - 2 * w] && dist[b] < 0) { dist[b] = dist[t] + 1; queue.push(b); }
      }
      // Push left: the box was right, the player two right.
      if (tx + 2 < w) {
        const b = t + 1;
        if (!wall[b] && !wall[t + 2] && dist[b] < 0) { dist[b] = dist[t] + 1; queue.push(b); }
      }
      // Push right: the box was left, the player two left.
      if (tx - 2 >= 0) {
        const b = t - 1;
        if (!wall[b] && !wall[t - 2] && dist[b] < 0) { dist[b] = dist[t] + 1; queue.push(b); }
      }
    }
    byGoal.push(dist);
  }
  return byGoal;
}

/**
 * The level's rooms: cut the hallway row's dead squares out of the floor --
 * a box in the hallway reaches no goal -- and each part left that holds a
 * box is a room. Null when the level is no set of rooms, so the caller can
 * fall back to searching the whole level at once.
 */
function roomsOf(level: Level, live: Uint8Array): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const size = w * h;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      const x = p % w;
      if (p - w >= 0) {
        const q = p - w;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
      if (p + w < size) {
        const q = p + w;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
      if (x > 0) {
        const q = p - 1;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
      if (x < w - 1) {
        const q = p + 1;
        if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rooms.length; squares.push(q); }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] >= 0) rooms[part[g]].goals.push(g);
  }
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Every ordering of n items, for the matching heuristic (n is at most 5). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Uint8Array(n);
  const build = (): void => {
    if (cur.length === n) { out.push(cur.slice()); return; }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = 1;
      cur.push(i);
      build();
      cur.pop();
      used[i] = 0;
    }
  };
  build();
  return out;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    if (ch === 'u') p -= w;
    else if (ch === 'd') p += w;
    else if (ch === 'l') p -= 1;
    else if (ch === 'r') p += 1;
  }
  return p;
}

/** The shortest walk from `from` to `to` around walls and boxes, in LURD letters. */
function walkPath(wall: Uint8Array, occupied: Uint8Array, from: number, to: number, w: number, size: number): string {
  if (from === to) return '';
  const prev = new Int32Array(size).fill(-1);
  const how = new Uint8Array(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    const x = p % w;
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k].dy * w + DIRS[k].dx;
      const q = p + d;
      if (q < 0 || q >= size) continue;
      if (k === 2 && x === 0) continue;
      if (k === 3 && x === w - 1) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkPath: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = DIRS[how[p]].letter + s;
  return s;
}

/**
 * Push-optimal search for one sub-problem (a room, or the whole level as a
 * fallback): A* when `useHeuristic`, breadth-first otherwise. A state is
 * the boxes plus the player's reachable region, named by its lowest square;
 * the first solved state popped from the heap is push-optimal. Returns the
 * pushes and the moves, or null when the budget runs out.
 */
function searchPushes(
  sub: Level,
  cap: number,
  useHeuristic: boolean,
): { pushes: number; moves: string } | null {
  if (workSpent >= RUN_BUDGET) return null;
  const { w, h, wall, goals, boxes, player } = sub;
  const size = w * h;
  const n = goals.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const byGoal = pushDistances(wall, goals, w, h);
  const live = new Uint8Array(size);
  for (let c = 0; c < size; c++) {
    for (let j = 0; j < n; j++) {
      if (byGoal[j][c] >= 0) { live[c] = 1; break; }
    }
  }
  const startBoxes = boxes.slice().sort((a, b) => a - b);
  for (const b of startBoxes) if (!live[b]) return null;
  const solved = (bs: number[]): boolean => {
    for (let i = 0; i < n; i++) if (!goalSet[bs[i]]) return false;
    return true;
  };
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // The matching heuristic: pair each box with a distinct goal as cheaply
  // as the lone-box distances allow. Every box needs at least its pair's
  // price in pushes, and pushes move one box at a time, so the sum never
  // overestimates what is left. With many boxes the pairing is replaced by
  // the sum of each box's cheapest goal, which is admissible the same way.
  const perms = n <= 5 ? permutations(n) : [];
  const heuristic = (bs: number[]): number => {
    if (!useHeuristic) return 0;
    if (n > 5) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        let m = Infinity;
        for (let j = 0; j < n; j++) {
          const d = byGoal[j][bs[i]];
          if (d >= 0 && d < m) m = d;
        }
        if (m === Infinity) return Infinity;
        s += m;
      }
      return s;
    }
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = byGoal[p[i]][bs[i]];
        if (d < 0) { s = Infinity; break; }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  // Flood fill: the squares the player reaches from `from` around walls and
  // the boxes in `occupied`, marked in `into` with a fresh stamp; returns
  // the region's lowest square. Two stamp arrays are kept so a child's
  // region never overwrites the parent's while its pushes are still being
  // tried.
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const regionLow = (into: Int32Array, occupied: Uint8Array, from: number): number => {
    stamp += 1;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    into[from] = stamp;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < low) low = p;
      const x = p % w;
      let q = p - w;
      if (q >= 0 && !wall[q] && !occupied[q] && into[q] !== stamp) { into[q] = stamp; stack[sp++] = q; }
      q = p + w;
      if (q < size && !wall[q] && !occupied[q] && into[q] !== stamp) { into[q] = stamp; stack[sp++] = q; }
      if (x > 0) {
        q = p - 1;
        if (!wall[q] && !occupied[q] && into[q] !== stamp) { into[q] = stamp; stack[sp++] = q; }
      }
      if (x < w - 1) {
        q = p + 1;
        if (!wall[q] && !occupied[q] && into[q] !== stamp) { into[q] = stamp; stack[sp++] = q; }
      }
    }
    return low;
  };

  // Whether the box just pushed to `t` sits in a frozen 2 x 2 block of
  // walls and boxes with a box off its goal: nothing in such a block moves
  // again, so the state is dead.
  const frozenAt = (occupied: Uint8Array, t: number): boolean => {
    const blocked = (i: number): boolean => i < 0 || i >= size || wall[i] === 1 || occupied[i] === 1;
    const offGoal = (i: number): boolean => i >= 0 && i < size && occupied[i] === 1 && goalSet[i] !== 1;
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      if (a < 0 || a >= size || a % w === w - 1 || a + w >= size) continue;
      if (blocked(a) && blocked(a + 1) && blocked(a + w) && blocked(a + w + 1)) {
        if (offGoal(a) || offGoal(a + 1) || offGoal(a + w) || offGoal(a + w + 1)) return true;
      }
    }
    return false;
  };

  // A state's key: the region's lowest square and the boxes' squares, as
  // digits in base `size`. The region is the connected part holding its
  // lowest square, so the key names the state exactly.
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (let i = 0; i < bs.length; i++) k = k * size + bs[i];
    return k;
  };

  // The states, in the order found: their boxes, the player's square, the
  // push that made them from their parent, and their push count and key.
  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sDir: number[] = [-1];
  const sG: number[] = [0];
  const sKey: number[] = [];
  const occupied = new Uint8Array(size);
  for (const b of startBoxes) occupied[b] = 1;
  const startLow = regionLow(seen, occupied, player);
  for (const b of startBoxes) occupied[b] = 0;
  sKey.push(keyOf(startBoxes, startLow));
  const best = new Map<number, number>([[sKey[0], 0]]);

  // The open list, a heap on (f, -g, insertion order): fewest pushes plus
  // heuristic first, deeper first on a tie, first found first after that.
  const heap: number[][] = [];
  const heapLess = (a: number[], b: number[]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
        if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
        if (m === i) break;
        const tmp = heap[i];
        heap[i] = heap[m];
        heap[m] = tmp;
        i = m;
      }
    }
    return top;
  };

  let seq = 0;
  let generated = 1;
  heapPush([heuristic(startBoxes), 0, seq++, 0]);
  let found = -1;
  while (heap.length > 0) {
    const entry = heapPop();
    const si = entry[3];
    const g = sG[si];
    if (best.get(sKey[si]) !== g) continue;
    if (solved(sBoxes[si])) { found = si; break; }
    const curBoxes = sBoxes[si];
    for (const b of curBoxes) occupied[b] = 1;
    regionLow(seen, occupied, sPlayer[si]);
    const here = stamp;
    for (let bi = 0; bi < curBoxes.length && found < 0; bi++) {
      const b = curBoxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k].dy * w + DIRS[k].dx;
        const t = b + d;
        const s = b - d;
        if (t < 0 || t >= size || s < 0 || s >= size) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        if (seen[s] !== here) continue;
        const next = curBoxes.slice();
        next[bi] = t;
        next.sort((a, c) => a - c);
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(occupied, t);
        let key = -1;
        if (!frozen) key = keyOf(next, regionLow(mark, occupied, b));
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || key < 0) continue;
        const bg = best.get(key);
        if (bg !== undefined && bg <= g + 1) continue;
        if (generated >= cap || workSpent >= RUN_BUDGET) {
          for (const x of curBoxes) occupied[x] = 0;
          return null;
        }
        generated += 1;
        workSpent += 1;
        const h = heuristic(next);
        if (h === Infinity) continue;
        best.set(key, g + 1);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(si);
        sFrom.push(b);
        sTo.push(t);
        sDir.push(k);
        sG.push(g + 1);
        sKey.push(key);
        heapPush([g + 1 + h, -(g + 1), seq++, sBoxes.length - 1]);
      }
    }
    for (const x of curBoxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let curPlayer = player;
  let curBoxes = startBoxes.slice();
  for (const si of path) {
    const from = sFrom[si];
    const to = sTo[si];
    const k = sDir[si];
    const d = DIRS[k].dy * w + DIRS[k].dx;
    for (const b of curBoxes) occupied[b] = 1;
    moves += walkPath(wall, occupied, curPlayer, from - d, w, size);
    for (const b of curBoxes) occupied[b] = 0;
    moves += DIRS[k].letter;
    curPlayer = from;
    const bi = curBoxes.indexOf(from);
    curBoxes[bi] = to;
    curBoxes.sort((a, c) => a - c);
  }
  return { pushes: path.length, moves };
}

/** One level's moves: room by room, the player walking on between rooms. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const size = level.w * level.h;
  const byGoal = pushDistances(level.wall, level.goals, level.w, level.h);
  const live = new Uint8Array(size);
  for (let c = 0; c < size; c++) {
    for (let j = 0; j < level.goals.length; j++) {
      if (byGoal[j][c] >= 0) { live[c] = 1; break; }
    }
  }
  const rooms = roomsOf(level, live);
  if (rooms === null) {
    const whole = searchPushes(level, BFS_CAP, false);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    // The other rooms' boxes stand as walls while this room is solved.
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      const src = other.done ? other.goals : other.boxes;
      for (const b of src) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let res = searchPushes(sub, ASTAR_CAP, true);
    if (res === null) res = searchPushes(sub, BFS_CAP, false);
    if (res === null) return '';
    moves += res.moves;
    player = endOf(level.w, player, res.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
