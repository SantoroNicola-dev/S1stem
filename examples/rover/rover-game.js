/**
 * CYBER-ROVER TACTICAL EDGE CORE // S1STEM-CDR
 * 100% Original Intellectual Property - Autonomous Outpost Recon & Threat Evasion
 * Generates natural language spatial telemetry, candidate actions,
 * polar noul propositions, and tactical threat metrics for S1stem-CDR.
 */

// Grid Tile Definitions:
// 0 = Cleared Conduit (Empty corridor)
// 1 = Reinforced Bulkhead / Energy Wall
// 2 = Plasma Energy Cell (10 pts)
// 3 = Supercharge Capacitor (50 pts, overclocks rover & stuns sentries)
// 4 = Sentry Containment Bay / Door
// 5 = Warp Transit Relay (wraps left <-> right)
// 6 = Classified Data Terminal (100 pts)

export const TILE = {
  EMPTY: 0,
  WALL: 1,
  PLASMA: 2,
  CAPACITOR: 3,
  DOOR: 4,
  WARP: 5,
  DATA: 6,
};

// 19 x 22 Symmetrical Outpost Facility Matrix
export const OUTPOST_MAP = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 1],
  [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
  [1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1],
  [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
  [1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1],
  [0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
  [1, 1, 1, 1, 2, 1, 0, 1, 1, 4, 1, 1, 0, 1, 2, 1, 1, 1, 1],
  [5, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 5],
  [1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
  [0, 0, 0, 1, 2, 1, 0, 0, 0, 6, 0, 0, 0, 1, 2, 1, 0, 0, 0],
  [1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
  [1, 3, 2, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 3, 1],
  [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1],
  [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

export const DIRS = {
  UP: { dx: 0, dy: -1, name: 'UP', label: 'North Conduit' },
  RIGHT: { dx: 1, dy: 0, name: 'RIGHT', label: 'East Conduit' },
  DOWN: { dx: 0, dy: 1, name: 'DOWN', label: 'South Conduit' },
  LEFT: { dx: -1, dy: 0, name: 'LEFT', label: 'West Conduit' },
};

export const OPPOSITE = {
  UP: 'DOWN',
  DOWN: 'UP',
  LEFT: 'RIGHT',
  RIGHT: 'LEFT',
};

export class RoverGame {
  constructor() {
    this.cols = OUTPOST_MAP[0].length;
    this.rows = OUTPOST_MAP.length;
    this.reset();
  }

  reset() {
    this.grid = OUTPOST_MAP.map(r => [...r]);
    this.score = 0;
    this.shields = 3;
    this.totalCells = 0;
    this.cellsHarvested = 0;
    this.overclockTimeLeft = 0;
    this.sentriesNeutralizedInOverclock = 0;
    this.isGameOver = false;
    this.isWon = false;
    this.decisionCount = 0;
    this.tickCount = 0;

    // Count plasma energy cells & capacitors
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (this.grid[y][x] === TILE.PLASMA || this.grid[y][x] === TILE.CAPACITOR) {
          this.totalCells++;
        }
      }
    }

    // Rover Unit TR-01 (Row 16, Col 9)
    this.rover = {
      x: 9,
      y: 16,
      subX: 9,
      subY: 16,
      dir: 'LEFT',
      nextDir: 'LEFT',
      scannerAngle: 0,
      speed: 0.125, // tiles per tick
    };

    // 4 Autonomous Rogue Sentries
    this.sentries = [
      {
        id: 'vanguard',
        name: 'Vanguard Sentry-X1',
        title: 'Direct Pursuit Drone',
        color: '#ef4444', // Red
        x: 9,
        y: 8,
        subX: 9,
        subY: 8,
        dir: 'LEFT',
        state: 'HUNT', // HUNT, PATROL, STUNNED, RECHARGING
        homeX: 17,
        homeY: 1,
        speed: 0.10,
        patrolCorner: { x: 17, y: 1 },
      },
      {
        id: 'flanker',
        name: 'Flanker Drone-F2',
        title: 'Ambush Interceptor',
        color: '#f472b6', // Pink
        x: 9,
        y: 10,
        subX: 9,
        subY: 10,
        dir: 'UP',
        state: 'HUNT',
        homeX: 1,
        homeY: 1,
        speed: 0.095,
        patrolCorner: { x: 1, y: 1 },
      },
      {
        id: 'stalker',
        name: 'Stalker Unit-S3',
        title: 'Vector Reflector',
        color: '#00f0ff', // Cyan
        x: 8,
        y: 10,
        subX: 8,
        subY: 10,
        dir: 'UP',
        state: 'HUNT',
        homeX: 17,
        homeY: 20,
        speed: 0.09,
        patrolCorner: { x: 17, y: 20 },
      },
      {
        id: 'warden',
        name: 'Warden Mech-W4',
        title: 'Perimeter Feigner',
        color: '#f59e0b', // Amber
        x: 10,
        y: 10,
        subX: 10,
        subY: 10,
        dir: 'UP',
        state: 'HUNT',
        homeX: 1,
        homeY: 20,
        speed: 0.085,
        patrolCorner: { x: 1, y: 20 },
      },
    ];

    this.particles = [];
    this.floatingAlerts = [];
  }

  isPassable(x, y, isSentry = false, isDoorAllowed = false) {
    if (x < 0 || x >= this.cols) return true; // Warp tunnel wrap
    if (y < 0 || y >= this.rows) return false;

    const t = this.grid[y][x];
    if (t === TILE.WALL) return false;
    if (t === TILE.DOOR) {
      return isSentry && isDoorAllowed;
    }
    return true;
  }

  dist(x1, y1, x2, y2) {
    return Math.hypot(x1 - x2, y1 - y2);
  }

  // BFS Shortest Path distance on grid
  bfsDistance(startX, startY, targetX, targetY, isSentry = false) {
    const queue = [{ x: startX, y: startY, d: 0 }];
    const visited = new Set();
    visited.add(`${startX},${startY}`);

    while (queue.length > 0) {
      const cur = queue.shift();
      if (cur.x === targetX && cur.y === targetY) {
        return cur.d;
      }
      for (const key of Object.keys(DIRS)) {
        const d = DIRS[key];
        let nx = cur.x + d.dx;
        let ny = cur.y + d.dy;

        // Wrap around warp tunnel
        if (nx < 0) nx = this.cols - 1;
        else if (nx >= this.cols) nx = 0;

        if (this.isPassable(nx, ny, isSentry, false)) {
          const vKey = `${nx},${ny}`;
          if (!visited.has(vKey)) {
            visited.add(vKey);
            queue.push({ x: nx, y: ny, d: cur.d + 1 });
          }
        }
      }
    }
    return 999;
  }

  // BFS Reachable Clearance (Flood-Fill Free Space)
  countReachableSpace(startX, startY, maxDepth = 25) {
    const queue = [{ x: startX, y: startY, d: 0 }];
    const visited = new Set([`${startX},${startY}`]);

    while (queue.length > 0) {
      const cur = queue.shift();
      if (cur.d >= maxDepth) continue;

      for (const key of Object.keys(DIRS)) {
        const d = DIRS[key];
        let nx = cur.x + d.dx;
        let ny = cur.y + d.dy;
        if (nx < 0) nx = this.cols - 1;
        else if (nx >= this.cols) nx = 0;

        if (this.isPassable(nx, ny, false, false)) {
          const vKey = `${nx},${ny}`;
          if (!visited.has(vKey)) {
            visited.add(vKey);
            queue.push({ x: nx, y: ny, d: cur.d + 1 });
          }
        }
      }
    }
    return visited.size;
  }

  // Yield of plasma cells along conduit
  countCellsInDirection(startX, startY, dirName, depth = 6) {
    const d = DIRS[dirName];
    let count = 0;
    let cx = startX;
    let cy = startY;

    for (let i = 0; i < depth; i++) {
      cx += d.dx;
      cy += d.dy;
      if (cx < 0) cx = this.cols - 1;
      else if (cx >= this.cols) cx = 0;

      if (!this.isPassable(cx, cy, false, false)) break;
      const t = this.grid[cy][cx];
      if (t === TILE.PLASMA) count += 1;
      else if (t === TILE.CAPACITOR) count += 5;
      else if (t === TILE.DATA) count += 10;
    }
    return count;
  }

  // Nearest supercharge capacitor
  findNearestCapacitor(rx, ry) {
    let bestDist = 999;
    let target = null;
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (this.grid[y][x] === TILE.CAPACITOR) {
          const d = this.bfsDistance(rx, ry, x, y);
          if (d < bestDist) {
            bestDist = d;
            target = { x, y, dist: d };
          }
        }
      }
    }
    return target;
  }

  isAtIntersection(x, y) {
    let exits = 0;
    for (const key of Object.keys(DIRS)) {
      const d = DIRS[key];
      let nx = x + d.dx;
      let ny = y + d.dy;
      if (nx < 0) nx = this.cols - 1;
      else if (nx >= this.cols) nx = 0;
      if (this.isPassable(nx, ny, false, false)) exits++;
    }
    return exits >= 3;
  }

  // Evaluate candidate directions for S1stem-CDR
  evaluateDirectionCandidates() {
    const rx = Math.round(this.rover.subX);
    const ry = Math.round(this.rover.subY);
    const candidates = [];

    const nearestCapacitor = this.findNearestCapacitor(rx, ry);
    const isOverclockActive = this.overclockTimeLeft > 0;

    for (const dirKey of Object.keys(DIRS)) {
      const dirInfo = DIRS[dirKey];
      let nx = rx + dirInfo.dx;
      let ny = ry + dirInfo.dy;

      // Warp wrap
      if (nx < 0) nx = this.cols - 1;
      else if (nx >= this.cols) nx = 0;

      const isWalkable = this.isPassable(nx, ny, false, false);
      const isTurnaround = dirKey === OPPOSITE[this.rover.dir];

      if (!isWalkable) {
        candidates.push({
          id: `MOVE_${dirKey}`,
          dir: dirKey,
          label: dirInfo.label,
          description: `Blocked reinforced bulkhead heading ${dirKey}. Impassable structural barrier.`,
          isWalkable: false,
          isSafe: false,
          minSentryDist: 0,
          reachableSpace: 0,
          cellYield: 0,
          bias: -999,
        });
        continue;
      }

      // Check distance to all sentries
      let minHostileDist = 999;
      let nearestHostileSentry = null;
      let minStunnedDist = 999;
      let nearestStunnedSentry = null;

      for (const s of this.sentries) {
        const sx = Math.round(s.subX);
        const sy = Math.round(s.subY);
        const d = this.dist(nx, ny, sx, sy);

        if (s.state === 'STUNNED') {
          if (d < minStunnedDist) {
            minStunnedDist = d;
            nearestStunnedSentry = s;
          }
        } else if (s.state === 'HUNT' || s.state === 'PATROL') {
          if (d < minHostileDist) {
            minHostileDist = d;
            nearestHostileSentry = s;
          }
        }
      }

      const reachableSpace = this.countReachableSpace(nx, ny, 20);
      const cellYield = this.countCellsInDirection(rx, ry, dirKey, 5);

      const isSafe = minHostileDist >= 2.0 && reachableSpace >= 5;
      const isDeadEnd = reachableSpace < 4;

      let distToCapacitor = 999;
      if (nearestCapacitor) {
        distToCapacitor = this.bfsDistance(nx, ny, nearestCapacitor.x, nearestCapacitor.y);
      }

      // Formulate contextual semantic description for S1stem-CDR
      let desc = '';
      if (!isSafe) {
        desc = `Critical hazard flank heading ${dirKey}. Approaching ${nearestHostileSentry ? nearestHostileSentry.name : 'Sentry'} at distance ${minHostileDist.toFixed(1)}m. Severe collision threat.`;
      } else if (isOverclockActive && minStunnedDist < 8) {
        desc = `Offensive neutralizing maneuver heading ${dirKey} engaging disabled ${nearestStunnedSentry ? nearestStunnedSentry.name : 'Sentry'} at distance ${minStunnedDist.toFixed(1)}m. Harvest 200 salvage bonus.`;
      } else if (nearestCapacitor && distToCapacitor <= 4) {
        desc = `Tactical sprint toward Supercharge Capacitor heading ${dirKey}. Capacitor located ${distToCapacitor}m ahead. Secure emergency overcharge shield.`;
      } else if (cellYield >= 3) {
        desc = `High-yield plasma conduit heading ${dirKey} yielding ${cellYield} plasma cells. Open traversable path (${reachableSpace} cells clearance).`;
      } else {
        desc = `Unobstructed tactical conduit heading ${dirKey}. Hostile distance ${minHostileDist.toFixed(1)}m. Reachable clearance: ${reachableSpace} cells.`;
      }

      // Biases for policy
      let bias = 0;
      if (!isSafe) bias -= 8.0;
      if (isTurnaround && this.isAtIntersection(rx, ry)) bias -= 1.5;
      if (isDeadEnd) bias -= 4.0;
      bias += Math.min(4.0, cellYield * 0.8);
      bias += Math.min(2.5, (reachableSpace / 20) * 2.0);

      if (isOverclockActive && minStunnedDist < 999) {
        bias += Math.max(0, (10 - minStunnedDist) * 1.2);
      } else if (!isOverclockActive && nearestCapacitor && distToCapacitor < 999) {
        if (minHostileDist < 4.0) {
          bias += Math.max(0, (6 - distToCapacitor) * 1.5);
        }
      }

      candidates.push({
        id: `MOVE_${dirKey}`,
        dir: dirKey,
        label: dirInfo.label,
        description: desc,
        isWalkable: true,
        isSafe,
        isDeadEnd,
        minSentryDist: minHostileDist,
        stunnedDist: minStunnedDist,
        reachableSpace,
        cellYield,
        bias,
      });
    }

    return candidates;
  }

  // Generate State Observation Prompt for S1stem-CDR
  generateStatePrompt() {
    const rx = Math.round(this.rover.subX);
    const ry = Math.round(this.rover.subY);

    const sentryReports = this.sentries.map(s => {
      const d = this.dist(rx, ry, s.subX, s.subY).toFixed(1);
      return `${s.name} (${s.state}, ${d}m)`;
    }).join(', ');

    const capacitorStatus = this.overclockTimeLeft > 0
      ? `Overclock active (${(this.overclockTimeLeft / 60).toFixed(1)}s remaining, sentries disabled)`
      : 'Standard drive (Sentries active & hostile)';

    const nearestCapacitor = this.findNearestCapacitor(rx, ry);
    const capacitorInfo = nearestCapacitor
      ? `Nearest Supercharge Capacitor at (${nearestCapacitor.x}, ${nearestCapacitor.y}), distance ${nearestCapacitor.dist}m.`
      : 'All Supercharge Capacitors depleted.';

    return `CYBER-ROVER TR-01 telemetry at sector coordinate (${rx}, ${ry}). Score: ${this.score}. Plasma cells remaining: ${this.totalCells - this.cellsHarvested}.
Capacitor system: ${capacitorStatus}.
Rogue sentries: ${sentryReports}.
${capacitorInfo}
Current velocity: heading ${this.rover.dir}. Hull shields: ${this.shields}.`;
  }

  // Generate Polar Noul Propositions for Boolean Hypothesis Verification
  generatePolarHypotheses() {
    const rx = Math.round(this.rover.subX);
    const ry = Math.round(this.rover.subY);

    let nearestHostileDist = 999;
    let closestSentry = this.sentries[0];
    for (const s of this.sentries) {
      if (s.state === 'HUNT' || s.state === 'PATROL') {
        const d = this.dist(rx, ry, s.subX, s.subY);
        if (d < nearestHostileDist) {
          nearestHostileDist = d;
          closestSentry = s;
        }
      }
    }

    const nearestCapacitor = this.findNearestCapacitor(rx, ry);

    return [
      {
        question: `Is the rogue sentry ${closestSentry.name} within dangerous intercept range?`,
        groundTruth: nearestHostileDist < 3.5 ? 'yes' : 'no',
        context: `TR-01 at (${rx}, ${ry}). Sentry ${closestSentry.name} at distance ${nearestHostileDist.toFixed(1)}m.`,
      },
      {
        question: `Can TR-01 safely secure the supercharge capacitor before sentry collision?`,
        groundTruth: (nearestCapacitor && nearestCapacitor.dist < nearestHostileDist) ? 'yes' : 'no',
        context: nearestCapacitor
          ? `Capacitor distance ${nearestCapacitor.dist}m vs hostile sentry ${closestSentry.name} distance ${nearestHostileDist.toFixed(1)}m.`
          : `No capacitors available.`,
      },
      {
        question: `Is the current sector clear of immediate ambush hazards?`,
        groundTruth: nearestHostileDist >= 4.0 ? 'yes' : 'no',
        context: `Sector perimeter clearance ${nearestHostileDist.toFixed(1)}m from rogue pursuit vectors.`,
      },
    ];
  }

  // Advance simulation by 1 tick
  step(chosenDir) {
    if (this.isGameOver || this.isWon) return;

    this.tickCount++;

    // Update Overclock timer
    if (this.overclockTimeLeft > 0) {
      this.overclockTimeLeft--;
      if (this.overclockTimeLeft === 0) {
        for (const s of this.sentries) {
          if (s.state === 'STUNNED') s.state = 'HUNT';
        }
      }
    }

    // Apply Chosen Direction
    if (chosenDir && DIRS[chosenDir]) {
      const d = DIRS[chosenDir];
      const curX = Math.round(this.rover.subX);
      const curY = Math.round(this.rover.subY);
      let nx = curX + d.dx;
      let ny = curY + d.dy;
      if (nx < 0) nx = this.cols - 1;
      else if (nx >= this.cols) nx = 0;

      if (this.isPassable(nx, ny, false, false)) {
        this.rover.nextDir = chosenDir;
      }
    }

    // Move Rover
    this._moveRover();

    // Check Pickups
    this._checkPickups();

    // Move Sentries AI
    this._moveSentries();

    // Check Collisions
    this._checkSentryCollisions();

    // Update Particle Systems
    this._updateParticles();
  }

  _moveRover() {
    const curX = Math.round(this.rover.subX);
    const curY = Math.round(this.rover.subY);
    const isAtCenter = Math.abs(this.rover.subX - curX) < 0.08 && Math.abs(this.rover.subY - curY) < 0.08;

    if (isAtCenter && this.rover.nextDir !== this.rover.dir) {
      const nextD = DIRS[this.rover.nextDir];
      let nx = curX + nextD.dx;
      let ny = curY + nextD.dy;
      if (nx < 0) nx = this.cols - 1;
      else if (nx >= this.cols) nx = 0;

      if (this.isPassable(nx, ny, false, false)) {
        this.rover.dir = this.rover.nextDir;
        this.rover.subX = curX;
        this.rover.subY = curY;
      }
    }

    const curD = DIRS[this.rover.dir];
    let targetX = this.rover.subX + curD.dx * this.rover.speed;
    let targetY = this.rover.subY + curD.dy * this.rover.speed;

    const nextTileX = Math.round(targetX + curD.dx * 0.45);
    const nextTileY = Math.round(targetY + curD.dy * 0.45);

    // Warp wrap
    if (targetX < 0) {
      targetX = this.cols - 1;
    } else if (targetX >= this.cols) {
      targetX = 0;
    }

    if (this.isPassable(nextTileX, nextTileY, false, false)) {
      this.rover.subX = targetX;
      this.rover.subY = targetY;
      this.rover.x = Math.round(this.rover.subX);
      this.rover.y = Math.round(this.rover.subY);

      this.rover.scannerAngle += 0.12;
    }
  }

  _checkPickups() {
    const rx = Math.round(this.rover.subX);
    const ry = Math.round(this.rover.subY);
    if (rx < 0 || rx >= this.cols || ry < 0 || ry >= this.rows) return;

    const t = this.grid[ry][rx];
    if (t === TILE.PLASMA) {
      this.grid[ry][rx] = TILE.EMPTY;
      this.score += 10;
      this.cellsHarvested++;
      this.spawnParticles(rx, ry, '#10b981', 3);
    } else if (t === TILE.CAPACITOR) {
      this.grid[ry][rx] = TILE.EMPTY;
      this.score += 50;
      this.cellsHarvested++;
      this.overclockTimeLeft = 360; // ~6 seconds
      this.sentriesNeutralizedInOverclock = 0;
      for (const s of this.sentries) {
        if (s.state !== 'RECHARGING') {
          s.state = 'STUNNED';
        }
      }
      this.spawnParticles(rx, ry, '#00f0ff', 14);
      this.floatingAlerts.push({ x: rx, y: ry, text: '+50 OVERCLOCK ACTIVE', opacity: 1.0, color: '#00f0ff' });
    } else if (t === TILE.DATA) {
      this.grid[ry][rx] = TILE.EMPTY;
      this.score += 100;
      this.spawnParticles(rx, ry, '#ec4899', 8);
      this.floatingAlerts.push({ x: rx, y: ry, text: '+100 DATA TERMINAL', opacity: 1.0, color: '#ec4899' });
    }

    if (this.cellsHarvested >= this.totalCells) {
      this.isWon = true;
    }
  }

  _moveSentries() {
    const rx = Math.round(this.rover.subX);
    const ry = Math.round(this.rover.subY);

    for (const s of this.sentries) {
      const sx = Math.round(s.subX);
      const sy = Math.round(s.subY);
      const isAtTileCenter = Math.abs(s.subX - sx) < 0.08 && Math.abs(s.subY - sy) < 0.08;

      if (isAtTileCenter) {
        s.subX = sx;
        s.subY = sy;
        s.x = sx;
        s.y = sy;

        let targetX = rx;
        let targetY = ry;

        if (s.state === 'RECHARGING') {
          targetX = 9;
          targetY = 9;
          if (sx === 9 && (sy === 8 || sy === 9 || sy === 10)) {
            s.state = 'HUNT';
          }
        } else if (s.state === 'STUNNED') {
          targetX = Math.floor(Math.random() * this.cols);
          targetY = Math.floor(Math.random() * this.rows);
        } else if (s.state === 'PATROL') {
          targetX = s.patrolCorner.x;
          targetY = s.patrolCorner.y;
        } else {
          // Tactical HUNT personalities:
          if (s.id === 'vanguard') {
            targetX = rx;
            targetY = ry;
          } else if (s.id === 'flanker') {
            const rD = DIRS[this.rover.dir];
            targetX = rx + rD.dx * 3;
            targetY = ry + rD.dy * 3;
          } else if (s.id === 'stalker') {
            const vanguard = this.sentries[0];
            targetX = rx * 2 - Math.round(vanguard.subX);
            targetY = ry * 2 - Math.round(vanguard.subY);
          } else if (s.id === 'warden') {
            const d = this.dist(sx, sy, rx, ry);
            if (d > 6) {
              targetX = rx;
              targetY = ry;
            } else {
              targetX = s.patrolCorner.x;
              targetY = s.patrolCorner.y;
            }
          }
        }

        let bestDir = s.dir;
        let bestDist = 999999;
        const opp = OPPOSITE[s.dir];

        for (const dirKey of Object.keys(DIRS)) {
          if (dirKey === opp && s.state !== 'RECHARGING') continue;

          const d = DIRS[dirKey];
          let nx = sx + d.dx;
          let ny = sy + d.dy;

          if (nx < 0) nx = this.cols - 1;
          else if (nx >= this.cols) nx = 0;

          if (this.isPassable(nx, ny, true, s.state === 'RECHARGING' || sy === 10)) {
            const dist = this.dist(nx, ny, targetX, targetY);
            if (dist < bestDist) {
              bestDist = dist;
              bestDir = dirKey;
            }
          }
        }
        s.dir = bestDir;
      }

      const sSpeed = s.state === 'STUNNED' ? s.speed * 0.65 : (s.state === 'RECHARGING' ? s.speed * 2.0 : s.speed);
      const d = DIRS[s.dir];
      s.subX += d.dx * sSpeed;
      s.subY += d.dy * sSpeed;

      if (s.subX < 0) s.subX = this.cols - 1;
      else if (s.subX >= this.cols) s.subX = 0;
    }
  }

  _checkSentryCollisions() {
    for (const s of this.sentries) {
      const d = this.dist(this.rover.subX, this.rover.subY, s.subX, s.subY);
      if (d < 0.75) {
        if (s.state === 'STUNNED') {
          // Neutralize Sentry
          s.state = 'RECHARGING';
          this.sentriesNeutralizedInOverclock++;
          const pts = 200 * Math.pow(2, this.sentriesNeutralizedInOverclock - 1);
          this.score += pts;
          this.spawnParticles(s.subX, s.subY, '#38bdf8', 16);
          this.floatingAlerts.push({
            x: s.subX,
            y: s.subY,
            text: `+${pts} NEUTRALIZED`,
            opacity: 1.0,
            color: '#38bdf8',
          });
        } else if (s.state === 'HUNT' || s.state === 'PATROL') {
          // Rover Shield Breach
          this.shields--;
          this.spawnParticles(this.rover.subX, this.rover.subY, '#ef4444', 20);
          if (this.shields <= 0) {
            this.isGameOver = true;
          } else {
            this.rover.subX = 9;
            this.rover.subY = 16;
            this.rover.x = 9;
            this.rover.y = 16;
            this.rover.dir = 'LEFT';
            this.rover.nextDir = 'LEFT';
          }
          break;
        }
      }
    }
  }

  spawnParticles(x, y, color, count = 5) {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.5 + Math.random() * 2.0;
      this.particles.push({
        x,
        y,
        vx: Math.cos(angle) * speed * 0.1,
        vy: Math.sin(angle) * speed * 0.1,
        color,
        life: 1.0,
        decay: 0.03 + Math.random() * 0.04,
      });
    }
  }

  _updateParticles() {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= p.decay;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }

    for (let i = this.floatingAlerts.length - 1; i >= 0; i--) {
      const f = this.floatingAlerts[i];
      f.y -= 0.02;
      f.opacity -= 0.025;
      if (f.opacity <= 0) {
        this.floatingAlerts.splice(i, 1);
      }
    }
  }
}
