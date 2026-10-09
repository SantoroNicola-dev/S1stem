/**
 * Autonomous Tetris Engine with Macro-Placement Evaluation
 * ========================================================
 * Standard 10x20 Grid with classic 7 Tetromino shapes (I, J, L, O, S, T, Z).
 * 
 * Strategic Evaluation for System 1 S1stem:
 * 1. Completing Lines (Primary Goal / Huge Reward)
 * 2. Hole Avoidance (Strict Constraint / Severe Penalty for covered empty pockets)
 * 3. Surface Bumpiness (Minimizing height differences between adjacent columns)
 * 4. Aggregate & Peak Height Management (Preserving vertical buffer)
 */

export const BOARD_WIDTH = 10;
export const BOARD_HEIGHT = 20;

export const TETROMINOES = {
  I: {
    color: '#06b6d4', // Cyan
    shapes: [
      [[1, 1, 1, 1]],
      [[1], [1], [1], [1]],
    ],
  },
  J: {
    color: '#3b82f6', // Blue
    shapes: [
      [[1, 0, 0], [1, 1, 1]],
      [[1, 1], [1, 0], [1, 0]],
      [[1, 1, 1], [0, 0, 1]],
      [[0, 1], [0, 1], [1, 1]],
    ],
  },
  L: {
    color: '#f97316', // Orange
    shapes: [
      [[0, 0, 1], [1, 1, 1]],
      [[1, 0], [1, 0], [1, 1]],
      [[1, 1, 1], [1, 0, 0]],
      [[1, 1], [0, 1], [0, 1]],
    ],
  },
  O: {
    color: '#eab308', // Yellow
    shapes: [
      [[1, 1], [1, 1]],
    ],
  },
  S: {
    color: '#22c55e', // Green
    shapes: [
      [[0, 1, 1], [1, 1, 0]],
      [[1, 0], [1, 1], [0, 1]],
    ],
  },
  T: {
    color: '#a855f7', // Purple
    shapes: [
      [[0, 1, 0], [1, 1, 1]],
      [[1, 0], [1, 1], [1, 0]],
      [[1, 1, 1], [0, 1, 0]],
      [[0, 1], [1, 1], [0, 1]],
    ],
  },
  Z: {
    color: '#ef4444', // Red
    shapes: [
      [[1, 1, 0], [0, 1, 1]],
      [[0, 1], [1, 1], [1, 0]],
    ],
  },
};

export class TetrisGame {
  constructor() {
    this.width = BOARD_WIDTH;
    this.height = BOARD_HEIGHT;
    this.grid = this.createEmptyGrid();
    this.score = 0;
    this.linesCleared = 0;
    this.level = 1;
    this.tetrisCount = 0;
    this.isGameOver = false;

    this.currentPiece = null;
    this.nextPieceType = null;
    this.currentPlacement = null; // Target (x, rot) selected by S1stem
    
    this.reset();
  }

  createEmptyGrid() {
    return Array.from({ length: this.height }, () => Array(this.width).fill(0));
  }

  reset() {
    this.grid = this.createEmptyGrid();
    this.score = 0;
    this.linesCleared = 0;
    this.level = 1;
    this.tetrisCount = 0;
    this.isGameOver = false;
    this.nextPieceType = this.randomPieceType();
    this.spawnNextPiece();
  }

  randomPieceType() {
    const types = Object.keys(TETROMINOES);
    return types[Math.floor(Math.random() * types.length)];
  }

  spawnNextPiece() {
    const type = this.nextPieceType || this.randomPieceType();
    this.nextPieceType = this.randomPieceType();
    const data = TETROMINOES[type];

    this.currentPiece = {
      type,
      color: data.color,
      rotation: 0,
      shape: data.shapes[0],
      x: Math.floor((this.width - data.shapes[0][0].length) / 2),
      y: 0,
    };

    // Check game over
    if (this.checkCollision(this.grid, this.currentPiece.shape, this.currentPiece.x, this.currentPiece.y)) {
      this.isGameOver = true;
    }
  }

  checkCollision(grid, shape, posX, posY) {
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] !== 0) {
          const x = posX + c;
          const y = posY + r;
          if (x < 0 || x >= this.width || y >= this.height) {
            return true;
          }
          if (y >= 0 && grid[y][x] !== 0) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Drops a shape to its lowest legal landing row on a given grid
   */
  findDropRow(grid, shape, posX) {
    let y = 0;
    while (!this.checkCollision(grid, shape, posX, y + 1)) {
      y++;
    }
    return y;
  }

  /**
   * Generates all valid landing macro-placements for the current falling piece
   */
  generateCandidatePlacements() {
    if (!this.currentPiece || this.isGameOver) return [];

    const type = this.currentPiece.type;
    const shapes = TETROMINOES[type].shapes;
    const candidates = [];

    for (let rot = 0; rot < shapes.length; rot++) {
      const shape = shapes[rot];
      const shapeWidth = shape[0].length;

      for (let x = 0; x <= this.width - shapeWidth; x++) {
        // If immediate top row collides, skip
        if (this.checkCollision(this.grid, shape, x, 0)) {
          continue;
        }

        const dropY = this.findDropRow(this.grid, shape, x);

        // Simulate hypothetical board after placing this piece
        const simGrid = this.grid.map((row) => [...row]);
        for (let r = 0; r < shape.length; r++) {
          for (let c = 0; c < shape[r].length; c++) {
            if (shape[r][c] !== 0) {
              const gy = dropY + r;
              const gx = x + c;
              if (gy >= 0 && gy < this.height) {
                simGrid[gy][gx] = type;
              }
            }
          }
        }

        // Evaluate hypothetical board metrics
        const metrics = this.evaluateGrid(simGrid);

        candidates.push({
          id: `PLACE_ROT${rot}_COL${x}`,
          rotation: rot,
          shape,
          x,
          dropY,
          metrics,
          label: `Col ${x}, Rot ${rot * 90}°`,
          description: this.formatCandidateDescription(x, rot, metrics),
        });
      }
    }

    return candidates;
  }

  evaluateGrid(grid) {
    // 1. Completed Lines
    let completedLines = 0;
    for (let r = 0; r < this.height; r++) {
      if (grid[r].every((cell) => cell !== 0)) {
        completedLines++;
      }
    }

    // 2. Column Heights & Aggregate Height
    const heights = Array(this.width).fill(0);
    for (let c = 0; c < this.width; c++) {
      for (let r = 0; r < this.height; r++) {
        if (grid[r][c] !== 0) {
          heights[c] = this.height - r;
          break;
        }
      }
    }

    const aggregateHeight = heights.reduce((a, b) => a + b, 0);
    const maxHeight = Math.max(...heights);

    // 3. Holes (empty cells covered by at least one occupied cell above)
    let holes = 0;
    for (let c = 0; c < this.width; c++) {
      let blockSeen = false;
      for (let r = 0; r < this.height; r++) {
        if (grid[r][c] !== 0) {
          blockSeen = true;
        } else if (blockSeen) {
          holes++;
        }
      }
    }

    // 4. Bumpiness (sum of adjacent height differences)
    let bumpiness = 0;
    for (let c = 0; c < this.width - 1; c++) {
      bumpiness += Math.abs(heights[c] - heights[c + 1]);
    }

    return {
      completedLines,
      holes,
      aggregateHeight,
      maxHeight,
      bumpiness,
    };
  }

  formatCandidateDescription(col, rot, m) {
    const linesTxt = m.completedLines > 0 ? `Clears ${m.completedLines} lines!` : 'No lines cleared.';
    const holesTxt = m.holes === 0 ? 'Creates ZERO holes (perfect).' : `Produces ${m.holes} buried holes.`;
    const heightTxt = `Max height ${m.maxHeight}/20, bumpiness ${m.bumpiness}.`;

    return `${linesTxt} ${holesTxt} ${heightTxt}`;
  }

  generateStatePrompt() {
    const currentMetrics = this.evaluateGrid(this.grid);
    const piece = this.currentPiece ? this.currentPiece.type : 'NONE';
    const next = this.nextPieceType || 'NONE';

    return [
      `TETRIS GOAL HIERARCHY: 1. Maximize Completed Lines. 2. Strictly Minimize Holes. 3. Keep Surface Flat (low bumpiness). 4. Maintain Safe Low Height.`,
      `CURRENT STATE: Piece=${piece}, Next=${next}. Board MaxHeight=${currentMetrics.maxHeight}/${this.height}, AggregateHeight=${currentMetrics.aggregateHeight}, ExistingHoles=${currentMetrics.holes}, Bumpiness=${currentMetrics.bumpiness}.`,
      `MISSION: Select the optimal macro-placement candidate that yields maximum line clears while minimizing hole creation and board roughness.`,
    ].join(' ');
  }

  /**
   * Advances current falling piece one step towards target macro-placement:
   * 1. Rotates toward target rotation
   * 2. Shifts laterally toward target column
   * 3. Drops vertically toward landing row
   * 4. When reached landing row, locks piece and clears lines
   */
  stepPieceTowards(target) {
    if (!this.currentPiece || !target || this.isGameOver) {
      return { locked: false, linesCleared: 0 };
    }

    // 1. Rotation alignment (incremental 90° turn per tick for visual smoothness)
    if (this.currentPiece.rotation !== target.rotation) {
      const shapes = TETROMINOES[this.currentPiece.type].shapes;
      this.currentPiece.rotation = (this.currentPiece.rotation + 1) % shapes.length;
      this.currentPiece.shape = shapes[this.currentPiece.rotation];
      // Keep inside board boundaries
      const shapeWidth = this.currentPiece.shape[0].length;
      if (this.currentPiece.x + shapeWidth > this.width) {
        this.currentPiece.x = Math.max(0, this.width - shapeWidth);
      }
      return { locked: false, linesCleared: 0, moved: true };
    }

    // 2. Lateral column alignment
    if (this.currentPiece.x < target.x) {
      this.currentPiece.x++;
      return { locked: false, linesCleared: 0, moved: true };
    } else if (this.currentPiece.x > target.x) {
      this.currentPiece.x--;
      return { locked: false, linesCleared: 0, moved: true };
    }

    // 3. Vertical descent towards landing row
    if (this.currentPiece.y < target.dropY) {
      this.currentPiece.y++;
      return { locked: false, linesCleared: 0, moved: true };
    }

    // 4. Reached landing position -> lock piece and clear lines
    const linesCleared = this.lockCurrentPieceAndClear();
    return { locked: true, linesCleared, moved: true };
  }

  lockCurrentPieceAndClear() {
    if (!this.currentPiece) return 0;
    const shape = this.currentPiece.shape;
    const posX = this.currentPiece.x;
    const posY = this.currentPiece.y;
    const type = this.currentPiece.type;

    // Lock piece into board
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] !== 0) {
          const gy = posY + r;
          const gx = posX + c;
          if (gy >= 0 && gy < this.height && gx >= 0 && gx < this.width) {
            this.grid[gy][gx] = type;
          }
        }
      }
    }

    // Clear completed lines
    let linesClearedNow = 0;
    for (let r = this.height - 1; r >= 0; r--) {
      if (this.grid[r].every((cell) => cell !== 0)) {
        this.grid.splice(r, 1);
        this.grid.unshift(Array(this.width).fill(0));
        linesClearedNow++;
        r++; // Recheck same row index after shift
      }
    }

    if (linesClearedNow > 0) {
      this.linesCleared += linesClearedNow;
      if (linesClearedNow === 4) this.tetrisCount++;

      // Classic scoring
      const lineScores = [0, 100, 300, 500, 800];
      this.score += (lineScores[linesClearedNow] || 100) * this.level;
      this.level = Math.floor(this.linesCleared / 10) + 1;
    } else {
      this.score += 10; // Placement points
    }

    // Spawn next piece
    this.spawnNextPiece();

    return linesClearedNow;
  }

  /**
   * Legacy instant application of placement
   */
  applyPlacement(placement) {
    if (!placement || this.isGameOver) return 0;
    this.currentPiece.x = placement.x;
    this.currentPiece.y = placement.dropY;
    this.currentPiece.rotation = placement.rotation;
    this.currentPiece.shape = placement.shape;
    return this.lockCurrentPieceAndClear();
  }
}
