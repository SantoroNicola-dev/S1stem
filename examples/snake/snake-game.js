/**
 * Autonomous Snake Agent Environment
 * ==================================
 * Spatial intelligence with hierarchical constraints:
 * 1. Survive: Strictly avoid fatal collisions (walls, body segments).
 * 2. Prevent Traps: Maintain continuous escape space (Flood-Fill >= snake length).
 * 3. Collect Target: Head towards food / bonus rubies to score points.
 * 4. Minimal Steps: Choose the shortest geodesic BFS path to avoid detours.
 */

export const SNAKE_ACTIONS = [
  {
    id: 'MOVE_UP',
    label: 'Move UP (North)',
    description: 'Advance north towards target food. Best when food is north and corridor has open space.',
  },
  {
    id: 'MOVE_RIGHT',
    label: 'Move RIGHT (East)',
    description: 'Advance east towards target food. Best when food is east and corridor has open space.',
  },
  {
    id: 'MOVE_DOWN',
    label: 'Move DOWN (South)',
    description: 'Advance south towards target food. Best when food is south and corridor has open space.',
  },
  {
    id: 'MOVE_LEFT',
    label: 'Move LEFT (West)',
    description: 'Advance west towards target food. Best when food is west and corridor has open space.',
  },
];

export const HIERARCHICAL_GOALS = [
  {
    id: 'goal_survival',
    label: '1. Survive',
    priority: 1,
    description: 'Strictly avoid fatal collisions with perimeter walls and own body segments.',
  },
  {
    id: 'goal_trap_avoidance',
    label: '2. Prevent Traps',
    priority: 2,
    description: 'Preserve continuous escape space (flood-fill >= snake length) avoiding dead-end pockets.',
  },
  {
    id: 'goal_collect_food',
    label: '3. Collect Target',
    priority: 3,
    description: 'Navigate directly towards the target apple to score points and fulfill the mission.',
  },
  {
    id: 'goal_shortest_path',
    label: '4. Minimal Steps',
    priority: 4,
    description: 'Take the shortest geodesic BFS corridor to reach the target in the fewest moves possible.',
  },
];

export class SnakeGame {
  constructor(gridSize = 16, targetFoods = 20) {
    this.gridSize = gridSize;
    this.targetFoods = targetFoods;
    this.snake = [];
    this.food = { x: 0, y: 0, isBonus: false, points: 25 };
    this.direction = 'RIGHT';
    this.score = 0;
    this.highScore = 0;
    this.movesCount = 0;
    this.foodsEaten = 0;
    this.stepsSinceLastFood = 0;
    this.isGameOver = false;
    this.isGameWon = false;
    this.statusMessage = 'Ready. Initializing S1stem decision loop.';
    this.floatingPoints = [];
    this.reset();
  }

  reset() {
    const center = Math.floor(this.gridSize / 2);
    this.snake = [
      { x: center, y: center },
      { x: center - 1, y: center },
      { x: center - 2, y: center },
    ];
    this.direction = 'RIGHT';
    this.score = 0;
    this.movesCount = 0;
    this.foodsEaten = 0;
    this.stepsSinceLastFood = 0;
    this.isGameOver = false;
    this.isGameWon = false;
    this.floatingPoints = [];
    this.statusMessage = 'Agent active. Evaluating spatial corridors.';
    this.spawnFood();
  }

  spawnFood() {
    const emptyCells = [];
    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        if (!this.snake.some((s) => s.x === x && s.y === y)) {
          emptyCells.push({ x, y });
        }
      }
    }

    if (emptyCells.length === 0) {
      this.isGameWon = true;
      this.statusMessage = 'Victory! Grid fully cleared.';
      return;
    }

    const randomIndex = Math.floor(Math.random() * emptyCells.length);
    const cell = emptyCells[randomIndex];
    const isBonus = Math.random() < 0.22 && this.foodsEaten > 0;

    this.food = {
      x: cell.x,
      y: cell.y,
      isBonus,
      points: isBonus ? 50 : 25,
    };
  }

  isCellDangerous(point, excludeTail = true) {
    // Wall bounds check
    if (point.x < 0 || point.x >= this.gridSize || point.y < 0 || point.y >= this.gridSize) {
      return true;
    }

    // Body collision check
    const bodyToCheck = excludeTail ? this.snake.slice(0, -1) : this.snake;
    return bodyToCheck.some((segment) => segment.x === point.x && segment.y === point.y);
  }

  getOppositeDirection(dir) {
    const map = { UP: 'DOWN', DOWN: 'UP', LEFT: 'RIGHT', RIGHT: 'LEFT' };
    return map[dir] || 'NONE';
  }

  /**
   * Breadth-First Search (BFS) to compute exact geodesic step distance to food
   */
  getBfsDistanceToFood(start) {
    if (this.isCellDangerous(start)) return 999;
    if (start.x === this.food.x && start.y === this.food.y) return 0;

    const queue = [{ p: start, dist: 0 }];
    const visited = new Set();
    visited.add(`${start.x},${start.y}`);

    while (queue.length > 0) {
      const current = queue.shift();
      if (current.p.x === this.food.x && current.p.y === this.food.y) {
        return current.dist;
      }

      const neighbors = [
        { x: current.p.x, y: current.p.y - 1 },
        { x: current.p.x + 1, y: current.p.y },
        { x: current.p.x, y: current.p.y + 1 },
        { x: current.p.x - 1, y: current.p.y },
      ];

      for (const n of neighbors) {
        const key = `${n.x},${n.y}`;
        if (!visited.has(key) && !this.isCellDangerous(n)) {
          visited.add(key);
          queue.push({ p: n, dist: current.dist + 1 });
        }
      }
    }

    return 999;
  }

  /**
   * Flood-Fill to compute count of open reachable cells from a starting point
   */
  getReachableSpace(start) {
    if (this.isCellDangerous(start)) return 0;

    const queue = [start];
    const visited = new Set();
    visited.add(`${start.x},${start.y}`);

    while (queue.length > 0) {
      const current = queue.shift();
      const neighbors = [
        { x: current.x, y: current.y - 1 },
        { x: current.x + 1, y: current.y },
        { x: current.x, y: current.y + 1 },
        { x: current.x - 1, y: current.y },
      ];

      for (const n of neighbors) {
        const key = `${n.x},${n.y}`;
        if (!visited.has(key) && !this.isCellDangerous(n)) {
          visited.add(key);
          queue.push(n);
        }
      }
    }

    return visited.size;
  }

  evaluateDirections() {
    const head = this.snake[0];
    const directions = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
    const opposite = this.getOppositeDirection(this.direction);

    const evals = directions.map((dir) => {
      if (dir === opposite) {
        return {
          dir,
          isSafe: false,
          distanceToFood: 999,
          freeSpace: 0,
          isDeadEnd: true,
          isOptimalShortest: false,
        };
      }

      let nx = head.x;
      let ny = head.y;
      if (dir === 'UP') ny--;
      if (dir === 'DOWN') ny++;
      if (dir === 'LEFT') nx--;
      if (dir === 'RIGHT') nx++;
      const nextPoint = { x: nx, y: ny };

      const isSafe = !this.isCellDangerous(nextPoint);
      if (!isSafe) {
        return {
          dir,
          isSafe: false,
          distanceToFood: 999,
          freeSpace: 0,
          isDeadEnd: true,
          isOptimalShortest: false,
        };
      }

      const distanceToFood = this.getBfsDistanceToFood(nextPoint);
      const freeSpace = this.getReachableSpace(nextPoint);
      const isDeadEnd = freeSpace < this.snake.length + 2;

      return {
        dir,
        isSafe: true,
        distanceToFood,
        freeSpace,
        isDeadEnd,
        isOptimalShortest: false,
      };
    });

    const safeViable = evals.filter((e) => e.isSafe && !e.isDeadEnd);
    const pool = safeViable.length > 0 ? safeViable : evals.filter((e) => e.isSafe);

    let minDistance = 999;
    if (pool.length > 0) {
      minDistance = Math.min(...pool.map((e) => e.distanceToFood));
    }

    return evals.map((e) => ({
      ...e,
      isOptimalShortest: e.isSafe && e.distanceToFood === minDistance && minDistance < 999,
    }));
  }

  generateStatePrompt() {
    const head = this.snake[0];
    const dx = this.food.x - head.x;
    const dy = this.food.y - head.y;

    const relX = dx > 0 ? `East (+${dx})` : dx < 0 ? `West (${dx})` : 'Aligned X';
    const relY = dy > 0 ? `South (+${dy})` : dy < 0 ? `North (${dy})` : 'Aligned Y';

    const evals = this.evaluateDirections();
    const optimalEval = evals.find((e) => e.isOptimalShortest);

    const summaries = evals.map((e) => {
      if (!e.isSafe) return `${e.dir}: [VIOLATES 1.SURVIVE] Collision danger`;
      if (e.isDeadEnd) return `${e.dir}: [VIOLATES 2.PREVENT TRAPS] Pocket trapped (${e.freeSpace} cells)`;
      const optTag = e.isOptimalShortest ? ' -> [OPTIMAL SHORTEST]' : '';
      return `${e.dir}: [VALID] Dist: ${e.distanceToFood} steps, Space: ${e.freeSpace} cells${optTag}`;
    });

    const foodLabel = this.food.isBonus ? 'SUPER RUBY (+50 pts)' : 'Apple (+25 pts)';
    const optimalHint = optimalEval
      ? `OPTIMAL GEODESIC MOVE: '${optimalEval.dir}' provides shortest path of ${optimalEval.distanceToFood} steps without detour.`
      : 'Select the safest open corridor.';

    return [
      `HIERARCHICAL GOALS (in strict order of importance):`,
      `1. [Survive] Avoid collisions.`,
      `2. [Prevent Traps] Maintain viable escape space (>= ${this.snake.length} cells).`,
      `3. [Collect Target] Reach ${foodLabel} at (${this.food.x}, ${this.food.y}) to score points.`,
      `4. [Minimal Steps] Minimize step count by choosing the geodesic shortest BFS path.`,
      `STATE: Head at (${head.x}, ${head.y}), heading ${this.direction}. Target offset: X=${relX}, Y=${relY}.`,
      `CANDIDATE CORRIDORS: ${summaries.join(' | ')}.`,
      `${optimalHint}`,
      `DECISION: Output the optimal direction strictly adhering to Priority 1 > Priority 2 > Priority 3 > Priority 4.`,
    ].join(' ');
  }

  step(dir) {
    if (this.isGameOver || this.isGameWon) {
      return { isFoodEaten: false, isGameOver: this.isGameOver, isGameWon: this.isGameWon };
    }

    if (dir !== this.getOppositeDirection(this.direction)) {
      this.direction = dir;
    }

    const head = this.snake[0];
    let nextX = head.x;
    let nextY = head.y;

    if (this.direction === 'UP') nextY--;
    if (this.direction === 'DOWN') nextY++;
    if (this.direction === 'LEFT') nextX--;
    if (this.direction === 'RIGHT') nextX++;

    const newHead = { x: nextX, y: nextY };

    if (this.isCellDangerous(newHead)) {
      this.isGameOver = true;
      this.statusMessage = 'Game Over: Collision detected.';
      return { isFoodEaten: false, isGameOver: true, isGameWon: false };
    }

    this.snake.unshift(newHead);
    this.movesCount++;
    this.stepsSinceLastFood++;

    // Fade floating score texts
    this.floatingPoints = this.floatingPoints
      .map((f) => ({ ...f, y: f.y - 0.3, opacity: f.opacity - 0.04 }))
      .filter((f) => f.opacity > 0);

    let isFoodEaten = false;
    if (newHead.x === this.food.x && newHead.y === this.food.y) {
      isFoodEaten = true;
      this.foodsEaten++;
      const earned = this.food.points;
      this.score += earned;
      if (this.score > this.highScore) this.highScore = this.score;

      this.floatingPoints.push({
        x: newHead.x,
        y: newHead.y,
        text: `+${earned}`,
        opacity: 1.0,
      });

      this.stepsSinceLastFood = 0;
      this.statusMessage = `Target captured! Score: ${this.score}`;
      this.spawnFood();
    } else {
      this.snake.pop();
    }

    return {
      isFoodEaten,
      isGameOver: this.isGameOver,
      isGameWon: this.isGameWon,
    };
  }
}
