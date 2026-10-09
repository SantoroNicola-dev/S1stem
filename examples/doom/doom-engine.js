/**
 * DOOM WebAssembly Engine Wrapper for S1stem Edge Reflex Integration.
 * Manages doom.wasm instantiation, fast 32-bit pixel rendering, key dispatching,
 * and high-speed visual perceptual extraction for real-time AI decision loops.
 */

export class DoomEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.instance = null;
    this.memory = null;
    this.width = 640;
    this.height = 400;
    this.frameBufferPtr = null;
    this.imageData = null;
    this.isRunning = false;
    this.tickInterval = null;
    this.fps = 0;
    this.frameCount = 0;
    this.lastFpsUpdate = performance.now();
    this.activeKeys = new Set();
    this.keyCodes = {};

    // Visual perception cached metrics
    this.perception = {
      sectorLum: { left: 0.5, center: 0.5, right: 0.5 },
      sectorThreat: { left: 0, center: 0, right: 0 },
      corridorDepth: { left: 0.5, center: 0.5, right: 0.5 },
      wallProximity: 0.2,
      centerTargetScore: 0,
      averageLuminance: 0.5,
      dangerDetected: false,
      motionDelta: 0,
      isStuck: false
    };
    this.prevLumGrid = null;
    this.stuckCounter = 0;

    // Logging listeners
    this.onInfo = null;
    this.onError = null;
  }

  async init(wasmUrl = './assets/doom.wasm') {
    const imports = {
      loading: {
        onGameInit: (w, h) => {
          this.width = w;
          this.height = h;
          this.canvas.width = w;
          this.canvas.height = h;
          this.imageData = this.ctx.createImageData(w, h);
          if (this.onInfo) this.onInfo(`[ENGINE] Initialized resolution: ${w}x${h}`);
        },
        wadSizes: () => {},
        readWads: () => {},
      },
      ui: {
        drawFrame: (bufferPtr) => {
          this.frameBufferPtr = bufferPtr;
          this.renderFrame();
        },
      },
      runtimeControl: {
        timeInMilliseconds: () => BigInt(Math.trunc(performance.now())),
      },
      console: {
        onInfoMessage: (ptr, len) => {
          if (this.onInfo && this.memory) {
            const msg = this.readMemoryString(ptr, len);
            this.onInfo(msg);
          }
        },
        onErrorMessage: (ptr, len) => {
          if (this.onError && this.memory) {
            const msg = this.readMemoryString(ptr, len);
            this.onError(msg);
          }
        },
      },
      gameSaving: {
        sizeOfSaveGame: () => 0,
        readSaveGame: () => 0,
        writeSaveGame: () => 0,
      }
    };

    try {
      const response = await fetch(wasmUrl);
      const wasmBinary = await response.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(wasmBinary, imports);
      this.instance = instance;
      this.memory = instance.exports.memory;

      // Extract key code constants exported by doom.wasm
      const exp = instance.exports;
      this.keyCodes = {
        LEFT: exp.KEY_LEFTARROW?.value ?? exp.KEY_LEFTARROW ?? 172,
        RIGHT: exp.KEY_RIGHTARROW?.value ?? exp.KEY_RIGHTARROW ?? 174,
        UP: exp.KEY_UPARROW?.value ?? exp.KEY_UPARROW ?? 173,
        DOWN: exp.KEY_DOWNARROW?.value ?? exp.KEY_DOWNARROW ?? 175,
        STRAFE_L: exp.KEY_STRAFE_L?.value ?? exp.KEY_STRAFE_L ?? 160,
        STRAFE_R: exp.KEY_STRAFE_R?.value ?? exp.KEY_STRAFE_R ?? 161,
        FIRE: exp.KEY_FIRE?.value ?? exp.KEY_FIRE ?? 163,
        USE: exp.KEY_USE?.value ?? exp.KEY_USE ?? 162,
        SHIFT: exp.KEY_SHIFT?.value ?? exp.KEY_SHIFT ?? 182,
        TAB: exp.KEY_TAB?.value ?? exp.KEY_TAB ?? 9,
        ESCAPE: exp.KEY_ESCAPE?.value ?? exp.KEY_ESCAPE ?? 27,
        ENTER: exp.KEY_ENTER?.value ?? exp.KEY_ENTER ?? 13,
        BACKSPACE: exp.KEY_BACKSPACE?.value ?? exp.KEY_BACKSPACE ?? 127,
        ALT: exp.KEY_ALT?.value ?? exp.KEY_ALT ?? 184,
      };

      // Initialize Doom generic core
      this.instance.exports.initGame();
      return true;
    } catch (err) {
      console.error('Doom WASM initialization failed:', err);
      throw err;
    }
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    // Classic Doom tick rate: 35 ticks/sec (~28.57ms)
    this.tickInterval = setInterval(() => {
      if (this.instance && this.isRunning) {
        this.instance.exports.tickGame();
      }
    }, 1000 / 35);
  }

  stop() {
    this.isRunning = false;
    if (this.tickInterval) {
      clearInterval(this.tickInterval);
      this.tickInterval = null;
    }
    this.releaseAllKeys();
  }

  renderFrame() {
    if (!this.frameBufferPtr || !this.memory || !this.imageData) return;

    const count = this.width * this.height;
    const src32 = new Uint32Array(this.memory.buffer, this.frameBufferPtr, count);
    const dst32 = new Uint32Array(this.imageData.data.buffer);

    // Blazing-fast little-endian 32-bit pixel transform (BGRA -> RGBA)
    for (let i = 0; i < count; i++) {
      const px = src32[i];
      // px is 0xAABBGGRR in little endian: byte0=B, byte1=G, byte2=R, byte3=A
      dst32[i] = 0xFF000000 | ((px & 0xFF) << 16) | (px & 0xFF00) | ((px >> 16) & 0xFF);
    }

    this.ctx.putImageData(this.imageData, 0, 0);

    // FPS calculation
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    // Extract visual perceptual cues for AI
    this.extractPerception(src32);
  }

  /**
   * Fast spatial perception sampler across viewport (excludes HUD status bar).
   * Computes sector lighting, red/orange fireball threats, and central crosshair enemy alignment.
   * Masks player weapon and muzzle flash to avoid self-reinforcing shooting loops.
   */
  extractPerception(src32) {
    const w = this.width;
    const h = this.height;
    // 3D Viewport: top 84% (approx 336 rows in 640x400)
    const viewHeight = Math.floor(h * 0.84);
    const leftWidth = Math.floor(w / 3);
    const centerWidth = Math.floor(w / 3);

    let lumLeft = 0, countLeft = 0, threatLeft = 0;
    let lumCenter = 0, countCenter = 0, threatCenter = 0;
    let lumRight = 0, countRight = 0, threatRight = 0;
    let crosshairThreat = 0, crosshairCount = 0;

    const step = 8; // Sparse sub-sampling for sub-millisecond efficiency

    // Crosshairs zone: upper-middle sector, strictly above player weapon sprite
    const chYMin = Math.floor(viewHeight * 0.22);
    const chYMax = Math.floor(viewHeight * 0.56);
    const chXMin = Math.floor(w * 0.38);
    const chXMax = Math.floor(w * 0.62);

    // Player weapon & muzzle flash zone (lower-center) to mask out
    const gunYMin = Math.floor(viewHeight * 0.58);
    const gunXMin = Math.floor(w * 0.28);
    const gunXMax = Math.floor(w * 0.72);

    let lumUpperLeft = 0, countUpperLeft = 0;
    let lumUpperCenter = 0, countUpperCenter = 0;
    let lumUpperRight = 0, countUpperRight = 0;

    for (let y = 0; y < viewHeight; y += step) {
      const rowOffset = y * w;
      const isUpper = (y < viewHeight * 0.50);

      for (let x = 0; x < w; x += step) {
        const px = src32[rowOffset + x];
        const b = px & 0xFF;
        const g = (px >> 8) & 0xFF;
        const r = (px >> 16) & 0xFF;

        const lum = (r * 0.299 + g * 0.587 + b * 0.114) / 255;

        // Mask out player's own weapon barrel and muzzle flashes
        const inGunZone = (y >= gunYMin && x >= gunXMin && x <= gunXMax);

        // Genuine threat signature (imp fireballs, demon flesh, crimson eyes/blood, toxic barrels)
        // Strictly distinguished from standard brown wall/door textures which have low (r - g)
        const isFireball = (r > 185 && g > 90 && g < 185 && b < 85 && (r - b) > 95);
        const isDemonThreat = (r > 150 && (r - g) > 55 && (r - b) > 65 && b < 100);
        const isBarrelThreat = (g > 165 && (g - r) > 50 && (g - b) > 50);
        const isThreatColor = !inGunZone && (isFireball || isDemonThreat || isBarrelThreat);
        const threatVal = isThreatColor ? 1.0 : 0.0;

        if (x < leftWidth) {
          lumLeft += lum;
          threatLeft += threatVal;
          countLeft++;
          if (isUpper) { lumUpperLeft += lum; countUpperLeft++; }
        } else if (x < leftWidth + centerWidth) {
          lumCenter += lum;
          threatCenter += threatVal;
          countCenter++;
          if (isUpper) { lumUpperCenter += lum; countUpperCenter++; }

          // Center crosshair evaluation strictly in upper-mid space
          if (x >= chXMin && x <= chXMax && y >= chYMin && y <= chYMax) {
            crosshairThreat += threatVal;
            crosshairCount++;
          }
        } else {
          lumRight += lum;
          threatRight += threatVal;
          countRight++;
          if (isUpper) { lumUpperRight += lum; countUpperRight++; }
        }
      }
    }

    // Depth openness based on Doom's diminishing distance lighting (deep corridors are darker than close walls)
    const avgUpperLeft = countUpperLeft ? lumUpperLeft / countUpperLeft : 0.3;
    const avgUpperCenter = countUpperCenter ? lumUpperCenter / countUpperCenter : 0.3;
    const avgUpperRight = countUpperRight ? lumUpperRight / countUpperRight : 0.3;

    const corridorDepth = {
      left: Math.max(0, Math.min(1.0, (0.42 - avgUpperLeft) * 3.8)),
      center: Math.max(0, Math.min(1.0, (0.42 - avgUpperCenter) * 3.8)),
      right: Math.max(0, Math.min(1.0, (0.42 - avgUpperRight) * 3.8)),
    };
    const wallProximity = Math.max(0, Math.min(1.0, (avgUpperCenter - 0.22) * 3.5));

    // Lightweight 8x8 optical flow / motion delta estimation to detect if marine is stuck
    const gridW = 8, gridH = 8;
    let frameDiff = 0;
    const currentGrid = new Float32Array(gridW * gridH);
    for (let gy = 0; gy < gridH; gy++) {
      const py = Math.floor((viewHeight * (gy + 0.5)) / gridH);
      for (let gx = 0; gx < gridW; gx++) {
        const px = Math.floor((w * (gx + 0.5)) / gridW);
        const pxi = src32[py * w + px];
        const lum = (((pxi >> 16) & 0xFF) * 0.299 + ((pxi >> 8) & 0xFF) * 0.587 + (pxi & 0xFF) * 0.114) / 255;
        const idx = gy * gridW + gx;
        currentGrid[idx] = lum;
        if (this.prevLumGrid) {
          frameDiff += Math.abs(lum - this.prevLumGrid[idx]);
        }
      }
    }
    this.prevLumGrid = currentGrid;
    const motionDelta = frameDiff / (gridW * gridH);

    // If forward key is held and scene difference is virtually zero, marine is stuck against wall/corner
    if (this.activeKeys.has(this.keyCodes?.UP ?? 173) && motionDelta < 0.012) {
      this.stuckCounter = Math.min(20, (this.stuckCounter || 0) + 1);
    } else {
      this.stuckCounter = Math.max(0, (this.stuckCounter || 0) - 1);
    }
    const isStuck = (this.stuckCounter || 0) >= 6;

    this.perception = {
      sectorLum: {
        left: countLeft ? lumLeft / countLeft : 0.5,
        center: countCenter ? lumCenter / countCenter : 0.5,
        right: countRight ? lumRight / countRight : 0.5,
      },
      sectorThreat: {
        left: countLeft ? Math.min(1.0, (threatLeft / countLeft) * 14.0) : 0,
        center: countCenter ? Math.min(1.0, (threatCenter / countCenter) * 14.0) : 0,
        right: countRight ? Math.min(1.0, (threatRight / countRight) * 14.0) : 0,
      },
      corridorDepth,
      wallProximity,
      centerTargetScore: crosshairCount ? Math.min(1.0, (crosshairThreat / crosshairCount) * 16.0) : 0,
      averageLuminance: (lumLeft + lumCenter + lumRight) / (countLeft + countCenter + countRight || 1),
      dangerDetected: (threatLeft + threatCenter + threatRight) > 2,
      motionDelta,
      isStuck
    };
  }

  pressKey(name) {
    if (!this.instance) return;
    const code = this.keyCodes[name] ?? (name.length === 1 ? name.toLowerCase().charCodeAt(0) : null);
    if (code !== null && !this.activeKeys.has(code)) {
      this.activeKeys.add(code);
      this.instance.exports.reportKeyDown(code);
    }
  }

  releaseKey(name) {
    if (!this.instance) return;
    const code = this.keyCodes[name] ?? (name.length === 1 ? name.toLowerCase().charCodeAt(0) : null);
    if (code !== null && this.activeKeys.has(code)) {
      this.activeKeys.delete(code);
      this.instance.exports.reportKeyUp(code);
    }
  }

  pulseKey(name, durationMs = 120) {
    this.pressKey(name);
    setTimeout(() => {
      this.releaseKey(name);
    }, durationMs);
  }

  releaseAllKeys() {
    if (!this.instance) return;
    for (const code of this.activeKeys) {
      this.instance.exports.reportKeyUp(code);
    }
    this.activeKeys.clear();
  }

  /**
   * Automates skipping title/menus straight into E1M1 Knee-Deep in the Dead.
   * Deliberately timed to accommodate Doom's menu skull transitions.
   */
  async autoStartGame() {
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    if (this.onInfo) this.onInfo('[SYSTEM] Automated launch sequence initiating: Booting Knee-Deep in the Dead...');

    this.releaseAllKeys();

    // 1. Give Doom engine time to complete WAD parsing and title splash init
    await sleep(1500);

    // 2. Dismiss any accidental quit dialog by pressing 'N'
    this.pressKey('n');
    await sleep(80);
    this.releaseKey('n');
    await sleep(350);

    // 3. Clear any active submenu with ESCAPE
    this.pressKey('ESCAPE');
    await sleep(100);
    this.releaseKey('ESCAPE');
    await sleep(650);

    // 4. Press ENTER 1: Open Main Menu / select New Game
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(750);

    // Press ENTER 2: Confirms "New Game" -> Opens Episode Selection (cursor defaults to "Knee-Deep in the Dead")
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(750);

    // Press ENTER 3: Confirms "Knee-Deep in the Dead" -> Opens Skill Selection (defaults to "Hurt me plenty")
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(750);

    // Press ENTER 4: Confirms Skill Selection -> Level wipe and spawn in E1M1 Hangar!
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(1200);

    // Safety pulse in case title screen was in demo sequence
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(600);

    if (this.onInfo) this.onInfo('[SYSTEM] Player spawned in E1M1 Hangar! AI Reflex Core engaged.');
  }

  /**
   * Restarts the current session, navigating menus back to E1M1 Hangar
   */
  async restartGame() {
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    this.releaseAllKeys();

    // Dismiss any dialog with 'n'
    this.pressKey('n');
    await sleep(80);
    this.releaseKey('n');
    await sleep(250);

    // ESCAPE to bring up menu
    this.pressKey('ESCAPE');
    await sleep(100);
    this.releaseKey('ESCAPE');
    await sleep(600);

    // ENTER: New Game
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(700);

    // ENTER: Episode 1
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(700);

    // ENTER: Skill
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(1200);

    // Safety ENTER
    this.pressKey('ENTER');
    await sleep(140);
    this.releaseKey('ENTER');
    await sleep(500);

    if (this.onInfo) this.onInfo('[SYSTEM] Session restarted in E1M1 Hangar.');
  }

  readMemoryString(ptr, len) {
    if (!this.memory) return '';
    const bytes = new Uint8Array(this.memory.buffer, ptr, len);
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes).trim();
  }
}
