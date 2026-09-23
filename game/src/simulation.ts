/** Deterministic 20 Hz game rules, shared by Node benchmarks and the browser. */
export const ACTIONS = ['attack', 'heal', 'resupply', 'collect', 'extract'] as const;
export type Action = typeof ACTIONS[number];
export type Controller = 'manual' | 'rule' | 'random' | 'laya' | 'decoder' | 'jev';
export const DT = 1 / 20;
export const ACTION_SECONDS = 2;
export const MAX_SECONDS = 120;
export const GRID_SIZE = 23;
export const TILE = 2.4;
export type Point = { x: number; z: number };
export type Enemy = Point & { id: number; hp: number; cooldown: number; hitAt: number; alive: boolean };
export type Pickup = Point & { id: string; kind: 'med' | 'ammo' | 'core'; taken: boolean };
export type Projectile = Point & { id: number; vx: number; vz: number; ttl: number };
export type GameEvent = Point & { time: number; type: string; value?: number; target?: number };
export type Input = { forward: number; strafe: number; turn: number; fire: boolean };
export type Status = 'playing' | 'won' | 'dead' | 'timeout';
export type Observation = {
  health: number; ammo: number; enemies: number; visibleEnemies: number;
  nearestEnemy: number | null; medkits: number; ammoCrates: number;
  carryingCore: boolean; coreAvailable: boolean; time: number;
  text: string;
};

export function rng(seed: number): () => number {
  let n = seed >>> 0;
  return () => {
    n += 0x6D2B79F5;
    let t = Math.imul(n ^ n >>> 15, n | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
export const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.z - b.z);
export const angleDelta = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
export function makeMap(): number[][] {
  const cells: number[][] = Array.from({ length: GRID_SIZE }, (_, z) =>
    Array.from({ length: GRID_SIZE }, (_, x) => (x === 0 || z === 0 || x === GRID_SIZE - 1 || z === GRID_SIZE - 1) ? 1 : 0));
  for (const x of [8, 14]) for (let z = 1; z < GRID_SIZE - 1; z++)
    if (![4, 5, 10, 11, 17, 18, 20].includes(z)) cells[z][x] = 1;
  for (const z of [8, 14]) for (let x = 1; x < GRID_SIZE - 1; x++)
    if (![3, 4, 9, 10, 11, 12, 13, 18, 19].includes(x)) cells[z][x] = 1;
  for (const [x, z] of [[4, 4], [18, 4], [4, 18], [18, 18]]) cells[z][x] = 2;
  return cells;
}

export function describeState(o: Omit<Observation, 'text'>): string {
  const health = o.health < 35 ? 'critically low' : o.health < 65 ? 'wounded' : 'healthy';
  const ammo = o.ammo <= 2 ? 'critically low or empty' : 'sufficient';
  return `Health: ${o.health}/100 (${health}). Ammunition: ${o.ammo} rounds (${ammo}). ` +
    `Hostiles remaining: ${o.enemies}; visible: ${o.visibleEnemies}. ` +
    `Medkits remaining: ${o.medkits}. Ammo crates remaining: ${o.ammoCrates}. ` +
    (o.carryingCore ? 'The reactor core is in your inventory. The exit is open.' :
      o.coreAvailable ? 'All hostiles are defeated. The reactor core is ready for collection. The exit requires the core.' :
        'The reactor core is sealed until all hostiles are defeated. The exit requires the core.');
}

/** Reference policy for the explicitly stated tactical rules, not an RL oracle. */
export function ruleAction(o: Observation): Action {
  if (o.health < 35 && o.medkits > 0) return 'heal';
  if (o.enemies > 0 && o.ammo <= 2 && o.ammoCrates > 0) return 'resupply';
  if (o.enemies > 0) return 'attack';
  if (!o.carryingCore) return 'collect';
  return 'extract';
}

export class World {
  readonly map = makeMap();
  readonly random: () => number;
  time = 0;
  ticks = 0;
  status: Status = 'playing';
  player = { x: 11.5, z: 20.5, yaw: 0, hp: 100, ammo: 10, kills: 0,
    shots: 0, hits: 0, damageTaken: 0, core: false, cooldown: 0, lastShot: -10, lastHurt: -10 };
  enemies: Enemy[] = [];
  pickups: Pickup[] = [];
  projectiles: Projectile[] = [];
  events: GameEvent[] = [];
  exit: Point = { x: 11.5, z: 21.2 };
  projectileId = 0;
  currentAction: Action | null = null;

  constructor(public readonly seed = 42) {
    this.random = rng(seed);
    this.player.hp = 72 + Math.floor(this.random() * 20);
    this.player.ammo = 8 + Math.floor(this.random() * 4);
    const spawns = [[11.5, 16.5], [5.5, 17.5], [18.5, 16.5], [11.5, 10.5], [5.5, 5.5], [18.5, 5.5]];
    this.enemies = spawns.map(([x, z], id) => ({ id, x: x + (this.random() - 0.5) * 0.6,
      z: z + (this.random() - 0.5) * 0.6, hp: 90, cooldown: 0.7 + this.random(), hitAt: -10, alive: true }));
    this.pickups = [
      { id: 'med-1', kind: 'med', x: 10.5, z: 19.5, taken: false },
      { id: 'med-2', kind: 'med', x: 17.5, z: 10.5, taken: false },
      { id: 'med-3', kind: 'med', x: 5.5, z: 3.5, taken: false },
      { id: 'ammo-1', kind: 'ammo', x: 12.5, z: 18.5, taken: false },
      { id: 'ammo-2', kind: 'ammo', x: 4.5, z: 10.5, taken: false },
      { id: 'ammo-3', kind: 'ammo', x: 18.5, z: 3.5, taken: false },
      { id: 'core', kind: 'core', x: 11.5, z: 2.5, taken: false },
    ];
  }

  wall(x: number, z: number): boolean {
    return (this.map[Math.floor(z)]?.[Math.floor(x)] ?? 1) !== 0;
  }
  free(x: number, z: number, radius = 0.2): boolean {
    return !this.wall(x - radius, z - radius) && !this.wall(x + radius, z - radius) &&
      !this.wall(x - radius, z + radius) && !this.wall(x + radius, z + radius);
  }
  move(entity: Point, dx: number, dz: number, radius = 0.2) {
    if (this.free(entity.x + dx, entity.z, radius)) entity.x += dx;
    if (this.free(entity.x, entity.z + dz, radius)) entity.z += dz;
  }
  lineOfSight(a: Point, b: Point): boolean {
    const n = Math.max(1, Math.ceil(distance(a, b) * 12));
    for (let i = 1; i <= n; i++) if (this.wall(a.x + (b.x - a.x) * i / n, a.z + (b.z - a.z) * i / n)) return false;
    return true;
  }

  /** Grid BFS motor planner; every controller has exactly this same executor. */
  waypoint(target: Point): Point {
    if (this.lineOfSight(this.player, target)) return target;
    const sx = Math.floor(this.player.x), sz = Math.floor(this.player.z);
    const tx = Math.floor(target.x), tz = Math.floor(target.z);
    const key = (x: number, z: number) => z * GRID_SIZE + x;
    const begin = key(sx, sz), end = key(tx, tz);
    const queue = [begin], parent = new Map<number, number>([[begin, -1]]);
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const n = queue[cursor];
      if (n === end) break;
      const x = n % GRID_SIZE, z = Math.floor(n / GRID_SIZE);
      for (const [dx, dz] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {
        const xx = x + dx, zz = z + dz;
        if (xx < 0 || xx >= GRID_SIZE || zz < 0 || zz >= GRID_SIZE || this.map[zz][xx]) continue;
        const next = key(xx, zz);
        if (!parent.has(next)) { parent.set(next, n); queue.push(next); }
      }
    }
    if (!parent.has(end)) return this.player;
    let at = end;
    while (parent.get(at) !== begin && parent.get(at) !== -1) at = parent.get(at)!;
    return { x: at % GRID_SIZE + 0.5, z: Math.floor(at / GRID_SIZE) + 0.5 };
  }

  closest<T extends Point>(items: T[]): T | undefined {
    return items.reduce<T | undefined>((best, item) => !best || distance(item, this.player) < distance(best, this.player) ? item : best, undefined);
  }
  motor(action: Action): Input {
    this.currentAction = action;
    let target: Point | undefined;
    let fire = false;
    const p = this.player;
    if (action === 'attack') {
      const live = this.enemies.filter(e => e.alive);
      const enemy = this.closest(live.filter(e => this.lineOfSight(p, e))) ?? this.closest(live);
      target = enemy;
      if (enemy && this.lineOfSight(p, enemy) && distance(p, enemy) < 7) {
        p.yaw = Math.atan2(enemy.x - p.x, -(enemy.z - p.z));
        fire = true;
        // Shared strafing/aiming motor, independent of the tactical decision model.
        return { forward: distance(p, enemy) > 3 ? 0.45 : 0, strafe: Math.sin(this.time * 2.1) * 0.48, turn: 0, fire };
      }
    } else if (action === 'heal') target = this.closest(this.pickups.filter(i => i.kind === 'med' && !i.taken));
    else if (action === 'resupply') target = this.closest(this.pickups.filter(i => i.kind === 'ammo' && !i.taken));
    else if (action === 'collect') target = this.pickups.find(i => i.kind === 'core' && !i.taken);
    else target = this.exit;
    if (!target) return { forward: 0, strafe: 0, turn: 0, fire: false };
    const next = this.waypoint(target);
    const d = distance(p, next);
    if (d > 0.08) p.yaw = Math.atan2(next.x - p.x, -(next.z - p.z));
    return { forward: d > 0.08 ? 1 : 0, strafe: 0, turn: 0, fire };
  }

  shoot() {
    const p = this.player;
    if (p.cooldown > 0 || p.ammo <= 0) return;
    p.ammo--; p.shots++; p.cooldown = 0.3; p.lastShot = this.time;
    const targets = this.enemies.filter(e => e.alive && distance(e, p) < 10 && this.lineOfSight(p, e) &&
      Math.abs(angleDelta(Math.atan2(e.x - p.x, -(e.z - p.z)), p.yaw)) < Math.atan2(0.42, distance(e, p)))
      .sort((a, b) => distance(a, p) - distance(b, p));
    const hit = targets[0];
    this.events.push({ type: 'shot', time: this.time, x: p.x, z: p.z, target: hit?.id });
    if (hit) {
      p.hits++; hit.hp -= 34; hit.hitAt = this.time;
      if (hit.hp <= 0) { hit.alive = false; p.kills++; this.events.push({ type: 'kill', time: this.time, x: hit.x, z: hit.z, target: hit.id }); }
    }
  }

  step(input: Input, dt = DT) {
    if (this.status !== 'playing') return;
    this.ticks++; this.time = this.ticks * DT;
    const p = this.player;
    p.cooldown = Math.max(0, p.cooldown - dt);
    p.yaw += input.turn;
    const norm = Math.max(1, Math.hypot(input.forward, input.strafe));
    const speed = 2.65 * dt / norm;
    this.move(p, (Math.sin(p.yaw) * input.forward + Math.cos(p.yaw) * input.strafe) * speed,
      (-Math.cos(p.yaw) * input.forward + Math.sin(p.yaw) * input.strafe) * speed);
    if (input.fire) this.shoot();
    for (const enemy of this.enemies) {
      if (!enemy.alive) continue;
      enemy.cooldown -= dt;
      const d = distance(enemy, p);
      if (d < 8 && this.lineOfSight(enemy, p)) {
        if (d > 2.2) this.move(enemy, (p.x - enemy.x) / d * 0.36 * dt, (p.z - enemy.z) / d * 0.36 * dt, 0.25);
        if (enemy.cooldown <= 0) {
          const aim = Math.atan2(p.z - enemy.z, p.x - enemy.x);
          this.projectiles.push({ id: this.projectileId++, x: enemy.x, z: enemy.z,
            vx: Math.cos(aim) * 4.0, vz: Math.sin(aim) * 4.0, ttl: 3 });
          enemy.cooldown = 1.4 + this.random() * 0.5;
        }
      }
    }
    this.projectiles = this.projectiles.filter(b => {
      b.x += b.vx * dt; b.z += b.vz * dt; b.ttl -= dt;
      if (b.ttl <= 0 || this.wall(b.x, b.z)) return false;
      if (distance(b, p) < 0.32) {
        p.hp = Math.max(0, p.hp - 9); p.damageTaken += 9; p.lastHurt = this.time;
        this.events.push({ type: 'hurt', time: this.time, x: p.x, z: p.z, value: 9 });
        return false;
      }
      return true;
    });
    for (const item of this.pickups) {
      if (item.taken || distance(item, p) > 0.58) continue;
      if (item.kind === 'med' && p.hp < 100) { p.hp = Math.min(100, p.hp + 55); item.taken = true; }
      if (item.kind === 'ammo') { p.ammo += 15; item.taken = true; }
      if (item.kind === 'core' && this.enemies.every(e => !e.alive)) { p.core = true; item.taken = true; }
      if (item.taken) this.events.push({ type: item.kind, time: this.time, x: item.x, z: item.z });
    }
    if (p.hp <= 0) this.status = 'dead';
    else if (p.core && distance(p, this.exit) < 0.7) this.status = 'won';
    else if (this.time >= MAX_SECONDS) this.status = 'timeout';
    if (this.status !== 'playing') this.events.push({ type: this.status, time: this.time, x: p.x, z: p.z });
  }

  execute(action: Action, seconds = ACTION_SECONDS) {
    for (let i = 0; i < Math.round(seconds / DT) && this.status === 'playing'; i++) this.step(this.motor(action));
  }
  observe(): Observation {
    const live = this.enemies.filter(e => e.alive);
    const o = { health: this.player.hp, ammo: this.player.ammo, enemies: live.length,
      visibleEnemies: live.filter(e => this.lineOfSight(this.player, e)).length,
      nearestEnemy: live.length ? Math.round(Math.min(...live.map(e => distance(e, this.player))) * 10) / 10 : null,
      medkits: this.pickups.filter(p => p.kind === 'med' && !p.taken).length,
      ammoCrates: this.pickups.filter(p => p.kind === 'ammo' && !p.taken).length,
      carryingCore: this.player.core, coreAvailable: live.length === 0 && !this.player.core,
      time: Math.round(this.time * 100) / 100 };
    return { ...o, text: describeState(o) };
  }
  snapshot() {
    return { seed: this.seed, ticks: this.ticks, time: this.time, status: this.status,
      player: { ...this.player }, enemies: this.enemies.map(e => ({ ...e })),
      pickups: this.pickups.map(p => ({ ...p })), projectiles: this.projectiles.map(p => ({ ...p })) };
  }
}
