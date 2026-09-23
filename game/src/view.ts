import * as THREE from 'three';
import { World, GRID_SIZE, TILE, rng, angleDelta } from './simulation.ts';

function texture(kind: 'wall' | 'floor' | 'crate' | 'flare'): THREE.CanvasTexture {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d')!;
  const random = rng(kind.length * 312);
  if (kind === 'flare') {
    g.translate(64, 64); g.fillStyle = '#ffe6a4'; g.beginPath();
    for (let i = 0; i < 16; i++) {
      const a = Math.PI * i / 8, r = i % 2 ? 17 : 61;
      g.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    g.closePath(); g.fill();
    g.fillStyle = '#fff9e1'; g.fillRect(-12, -12, 24, 24);
  } else {
    const base = kind === 'floor' ? [54, 57, 47] : kind === 'crate' ? [109, 89, 53] : [93, 97, 80];
    const pixels = g.createImageData(128, 128);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const d = Math.floor(random() * 20) - 10;
      pixels.data[i] = base[0] + d; pixels.data[i + 1] = base[1] + d; pixels.data[i + 2] = base[2] + d; pixels.data[i + 3] = 255;
    }
    g.putImageData(pixels, 0, 0);
    g.strokeStyle = '#171e19'; g.lineWidth = 3; g.strokeRect(1, 1, 126, 126);
    if (kind === 'wall') {
      g.fillStyle = '#202a22'; g.fillRect(0, 4, 128, 8); g.fillRect(0, 101, 128, 17);
      g.fillStyle = '#b6b29a'; g.fillRect(0, 14, 128, 2); g.fillRect(0, 96, 128, 2);
      g.strokeStyle = '#4d5644'; g.lineWidth = 2; g.strokeRect(9, 25, 110, 59);
      for (const x of [7, 120]) for (const y of [20, 90]) { g.fillStyle = '#30392e'; g.fillRect(x, y, 3, 3); }
      g.fillStyle = '#444f3c'; for (let i = 0; i < 7; i++) g.fillRect(42, 37 + i * 5, 43, 2);
    } else if (kind === 'floor') {
      g.fillStyle = '#1c251c'; for (let i = 0; i < 8; i++) g.fillRect(10, 10 + i * 14, 108, 2);
      g.fillStyle = '#778069'; g.fillRect(4, 4, 2, 2); g.fillRect(122, 122, 2, 2);
    } else {
      g.fillStyle = '#242b20'; g.fillRect(0, 8, 128, 12); g.fillRect(0, 108, 128, 12);
      for (let i = 0; i < 5; i++) g.fillRect(i * 28 - 10, 24, 7, 80);
      g.fillStyle = '#b0a05f'; g.fillRect(36, 43, 54, 42); g.fillStyle = '#323d26'; g.font = 'bold 15px monospace'; g.fillText('09', 49, 70);
    }
  }
  const t = new THREE.CanvasTexture(c); t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestMipmapNearestFilter;
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

const box = new THREE.BoxGeometry(1, 1, 1);
function cube(parent: THREE.Object3D, material: THREE.Material, size: [number, number, number], position: [number, number, number]) {
  const mesh = new THREE.Mesh(box, material); mesh.scale.set(...size); mesh.position.set(...position); parent.add(mesh); return mesh;
}
const lit = (color: number, extra: object = {}) => new THREE.MeshLambertMaterial({ color, ...extra });
const unlit = (color: number, extra: object = {}) => new THREE.MeshBasicMaterial({ color, ...extra });

export class GameView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(78, innerWidth / innerHeight, 0.045, 100);
  dynamic = new THREE.Group();
  enemies: THREE.Group[] = [];
  pickups = new Map<string, THREE.Group>();
  bullets: THREE.InstancedMesh;
  sparks: THREE.InstancedMesh;
  weapon = new THREE.Group();
  muzzle: THREE.Mesh;
  portal: THREE.Mesh;
  portalMaterial = unlit(0xc06836, { transparent: true, opacity: 0.24, side: THREE.DoubleSide });
  enemyMaterials: THREE.MeshLambertMaterial[] = [];
  world: World;
  cameraYaw = 0;
  drawCalls = 0;
  triangles = 0;
  rendererName = 'unknown';
  matrixObject = new THREE.Object3D();

  constructor(canvas: HTMLCanvasElement, world: World) {
    this.world = world;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x1b2118);
    const gl = this.renderer.getContext();
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    if (info) this.rendererName = gl.getParameter(info.UNMASKED_RENDERER_WEBGL);
    this.scene.fog = new THREE.FogExp2(0x1b2118, 0.035);
    this.scene.add(new THREE.HemisphereLight(0xddd9a5, 0x404b34, 2.2));
    const key = new THREE.DirectionalLight(0xffcc8f, 1.25); key.position.set(12, 30, 18); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x7b9d80, 0.65); fill.position.set(-20, 8, -12); this.scene.add(fill);
    this.scene.add(this.camera);
    const torch = new THREE.PointLight(0xf2d2a7, 8, 11, 1.6); torch.position.set(0, 0.5, 0); this.camera.add(torch);
    this.buildStation();
    this.buildWeapon();
    this.muzzle = new THREE.Mesh(new THREE.PlaneGeometry(0.48, 0.48), unlit(0xffcb64, {
      map: texture('flare'), transparent: true, depthWrite: false, depthTest: false, fog: false, blending: THREE.AdditiveBlending }));
    this.muzzle.position.set(0.20, -0.22, -1.23); this.muzzle.renderOrder = 99; this.muzzle.visible = false; this.camera.add(this.muzzle);
    this.portal = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 2.8), this.portalMaterial);
    this.portal.position.set(11.5 * TILE, 1.55, 21.7 * TILE); this.scene.add(this.portal);
    this.bullets = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.13, 0), unlit(0xff7845), 128);
    this.bullets.count = 0; this.scene.add(this.bullets);
    this.sparks = new THREE.InstancedMesh(box, unlit(0xffb461), 256); this.sparks.count = 0; this.scene.add(this.sparks);
    this.scene.add(this.dynamic);
    this.setWorld(world);
    this.resize();
    window.addEventListener('resize', () => this.resize());
    canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); document.querySelector('#message')!.textContent = 'GRAPHICS CONTEXT LOST'; });
    canvas.addEventListener('webglcontextrestored', () => { document.querySelector('#message')!.textContent = 'GRAPHICS RESTORED'; });
  }

  resize() {
    // Intentionally low resolution: the retro art direction also works on software WebGL.
    const scale = Math.min(1, 960 / innerWidth);
    this.renderer.setSize(Math.round(innerWidth * scale), Math.round(innerHeight * scale), false);
    this.camera.aspect = innerWidth / innerHeight; this.camera.updateProjectionMatrix();
  }
  buildStation() {
    const wallMap = texture('wall'), floorMap = texture('floor'); floorMap.repeat.set(GRID_SIZE, GRID_SIZE);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(GRID_SIZE * TILE, GRID_SIZE * TILE), lit(0xa7aa91, { map: floorMap }));
    floor.rotation.x = -Math.PI / 2; floor.position.set(GRID_SIZE * TILE / 2, 0, GRID_SIZE * TILE / 2); this.scene.add(floor);
    const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(GRID_SIZE * TILE, GRID_SIZE * TILE), lit(0x6a705b, { map: floorMap }));
    ceiling.rotation.x = Math.PI / 2; ceiling.position.set(GRID_SIZE * TILE / 2, 3.6, GRID_SIZE * TILE / 2); this.scene.add(ceiling);
    const walls = new THREE.InstancedMesh(box, lit(0xced0b8, { map: wallMap }), GRID_SIZE * GRID_SIZE);
    const trim = new THREE.InstancedMesh(box, lit(0x282f24), GRID_SIZE * GRID_SIZE);
    let count = 0;
    const m = new THREE.Object3D();
    for (let z = 0; z < GRID_SIZE; z++) for (let x = 0; x < GRID_SIZE; x++) if (this.world.map[z][x]) {
      m.position.set((x + 0.5) * TILE, 1.8, (z + 0.5) * TILE); m.scale.set(TILE, 3.6, TILE); m.updateMatrix(); walls.setMatrixAt(count, m.matrix);
      m.position.y = 0.19; m.scale.set(TILE + 0.04, 0.38, TILE + 0.04); m.updateMatrix(); trim.setMatrixAt(count, m.matrix); count++;
    }
    walls.count = trim.count = count; this.scene.add(walls, trim);
    const strips = new THREE.InstancedMesh(box, unlit(0xe5b06b), 80); count = 0;
    for (let z = 2; z < 22; z += 3) for (const x of [3.5, 11.5, 18.5]) {
      m.position.set(x * TILE, 3.54, (z + 0.5) * TILE); m.scale.set(1.5, 0.035, 0.24); m.updateMatrix(); strips.setMatrixAt(count++, m.matrix);
    }
    strips.count = count; this.scene.add(strips);
    const rails = new THREE.InstancedMesh(box, unlit(0x8f8050), 80); count = 0;
    for (let z = 1; z < 22; z++) for (const x of [9.65, 13.35]) {
      m.position.set(x * TILE, 0.009, (z + 0.5) * TILE); m.scale.set(0.055, 0.015, 1.5); m.updateMatrix(); rails.setMatrixAt(count++, m.matrix);
    }
    rails.count = count; this.scene.add(rails);
    const crate = lit(0xc3b58b, { map: texture('crate') });
    // Decoration stays inside blocked pillar cells, so visual and collision geometry agree.
    for (const [x, z] of [[4.5, 4.5], [18.5, 4.5], [4.5, 18.5], [18.5, 18.5]]) {
      cube(this.scene, crate, [TILE + 0.04, 1.4, TILE + 0.04], [x * TILE, 0.7, z * TILE]);
    }
    const frame = lit(0x86917b);
    for (const x of [10.94, 12.06]) cube(this.scene, frame, [0.28, 3.1, 0.35], [x * TILE, 1.55, 21.8 * TILE]);
    cube(this.scene, frame, [3, 0.3, 0.35], [11.5 * TILE, 3, 21.8 * TILE]);
    cube(this.scene, unlit(0xa34d2c), [2.3, 0.045, 0.04], [11.5 * TILE, 2.77, 21.59 * TILE]);
    const terminal = new THREE.Group();
    cube(terminal, lit(0x343e2e), [1.2, 1.5, 0.6], [0, 0.75, 0]);
    cube(terminal, unlit(0x7c9761), [0.92, 0.5, 0.02], [0, 1.06, 0.31]);
    for (let i = 0; i < 4; i++) cube(terminal, unlit(0x283c26), [0.7 - i * 0.09, 0.027, 0.022], [-i * 0.04, 1.2 - i * 0.09, 0.33]);
    terminal.position.set(13.4 * TILE, 0, 19.1 * TILE); terminal.rotation.y = -0.5; this.scene.add(terminal);
  }
  buildWeapon() {
    const steel = lit(0x8d9180, { fog: false }), dark = lit(0x282e28, { fog: false }), copper = lit(0xa57740, { fog: false });
    this.weapon.position.set(0.18, -0.34, -0.63);
    cube(this.weapon, dark, [0.23, 0.18, 0.40], [0, 0, 0]);
    cube(this.weapon, steel, [0.20, 0.105, 0.57], [0, 0.045, -0.15]);
    cube(this.weapon, dark, [0.08, 0.07, 0.65], [0, 0.08, -0.18]);
    cube(this.weapon, copper, [0.26, 0.13, 0.24], [0, -0.048, -0.26]);
    for (let i = 0; i < 5; i++) cube(this.weapon, dark, [0.275, 0.035, 0.024], [0, -0.007, -0.35 + i * 0.038]);
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.047, 0.047, 0.68, 8), steel); barrel.rotation.x = Math.PI / 2; barrel.position.set(0, 0.035, -0.31); this.weapon.add(barrel);
    const bore = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.69, 8), dark); bore.rotation.x = Math.PI / 2; bore.position.copy(barrel.position); this.weapon.add(bore);
    cube(this.weapon, dark, [0.035, 0.065, 0.04], [0, 0.14, -0.5]);
    cube(this.weapon, unlit(0xc9d5ad, { fog: false }), [0.01, 0.023, 0.012], [0, 0.183, -0.48]);
    cube(this.weapon, dark, [0.13, 0.24, 0.14], [0.03, -0.15, 0.12]);
    cube(this.weapon, lit(0x5e6447, { fog: false }), [0.13, 0.17, 0.22], [0.05, -0.20, 0.08]);
    cube(this.weapon, lit(0x676d50, { fog: false }), [0.13, 0.12, 0.23], [-0.12, -0.13, -0.23]);
    cube(this.weapon, unlit(0x9cc27a, { fog: false }), [0.04, 0.03, 0.006], [0.068, 0.027, 0.205]);
    this.camera.add(this.weapon);
  }
  setWorld(world: World) {
    this.world = world;
    this.dynamic.clear(); this.enemies = []; this.pickups.clear(); this.enemyMaterials = [];
    this.camera.position.set(world.player.x * TILE, 1.58, world.player.z * TILE);
    this.cameraYaw = world.player.yaw;
    for (const enemy of world.enemies) {
      const g = new THREE.Group();
      const flesh = lit(0x754736), armor = lit(0x424737), bone = lit(0xbbaf80), black = lit(0x272b22);
      this.enemyMaterials.push(flesh);
      cube(g, flesh, [0.79, 0.89, 0.47], [0, 1.18, 0]);
      cube(g, armor, [0.91, 0.43, 0.60], [0, 1.5, 0]);
      cube(g, flesh, [0.44, 0.45, 0.43], [0, 1.93, 0.04]);
      cube(g, black, [0.39, 0.1, 0.035], [0, 1.97, 0.27]);
      cube(g, unlit(0xffae51), [0.30, 0.065, 0.04], [0, 1.97, 0.3]);
      cube(g, bone, [0.23, 0.09, 0.04], [0, 1.82, 0.275]);
      for (const side of [-1, 1]) {
        cube(g, flesh, [0.26, 0.63, 0.28], [side * 0.57, 1.14, 0]);
        cube(g, armor, [0.3, 0.27, 0.34], [side * 0.61, 0.91, 0.07]);
        cube(g, armor, [0.28, 0.70, 0.35], [side * 0.24, 0.47, 0]);
        cube(g, black, [0.33, 0.2, 0.5], [side * 0.24, 0.12, 0.07]);
        const horn = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.41, 4), bone);
        horn.position.set(side * 0.22, 2.28, 0.03); horn.rotation.z = side * -0.42; g.add(horn);
      }
      cube(g, unlit(0xff8f40), [0.20, 0.24, 0.055], [0, 1.38, 0.32]);
      g.position.set(enemy.x * TILE, 0, enemy.z * TILE); this.dynamic.add(g); this.enemies.push(g);
    }
    for (const item of world.pickups) {
      const g = new THREE.Group();
      if (item.kind === 'core') {
        const core = new THREE.Mesh(new THREE.OctahedronGeometry(0.5, 0), unlit(0xe0ab62)); g.add(core);
        const cage = new THREE.Mesh(new THREE.OctahedronGeometry(0.76, 0), new THREE.MeshBasicMaterial({ color: 0x87aa76, wireframe: true })); g.add(cage);
        const base = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.9, 0.18, 8), lit(0x6d735d)); base.position.y = -0.85; g.add(base);
      } else {
        const mat = lit(item.kind === 'med' ? 0xc8c2a0 : 0x8e9666);
        cube(g, mat, [0.68, 0.38, 0.49], [0, 0, 0]);
        const icon = unlit(item.kind === 'med' ? 0xc24f31 : 0xe3b669);
        if (item.kind === 'med') {
          cube(g, icon, [0.28, 0.07, 0.012], [0, 0.02, 0.253]); cube(g, icon, [0.07, 0.28, 0.013], [0, 0.02, 0.253]);
          cube(g, icon, [0.28, 0.012, 0.07], [0, 0.197, 0]); cube(g, icon, [0.07, 0.013, 0.28], [0, 0.197, 0]);
        } else for (let i = 0; i < 3; i++) cube(g, icon, [0.08, 0.24, 0.013], [-0.16 + i * 0.16, 0.02, 0.252]);
      }
      g.position.set(item.x * TILE, item.kind === 'core' ? 1.05 : 0.31, item.z * TILE); this.pickups.set(item.id, g); this.dynamic.add(g);
    }
  }

  render(dt: number, menu = false) {
    const w = this.world, p = w.player;
    const factor = 1 - Math.exp(-15 * dt);
    this.camera.position.x += (p.x * TILE - this.camera.position.x) * factor;
    this.camera.position.z += (p.z * TILE - this.camera.position.z) * factor;
    this.cameraYaw += angleDelta(p.yaw, this.cameraYaw) * factor;
    this.camera.rotation.set(menu ? -0.035 : 0, -this.cameraYaw + (menu ? -0.17 : 0), 0, 'YXZ');
    const recoil = Math.max(0, 1 - (w.time - p.lastShot) / 0.19);
    this.weapon.visible = !menu;
    this.weapon.position.y = -0.34 + Math.sin(w.time * 8) * 0.006 - recoil * 0.05;
    this.weapon.position.z = -0.63 + recoil * 0.15;
    this.weapon.rotation.x = recoil * 0.11;
    this.muzzle.visible = !menu && w.time - p.lastShot < 0.09;
    this.muzzle.rotation.z = p.shots * 2.5;
    this.portalMaterial.color.setHex(p.core ? 0xa5d885 : 0xc06836);
    this.portalMaterial.opacity = p.core ? 0.45 + Math.sin(w.time * 3) * 0.12 : 0.15;
    w.enemies.forEach((e, i) => {
      const g = this.enemies[i];
      g.position.set(e.x * TILE, e.alive ? Math.sin(w.time * 5 + i) * 0.02 : 0.15, e.z * TILE);
      g.rotation.y = Math.atan2(p.x - e.x, p.z - e.z);
      g.scale.y = e.alive ? 1 : 0.13;
      this.enemyMaterials[i].emissive.setHex(w.time - e.hitAt < 0.12 ? 0xd3976b : 0x000000);
    });
    for (const item of w.pickups) {
      const g = this.pickups.get(item.id)!; g.visible = !item.taken;
      if (item.kind === 'core') { g.rotation.y = w.time * 0.8; g.position.y = 1.05 + Math.sin(w.time * 2) * 0.09; }
    }
    const m = this.matrixObject;
    this.bullets.count = Math.min(128, w.projectiles.length);
    w.projectiles.slice(0, 128).forEach((b, i) => {
      m.position.set(b.x * TILE, 1.2, b.z * TILE); m.scale.setScalar(1); m.rotation.set(0, 0, 0); m.updateMatrix(); this.bullets.setMatrixAt(i, m.matrix);
    });
    this.bullets.instanceMatrix.needsUpdate = true;
    let count = 0;
    for (let i = w.events.length - 1; i >= 0; i--) {
      const event = w.events[i], age = w.time - event.time;
      if (age > 0.5) break;
      if (event.type !== 'kill' && (event.type !== 'shot' || event.target === undefined)) continue;
      const enemy = w.enemies[event.target!];
      for (let j = 0; j < 7 && count < 256; j++) {
        const a = j * 2.4 + i;
        m.position.set(enemy.x * TILE + Math.sin(a) * age * 3, 1.2 + Math.cos(a) * age * 3 - age * age * 4, enemy.z * TILE + Math.cos(a * 2) * age * 3);
        m.scale.setScalar(Math.max(0.01, 0.07 * (1 - age * 2))); m.updateMatrix(); this.sparks.setMatrixAt(count++, m.matrix);
      }
    }
    this.sparks.count = count; this.sparks.instanceMatrix.needsUpdate = true;
    this.renderer.render(this.scene, this.camera);
    this.drawCalls = this.renderer.info.render.calls; this.triangles = this.renderer.info.render.triangles;
  }
}

export class GameAudio {
  context: AudioContext | null = null;
  enabled = true;
  cursor = 0;
  start() { this.context ??= new AudioContext(); void this.context.resume(); }
  update(world: World) {
    if (!this.context || !this.enabled) { this.cursor = world.events.length; return; }
    for (; this.cursor < world.events.length; this.cursor++) {
      const e = world.events[this.cursor];
      if (e.type === 'shot') this.tone(90, 0.12, 'sawtooth', 0.08, 22);
      else if (e.type === 'kill') this.tone(160, 0.18, 'square', 0.035, 45);
      else if (e.type === 'hurt') this.tone(48, 0.17, 'sawtooth', 0.06, 25);
      else if (e.type === 'med' || e.type === 'ammo' || e.type === 'core') this.tone(530, 0.13, 'sine', 0.05, 1000);
    }
  }
  tone(frequency: number, duration: number, type: OscillatorType, volume: number, end: number) {
    const c = this.context!, osc = c.createOscillator(), gain = c.createGain();
    osc.type = type; osc.frequency.setValueAtTime(frequency, c.currentTime); osc.frequency.exponentialRampToValueAtTime(end, c.currentTime + duration);
    gain.gain.setValueAtTime(volume, c.currentTime); gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration);
    osc.connect(gain); gain.connect(c.destination); osc.start(); osc.stop(c.currentTime + duration);
  }
}
