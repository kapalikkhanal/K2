/* K2 joint viewer.
 *
 * Loads site/assets/model/model.json -- the body tree baked out of
 * mjcf/k2_physics.xml by scripts/build_site_model.py -- and drives each joint
 * from a slider labelled with the servo that moves it on the real bus.
 *
 * The model is in MuJoCo's convention (+x forward, +y toward the model's
 * *_right chain, +z up); the whole robot sits under one group rotated into
 * three.js's Y-up. Slider ranges are the robot's operational limits from
 * hardware/joint_limits.json, not the CAD export's.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const MODEL = 'assets/model/';
const DEG = 180 / Math.PI;

const els = {
  canvas: document.getElementById('view'),
  panel: document.getElementById('joints'),
  status: document.getElementById('status'),
  readout: document.getElementById('readout'),
};

/* ---------------------------------------------------------------- scene --- */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c10);

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x141a22, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(0.8, 1.2, 0.9);
scene.add(key);
const rim = new THREE.DirectionalLight(0xff9d4d, 0.8);
rim.position.set(-0.8, 0.4, -0.7);
scene.add(rim);

const HILITE = new THREE.Color(0xff9d4d);

/* --------------------------------------------------------------- build --- */
const rotors = new Map();       // joint name -> {rotor, axis}
const jointMeshes = new Map();  // joint name -> [Mesh] (the whole subtree it moves)

/** MuJoCo stores quaternions w,x,y,z; three.js wants x,y,z,w. */
const mjQuat = ([w, x, y, z]) => new THREE.Quaternion(x, y, z, w);

async function buildBody(node, parent, gltf, jobs) {
  const body = new THREE.Group();
  body.name = node.name;
  body.position.fromArray(node.pos);
  body.quaternion.copy(mjQuat(node.quat));
  parent.add(body);

  // A joint rotates its own body and everything below it, about the body origin.
  let attach = body;
  if (node.joint) {
    const rotor = new THREE.Group();
    rotor.name = `rotor:${node.joint.name}`;
    body.add(rotor);
    attach = rotor;
    rotors.set(node.joint.name, {
      rotor,
      axis: new THREE.Vector3().fromArray(node.joint.axis).normalize(),
    });
    jointMeshes.set(node.joint.name, []);
  }

  for (const g of node.geoms) {
    jobs.push(gltf.loadAsync(`${MODEL}meshes/${g.mesh}.glb`).then((res) => {
      res.scene.updateWorldMatrix(true, true);
      res.scene.traverse((n) => {
        if (!n.isMesh) return;
        const geo = n.geometry.clone().applyMatrix4(n.matrixWorld);
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
          color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
          metalness: 0.3, roughness: 0.55,
        }));
        mesh.position.fromArray(g.pos);
        mesh.quaternion.copy(mjQuat(g.quat));
        attach.add(mesh);
      });
    }));
  }

  for (const child of node.children) await buildBody(child, attach, gltf, jobs);
  return body;
}

/** Every mesh a joint moves, i.e. its whole subtree -- used for highlighting. */
function indexSubtrees() {
  for (const [name, { rotor }] of rotors) {
    const list = jointMeshes.get(name);
    rotor.traverse((n) => { if (n.isMesh) list.push(n); });
  }
}

/* ------------------------------------------------------------------ ui --- */
const state = new Map();   // joint name -> radians
let M = null;              // model.json
let sweep = null;

const setJoint = (name, q) => {
  const r = rotors.get(name);
  if (r) r.rotor.quaternion.setFromAxisAngle(r.axis, q);
};

const pretty = (short) => short.replace(/_[RL]$/, '').replace(/_/g, ' ')
  .replace(/^./, (c) => c.toUpperCase());

function highlight(name, on) {
  for (const m of jointMeshes.get(name) ?? []) {
    m.material.emissive.set(on ? HILITE : 0x000000);
    m.material.emissiveIntensity = on ? 0.4 : 0;
  }
}

function buildPanel() {
  const sides = [
    { key: 'left', head: 'Physical LEFT leg', sub: 'model <code>*_right</code> · servos 0–5' },
    { key: 'right', head: 'Physical RIGHT leg', sub: 'model <code>*_left</code> · servos 6–11' },
  ];

  for (const side of sides) {
    const box = document.createElement('div');
    box.className = 'legbox';
    box.innerHTML = `<h3>${side.head}<span>${side.sub}</span></h3>`;

    for (const name of M.order) {
      const j = M.joints[name];
      if (j.physical_side !== side.key) continue;

      const row = document.createElement('div');
      row.className = 'jrow';
      row.innerHTML = `
        <div class="jhead">
          <span class="jname">${pretty(j.short)}</span>
          <span class="servo" title="Servo ID on the STS3215 bus">ID ${j.servo_id}</span>
          <span class="jval" data-val></span>
        </div>
        <input type="range" min="${j.lower}" max="${j.upper}" step="0.005">
        <div class="jfoot">
          <span class="lim">${(j.lower * DEG).toFixed(0)}°</span>
          <button class="sweep" type="button" title="Sweep this joint through its range">sweep</button>
          <span class="lim">${(j.upper * DEG).toFixed(0)}°</span>
        </div>
        <p class="jdesc">+ &rarr; ${j.positive_does.toLowerCase()}</p>`;

      const slider = row.querySelector('input');
      const val = row.querySelector('[data-val]');
      j._slider = slider;
      j._show = () => { val.textContent = `${(state.get(name) * DEG).toFixed(1)}°`; };

      slider.addEventListener('input', () => {
        sweep = null;
        state.set(name, Number(slider.value));
        setJoint(name, state.get(name));
        j._show();
      });
      row.addEventListener('pointerenter', () => highlight(name, true));
      row.addEventListener('pointerleave', () => { if (!sweep) highlight(name, false); });
      row.querySelector('.sweep').addEventListener('click', () => {
        if (sweep) highlight(sweep.name, false);
        sweep = { name, j, t: 0, dur: 2.4, cycles: 2 };
      });

      box.appendChild(row);
    }
    els.panel.appendChild(box);
  }
}

function setPose(pose) {
  if (sweep) highlight(sweep.name, false);
  sweep = null;
  for (const name of M.order) {
    const j = M.joints[name];
    const q = THREE.MathUtils.clamp(pose[name] ?? 0, j.lower, j.upper);
    state.set(name, q);
    setJoint(name, q);
    if (j._slider) { j._slider.value = q; j._show(); }
  }
}

/** Pull the camera back far enough that `box` fits both axes of the viewport. */
function frame(box) {
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  const dist = Math.max(sphere.radius / Math.sin(vFov / 2),
                        sphere.radius / Math.sin(hFov / 2)) * 1.12;
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center)
    .add(new THREE.Vector3(0.62, 0.30, 1).normalize().multiplyScalar(dist));
  camera.near = Math.max(dist / 200, 0.005);
  camera.far = dist * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

/* ---------------------------------------------------------------- boot --- */
(async function boot() {
  try {
    M = await fetch(MODEL + 'model.json').then((r) => {
      if (!r.ok) throw new Error(`model.json: HTTP ${r.status}`);
      return r.json();
    });
    for (const name of M.order) state.set(name, 0);

    els.status.textContent = 'loading meshes…';

    // MuJoCo is Z-up; three.js is Y-up.
    const upFix = new THREE.Group();
    upFix.rotation.x = -Math.PI / 2;
    scene.add(upFix);

    const jobs = [];
    await buildBody(M.tree, upFix, new GLTFLoader(), jobs);
    await Promise.all(jobs);
    indexSubtrees();

    buildPanel();
    setPose(M.presets.policy_default);

    const box = new THREE.Box3().setFromObject(upFix);
    const mid = box.getCenter(new THREE.Vector3());
    const grid = new THREE.GridHelper(1.2, 24, 0x2f3846, 0x1b2029);
    grid.position.set(mid.x, box.min.y, mid.z);
    scene.add(grid);

    resize();                       // establish the aspect before framing
    frame(box);

    els.status.remove();
    els.readout.textContent = `${M.order.length} joints · ${M.limits_source}`;

    document.getElementById('preset-default')
      .addEventListener('click', () => setPose(M.presets.policy_default));
    document.getElementById('preset-zero')
      .addEventListener('click', () => setPose(M.presets.zero));
  } catch (err) {
    els.status.innerHTML = `<b>Could not load the model.</b><br>${err.message}`;
    els.status.classList.add('err');
    console.error(err);
  }
})();

/* ---------------------------------------------------------------- loop --- */
function resize() {
  const { clientWidth: w, clientHeight: h } = els.canvas;
  if (w && h && (els.canvas.width !== w || els.canvas.height !== h)) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

let last = performance.now();
renderer.setAnimationLoop((now) => {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;

  if (sweep) {
    sweep.t += dt;
    const { name, j, dur, cycles } = sweep;
    if (sweep.t >= dur * cycles) {
      highlight(name, false);
      sweep = null;
    } else {
      const u = (1 - Math.cos((sweep.t / dur) * Math.PI * 2)) / 2;   // 0→1→0, eased
      const q = j.lower + (j.upper - j.lower) * u;
      state.set(name, q);
      setJoint(name, q);
      j._slider.value = q;
      j._show();
      highlight(name, true);
    }
  }

  resize();
  controls.update();
  renderer.render(scene, camera);
});
