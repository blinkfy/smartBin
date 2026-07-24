const STAGES = Object.freeze({
  idle: { range: [47, 47], label: '待命' }, scan: { range: [0, 5], label: '扫描识别' },
  approach: { range: [6, 12], label: '接近目标' }, grasp: { range: [13, 19], label: '抓取' },
  transport: { range: [20, 30], label: '运输' }, place: { range: [31, 38], label: '定位投放' },
  release: { range: [39, 42], label: '释放投放物' }, return: { range: [43, 47], label: '返回待命区' },
  completed: { range: [47, 47], label: '任务完成' }, error: { range: [0, 0], label: '异常停止' }
})
const ALIASES = Object.freeze({ recognition: 'scan', detect: 'scan', navigation: 'approach', navigate: 'approach', pick: 'grasp', carry: 'transport', drop: 'release', reset: 'return', ready: 'idle', done: 'completed' })
const ERROR_RE = /(?:error|fault|failed|abnormal|emergency|stopped|cancelled|offline|异常|故障|急停|停止|失败)/i
const COMPLETE_RE = /(?:completed|succeeded|success|done|finished|已完成|任务完成|闭环任务完成)/i
const IDLE_RE = /(?:idle|ready|standby|pending|待命|就绪|等待)/i

export function normalizeSortingWorkflowStage(value, fallback = 'idle') {
  const raw = String(value || '').trim().toLowerCase(), exact = ALIASES[raw] || raw
  if (STAGES[exact]) return exact
  if (/(?:recogn|detect|scan|识别|检测|扫描)/i.test(raw)) return 'scan'
  if (/(?:navigat|approach|导航|接近|到达抓取)/i.test(raw)) return 'approach'
  if (/(?:grasp|pick|抓取|夹取)/i.test(raw)) return 'grasp'
  if (/(?:transport|carry|运输|搬运|移送)/i.test(raw)) return 'transport'
  if (/(?:place|定位投放|投放至|对准投放)/i.test(raw)) return 'place'
  if (/(?:release|drop|释放|落入|完成投放)/i.test(raw)) return 'release'
  if (/(?:return|reset|返回|返航|复位)/i.test(raw)) return 'return'
  if (COMPLETE_RE.test(raw)) return 'completed'
  if (ERROR_RE.test(raw)) return 'error'
  if (IDLE_RE.test(raw)) return 'idle'
  return fallback
}

export function workflowStageFromProgress(progress) {
  const value = Number(progress)
  if (!Number.isFinite(value) || value <= 0) return 'scan'
  if (value >= 100) return 'completed'
  if (value < 20) return 'scan'
  if (value < 40) return 'approach'
  if (value < 55) return 'grasp'
  if (value < 70) return 'transport'
  if (value < 82) return 'place'
  if (value < 90) return 'release'
  return 'return'
}

export function mapSortingWorkflowStage({ status, deviceState, phase, progress, mode = 'robot' } = {}) {
  if (mode !== 'robot') return 'idle'
  const primaryState = String(status || ''), state = `${primaryState} ${deviceState || ''}`, phaseText = String(phase || '')
  const isActive = /(?:running|paused|active|executing|执行|暂停)/i.test(primaryState)
  if (ERROR_RE.test(state) || ERROR_RE.test(phaseText)) return 'error'
  if (COMPLETE_RE.test(state) || COMPLETE_RE.test(phaseText)) return 'completed'
  if (!isActive && IDLE_RE.test(state)) return 'idle'
  if (Number.isFinite(Number(progress))) return workflowStageFromProgress(progress)
  return normalizeSortingWorkflowStage(phaseText)
}

export function sortingWorkflowStageLabel(stage) { return STAGES[normalizeSortingWorkflowStage(stage)].label }

const STAGE_ORDER = ['scan', 'approach', 'grasp', 'transport', 'place', 'release', 'return']
const STAGE_LABELS = ['扫描', '接近', '抓取', '运输', '投放', '释放', '返回']
const CSS = `
  :host { display:block; width:100%; height:100%; min-width:0; }
  .viewport { position:relative; width:100%; height:100%; min-height:250px; overflow:hidden; border-radius:10px; background:#f2fbf8; }
  canvas { display:block; width:100%; height:100%; }
  .error { position:absolute; right:12px; bottom:10px; padding:5px 10px; border-radius:999px; color:#b91c1c; background:rgba(254,226,226,.94); font:600 12px system-ui,sans-serif; }
  .error[hidden] { display:none; }
  @media (max-width:760px) { .viewport { min-height:254px; } }
`

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value))
const lerp = (start, end, amount) => start + (end - start) * amount
const smoothstep = value => { const t = clamp(value); return t * t * (3 - 2 * t) }

function mixPose(from, to, progress) {
  const eased = smoothstep(progress), pose = {}
  Object.keys(from).forEach(key => { pose[key] = typeof from[key] === 'number' && typeof to[key] === 'number' ? lerp(from[key], to[key], eased) : (eased < .5 ? from[key] : to[key]) })
  return pose
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2)
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + width, y, x + width, y + height, r); ctx.arcTo(x + width, y + height, x, y + height, r); ctx.arcTo(x, y + height, x, y, r); ctx.arcTo(x, y, x + width, y, r); ctx.closePath()
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.decoding = 'async'; image.onload = () => resolve(image); image.onerror = () => reject(new Error(`Failed to load image: ${src}`)); image.src = src
  })
}

export class SortingWorkflowPlayer extends HTMLElement {
  static get observedAttributes() { return ['stage', 'progress', 'autoplay', 'running', 'loop', 'src', 'rig-src', 'timeline-src'] }

  constructor() {
    super(); this.attachShadow({ mode: 'open' })
    this._stage = 'idle'; this._progress = 0; this._frame = 47; this._playing = false; this._ready = false; this._raf = 0; this._startedAt = 0; this._revision = 0; this._errorLogged = false
    this._images = {}; this._rig = null; this._timeline = null
    this._tick = timestamp => {
      if (!this._playing) return
      const spec = this._timeline?.stages?.[this._stage], duration = Math.max(1, spec?.durationMs || 1000)
      if (!this._startedAt) this._startedAt = timestamp - this._progress * duration
      if (!this.hasAttribute('progress')) {
        const elapsed = timestamp - this._startedAt
        if (this._stage === 'scan' || this.loop) { const cycle = elapsed / duration % 2; this._progress = cycle <= 1 ? cycle : 2 - cycle }
        else this._progress = clamp(elapsed / duration)
      }
      const [start, end] = STAGES[this._stage].range
      this._frame = Math.round(lerp(start, end, this._progress))
      this.renderNow()
      this.dispatchEvent(new CustomEvent('framechange', { detail: { frame: this._frame, stage: this._stage, progress: this._progress } }))
      this._raf = requestAnimationFrame(this._tick)
    }
    this._handleResize = () => this._resize()
    this._handleVisibility = () => document.hidden ? this.pause() : (this.autoplay || this.running) && this.resume()
  }

  connectedCallback() {
    this.shadowRoot.innerHTML = `<style>${CSS}</style><div class="viewport"><canvas aria-label="自主垃圾分类投放分层关节动画"></canvas><span class="error" hidden>机器人分层资源暂时无法加载</span></div>`
    this._canvas = this.shadowRoot.querySelector('canvas'); this._ctx = this._canvas.getContext('2d', { alpha: true }); this._error = this.shadowRoot.querySelector('.error')
    this._stage = normalizeSortingWorkflowStage(this.getAttribute('stage') || 'idle')
    this._progress = this.hasAttribute('progress') ? clamp(Number(this.getAttribute('progress')) || 0) : 0
    if (typeof ResizeObserver === 'function') { this._resizeObserver = new ResizeObserver(this._handleResize); this._resizeObserver.observe(this) }
    else window.addEventListener('resize', this._handleResize)
    document.addEventListener('visibilitychange', this._handleVisibility)
    this._resize(); this._loadAssets()
  }

  disconnectedCallback() { this.destroy() }

  attributeChangedCallback(name, oldValue, newValue) {
    if (oldValue === newValue || !this.isConnected) return
    if (name === 'stage') this.setStage(newValue)
    else if (name === 'progress') { this._progress = clamp(Number(newValue) || 0); this.renderNow() }
    else if (['src', 'rig-src', 'timeline-src'].includes(name)) this._loadAssets()
    else if (name === 'autoplay' || name === 'running') (this.autoplay || this.running) ? this.resume() : this.pause()
  }

  get autoplay() { return this.hasAttribute('autoplay') }
  get running() { return this.hasAttribute('running') }
  get loop() { return this.hasAttribute('loop') }
  get stage() { return this._stage }
  set stage(value) { this.setStage(value) }

  async _loadAssets() {
    const revision = ++this._revision
    this.pause(); this._ready = false; this._errorLogged = false; if (this._error) this._error.hidden = true
    try {
      const rigUrl = this.getAttribute('rig-src') || '/sorting-robot/rig.json'
      const timelineUrl = this.getAttribute('timeline-src') || '/sorting-robot/timeline.json'
      const assetBase = this.getAttribute('src') || '/sorting-robot/layers'
      const [rigResponse, timelineResponse] = await Promise.all([fetch(rigUrl), fetch(timelineUrl)])
      if (!rigResponse.ok) throw new Error(`Rig HTTP ${rigResponse.status}`)
      if (!timelineResponse.ok) throw new Error(`Timeline HTTP ${timelineResponse.status}`)
      const rig = await rigResponse.json(), timeline = await timelineResponse.json()
      const entries = await Promise.all(Object.entries(rig.layers).map(async ([key, filename]) => [key, await loadImage(`${assetBase}/${filename}`)]))
      if (revision !== this._revision || !this.isConnected) return
      this._rig = rig; this._timeline = timeline; this._images = Object.fromEntries(entries); this._ready = true
      this._resize(); this.renderNow(); if (this.autoplay || this.running) this.resume(); this.dispatchEvent(new CustomEvent('ready'))
    } catch (error) { if (revision === this._revision) this._showError(error) }
  }

  _showError(error) {
    this.pause(); this._ready = false; if (this._error) this._error.hidden = false
    if (!this._errorLogged) { this._errorLogged = true; console.error('[SortingWorkflowPlayer] 分层机器人资源加载失败', error) }
    this.dispatchEvent(new CustomEvent('error', { detail: { message: '分层机器人资源加载失败', error } }))
  }

  _resize() {
    if (!this._canvas || !this._ctx) return
    const rect = this.getBoundingClientRect(); if (!rect.width || !rect.height) return
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1)), width = Math.round(rect.width * dpr), height = Math.round(rect.height * dpr)
    if (this._canvas.width !== width || this._canvas.height !== height) { this._canvas.width = width; this._canvas.height = height }
    this.renderNow()
  }

  _pose() {
    const spec = this._timeline?.stages?.[this._stage] || this._timeline?.stages?.idle
    if (!spec) return null
    const pose = mixPose(spec.from, spec.to, this._progress)
    if (Number.isFinite(spec.holdBottleAt)) pose.holdBottle = this._progress >= spec.holdBottleAt
    if (Number.isFinite(spec.holdBottleUntil)) pose.holdBottle = this._progress < spec.holdBottleUntil
    return pose
  }

  _rotation(name, degrees) { return { pivot: this._rig.pivots[name], degrees: Number(degrees) || 0 } }

  _drawLayer(ctx, key, rotations, left, top, scale, sourceRect = null) {
    const image = this._images[key]; if (!image) return
    ctx.save(); ctx.translate(left, top); ctx.scale(scale, scale)
    rotations.forEach(({ pivot, degrees }) => { const [x, y] = pivot; ctx.translate(x, y); ctx.rotate(degrees * Math.PI / 180); ctx.translate(-x, -y) })
    if (sourceRect) { const [sx, sy, sw, sh] = sourceRect; ctx.drawImage(image, sx, sy, sw, sh, sx, sy, sw, sh) }
    else ctx.drawImage(image, 0, 0, this._rig.sourceSize.width, this._rig.sourceSize.height)
    ctx.restore()
  }

  _drawRobot(ctx, pose, width, height, groundY) {
    const robotHeight = height * .72, scale = robotHeight / this._rig.sourceSize.height
    const travelProgress = clamp((pose.robotX - 500) / (1030 - 500)), robotX = width * lerp(.2, .82, travelProgress)
    const left = robotX - this._rig.sourceSize.width * scale / 2, top = groundY - robotHeight
    const waist = this._rotation('waist', pose.waist), head = this._rotation('head', pose.head)
    const rightShoulder = this._rotation('rightShoulder', pose.rightShoulder), rightElbow = this._rotation('rightElbow', pose.rightElbow)
    const leftShoulder = this._rotation('leftShoulder', pose.leftShoulder), leftElbow = this._rotation('leftElbow', pose.leftElbow), leftLower = [waist, leftShoulder, leftElbow]
    this._drawLayer(ctx, 'base', [], left, top, scale)
    this._drawLayer(ctx, 'rightUpperArm', [waist, rightShoulder], left, top, scale)
    this._drawLayer(ctx, 'torso', [waist], left, top, scale)
    this._drawLayer(ctx, 'head', [waist, head], left, top, scale)
    this._drawLayer(ctx, 'rightLowerArmHand', [waist, rightShoulder, rightElbow], left, top, scale)
    this._drawLayer(ctx, 'leftUpperArm', [waist, leftShoulder], left, top, scale)
    this._drawLayer(ctx, 'leftLowerEmptyHand', leftLower, left, top, scale)
    if (pose.holdBottle) this._drawLayer(ctx, 'bottle', leftLower, left, top, scale)
    this._drawLayer(ctx, 'rightShoulderCover', [waist], left, top, scale)
    this._drawLayer(ctx, 'leftShoulderCover', [waist], left, top, scale)
    this._drawLayer(ctx, 'leftElbowCover', [waist, leftShoulder], left, top, scale)
    this._drawLayer(ctx, 'sockets', [waist], left, top, scale)
    return { scale, robotX }
  }

  _drawBottle(ctx, x, y, scale, angle = 0, alpha = 1) {
    const image = this._images.bottle; if (!image) return
    const [anchorX, anchorY] = this._rig.anchors?.bottleCenter || [59.5, 360.5]
    ctx.save(); ctx.globalAlpha = alpha; ctx.translate(x, y); ctx.rotate(angle); ctx.scale(scale, scale); ctx.drawImage(image, -anchorX, -anchorY); ctx.restore()
  }

  _drawRail(ctx, width, height) {
    const top = Math.max(24, height * .11), left = width * .09, available = width * .83
    const activeIndex = this._stage === 'completed' ? 6 : STAGE_ORDER.indexOf(this._stage)
    ctx.save(); ctx.lineCap = 'round'; ctx.strokeStyle = '#dbe7e3'; ctx.lineWidth = Math.max(2, height * .009); ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left + available, top); ctx.stroke()
    if (activeIndex >= 0) { ctx.strokeStyle = '#10b981'; ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left + available * activeIndex / 6, top); ctx.stroke() }
    STAGE_ORDER.forEach((stage, index) => {
      const x = left + available * index / 6, done = this._stage === 'completed' || activeIndex >= 0 && index <= activeIndex, active = index === activeIndex && this._stage !== 'completed'
      ctx.beginPath(); ctx.arc(x, top, active ? 7 : 5, 0, Math.PI * 2); ctx.fillStyle = done ? '#10b981' : '#dbe7e3'; ctx.fill()
      if (active) { ctx.beginPath(); ctx.arc(x, top, 12, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(16,185,129,.28)'; ctx.lineWidth = 3; ctx.stroke() }
      if (width >= 520 || active) { ctx.fillStyle = done ? '#047857' : '#94a3b8'; ctx.font = `700 ${Math.max(10, height * .034)}px system-ui,sans-serif`; ctx.textAlign = 'center'; ctx.fillText(STAGE_LABELS[index], x, top + Math.max(18, height * .065)) }
    }); ctx.restore()
  }

  renderNow() {
    if (!this._ready || !this._ctx || !this._rig || !this._timeline) return
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1)), width = this._canvas.width / dpr, height = this._canvas.height / dpr
    if (!width || !height) return
    const ctx = this._ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'
    const background = ctx.createLinearGradient(0, 0, width, height); background.addColorStop(0, '#f8fffd'); background.addColorStop(.58, '#f2fbf8'); background.addColorStop(1, '#eaf8f5'); ctx.fillStyle = background; ctx.fillRect(0, 0, width, height)
    this._drawRail(ctx, width, height)
    const pose = this._pose(); if (!pose) return
    const groundY = height * .9, sourceX = width * .1, binX = width * .76, binWidth = Math.max(54, Math.min(96, height * .23)), binHeight = height * .31
    ctx.save(); ctx.strokeStyle = 'rgba(16,185,129,.2)'; ctx.lineWidth = 2; ctx.setLineDash([7, 10]); ctx.beginPath(); ctx.moveTo(sourceX, groundY); ctx.lineTo(binX, groundY); ctx.stroke(); ctx.restore()
    ctx.save(); ctx.shadowColor = ['place', 'release', 'completed'].includes(this._stage) ? 'rgba(16,185,129,.32)' : 'rgba(15,23,42,.12)'; ctx.shadowBlur = 18; ctx.fillStyle = '#fff'; ctx.strokeStyle = ['place', 'release'].includes(this._stage) ? '#10b981' : '#cbd5e1'; ctx.lineWidth = 2; roundedRect(ctx, binX - binWidth / 2, groundY - binHeight, binWidth, binHeight, 12); ctx.fill(); ctx.stroke(); ctx.fillStyle = '#16352f'; roundedRect(ctx, binX - binWidth * .34, groundY - binHeight - 7, binWidth * .68, 12, 5); ctx.fill(); ['#2563eb', '#16a34a', '#dc2626', '#64748b'].forEach((color, index) => { ctx.fillStyle = color; ctx.fillRect(binX - binWidth / 2 + binWidth / 4 * index, groundY - 18, binWidth / 4, 18) }); ctx.restore()
    const robot = this._drawRobot(ctx, pose, width, height, groundY), bottleScale = Math.max(.16, robot.scale * .65), targetY = height * (500 / 900)
    if (pose.showTarget && !pose.holdBottle) { this._drawBottle(ctx, sourceX, targetY, bottleScale, -.16); ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.strokeRect(sourceX - 22, targetY - 35, 44, 70) }
    if (pose.bottleDrop > 0) { const drop = smoothstep(pose.bottleDrop); this._drawBottle(ctx, lerp(binX - width * .06, binX, drop), lerp(height * (470 / 900), height * (640 / 900), drop), bottleScale, .35 + drop * .9, 1 - drop * .35) }
    const text = this._stage === 'idle' ? '等待任务' : this._stage === 'completed' ? '分类投放完成' : sortingWorkflowStageLabel(this._stage)
    const chipX = width * .035, chipY = height * .2, chipWidth = Math.min(160, width * .27), chipHeight = Math.max(34, height * .11)
    ctx.fillStyle = this._stage === 'error' ? 'rgba(254,226,226,.94)' : 'rgba(255,255,255,.94)'; ctx.strokeStyle = this._stage === 'error' ? 'rgba(220,38,38,.28)' : 'rgba(5,150,105,.2)'; roundedRect(ctx, chipX, chipY, chipWidth, chipHeight, 11); ctx.fill(); ctx.stroke(); ctx.fillStyle = this._stage === 'error' ? '#dc2626' : '#059669'; ctx.font = `800 ${Math.max(12, height * .044)}px system-ui,sans-serif`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(text, chipX + Math.max(10, width * .02), chipY + chipHeight / 2)
  }

  play() { this.resume() }
  resume() {
    if (!this._ready || this._playing || document.hidden || ['idle', 'completed', 'error'].includes(this._stage)) { this.renderNow(); return }
    this._playing = true; this._startedAt = 0; this._raf = requestAnimationFrame(this._tick)
  }
  pause() { this._playing = false; cancelAnimationFrame(this._raf); this._raf = 0; this._startedAt = 0 }
  seek(value) {
    this._frame = clamp(Number(value) || 0, 0, 47); const [start, end] = STAGES[this._stage].range; this._progress = end === start ? 0 : clamp((this._frame - start) / (end - start)); this.renderNow()
    this.dispatchEvent(new CustomEvent('framechange', { detail: { frame: this._frame, stage: this._stage, progress: this._progress } }))
  }
  setStage(value, progress) {
    const next = normalizeSortingWorkflowStage(value), nextProgress = Number.isFinite(progress) ? clamp(progress) : this.hasAttribute('progress') ? clamp(Number(this.getAttribute('progress')) || 0) : 0
    if (next === this._stage && nextProgress === this._progress) { if (this.autoplay || this.running) this.resume(); return }
    this.pause(); this._stage = next; this._progress = nextProgress; this._frame = STAGES[next].range[0]; this.renderNow(); this.dispatchEvent(new CustomEvent('stagechange', { detail: { stage: next, progress: this._progress } })); if (this.autoplay || this.running) this.resume()
  }
  destroy() {
    this.pause(); this._revision += 1; this._resizeObserver?.disconnect(); window.removeEventListener('resize', this._handleResize); document.removeEventListener('visibilitychange', this._handleVisibility); this._images = {}; this._rig = null; this._timeline = null; this._ready = false
  }
}

if (!customElements.get('sorting-workflow-player')) customElements.define('sorting-workflow-player', SortingWorkflowPlayer)
