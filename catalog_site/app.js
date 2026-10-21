const state = { tags: new Set(), query: '', running: false, analysisTag: '', analyticsTags: [] };
const $ = (selector) => document.querySelector(selector);
const grid = $('#skin-grid');
const search = $('#search');

function skinUrl(filename) { return `/skins/${filename.split('/').map(encodeURIComponent).join('/')}`; }
// Быстрый изометрический рендер по настоящим областям Minecraft-развёртки.
// Это даёт 3D-превью для 48 карточек без 48 тяжёлых WebGL-контекстов.
function drawTexture(ctx, image, source, a, b, c, d, e, f) {
  const [sx, sy, sw, sh] = source;
  ctx.save(); ctx.setTransform(a, b, c, d, e, f); ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh); ctx.restore();
}
function drawCube(ctx, image, faces, x, y, width, height, depth) {
  const offset = depth * .34;
  // Верхняя, правая и фронтальная плоскости — порядок создаёт объём без WebGL.
  drawTexture(ctx, image, faces.top, width / faces.top[2], 0, depth / faces.top[3], -offset / faces.top[3], x, y + offset);
  drawTexture(ctx, image, faces.right, depth / faces.right[2], -offset / faces.right[2], 0, height / faces.right[3], x + width, y + offset);
  drawTexture(ctx, image, faces.front, width / faces.front[2], 0, 0, height / faces.front[3], x, y + offset);
}
const THUMB = {
  head: { front:[8,8,8,8], right:[0,8,8,8], top:[8,0,8,8] }, hat: { front:[40,8,8,8], right:[32,8,8,8], top:[40,0,8,8] },
  body: { front:[20,20,8,12], right:[16,20,4,12], top:[20,16,8,4] },
  rArm: { front:[44,20,4,12], right:[40,20,4,12], top:[44,16,4,4] }, lArm: { front:[36,52,4,12], right:[32,52,4,12], top:[36,48,4,4] },
  rLeg: { front:[4,20,4,12], right:[0,20,4,12], top:[4,16,4,4] }, lLeg: { front:[20,52,4,12], right:[16,52,4,12], top:[20,48,4,4] },
};
function drawSkinThumbnail(canvas, image) {
  const ctx = canvas.getContext('2d'); const { width, height } = canvas;
  ctx.imageSmoothingEnabled = false; ctx.fillStyle = '#ede9f8'; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#dfd9ee'; ctx.beginPath(); ctx.ellipse(width / 2, 211, 69, 12, 0, 0, Math.PI * 2); ctx.fill();
  // Задние конечности, тело, передние конечности, затем голова — естественный порядок слоёв.
  drawCube(ctx, image, THUMB.lArm, 54, 69, 25, 56, 10); drawCube(ctx, image, THUMB.lLeg, 109, 127, 26, 58, 10);
  drawCube(ctx, image, THUMB.rLeg, 82, 127, 26, 58, 10); drawCube(ctx, image, THUMB.body, 84, 67, 52, 60, 16);
  drawCube(ctx, image, THUMB.rArm, 142, 69, 25, 56, 10); drawCube(ctx, image, THUMB.head, 80, 18, 59, 48, 18);
  // Второй слой головы (шапка/волосы/аксессуары) рисуется поверх базового лица.
  drawCube(ctx, image, THUMB.hat, 79, 17, 61, 50, 19);
}
function chip(tag, clickable = false) {
  const node = document.createElement(clickable ? 'button' : 'span');
  node.className = clickable ? `tag-filter ${state.tags.has(tag) ? 'selected' : ''}` : 'tag';
  node.textContent = tag;
  if (clickable) node.onclick = () => { state.tags.has(tag) ? state.tags.delete(tag) : state.tags.add(tag); loadSkins(); renderActiveFilters(); renderPopularTags(); };
  return node;
}

function renderPopularTags() {
  document.querySelectorAll('#popular-tags .tag-filter').forEach((button) => button.classList.toggle('selected', state.tags.has(button.textContent)));
}
function renderActiveFilters() {
  const wrap = $('#active-filters'); wrap.replaceChildren();
  for (const tag of state.tags) {
    const button = chip(`× ${tag}`, true);
    button.onclick = () => { state.tags.delete(tag); loadSkins(); renderActiveFilters(); renderPopularTags(); };
    wrap.append(button);
  }
}

function renderTagMatrix() {
  const matrix = $('#tag-matrix'); matrix.replaceChildren();
  for (const { tag, count } of state.analyticsTags) {
    const button = document.createElement('button'); button.className = 'matrix-tag'; button.dataset.tag = tag;
    button.innerHTML = `${tag}<small>${count.toLocaleString('ru-RU')}</small>`;
    button.onclick = () => showRelations(tag);
    matrix.append(button);
  }
}

async function showRelations(tag) {
  state.analysisTag = tag;
  const data = await (await fetch(`/api/tag-relations?tag=${encodeURIComponent(tag)}&limit=60`)).json();
  const related = new Map(data.related.map(({ tag: relatedTag, count }) => [relatedTag, count]));
  const maximum = Math.max(...related.values(), 1);
  document.querySelectorAll('.matrix-tag').forEach((button) => {
    const count = related.get(button.dataset.tag) || 0;
    button.classList.toggle('active', button.dataset.tag === tag);
    button.classList.toggle('related', count > 0);
    button.style.opacity = count ? String(.45 + .55 * Math.sqrt(count / maximum)) : '.3';
    button.title = count ? `${tag} + ${button.dataset.tag}: ${count.toLocaleString('ru-RU')} скинов` : '';
  });
  $('#relation-info').textContent = data.related.length
    ? `Для «${tag}» подсвечены теги, которые чаще всего встречаются вместе с ним. Нажмите «Добавить в фильтр», чтобы сузить выдачу.`
    : `Для «${tag}» среди популярных тегов пока нет связанных пар.`;
  const addButton = $('#add-analysis-tag'); addButton.disabled = false; addButton.textContent = `+ Добавить «${tag}» в фильтр`;
}

function renderSkins(skins, total) {
  grid.replaceChildren();
  const template = $('#card-template');
  for (const skin of skins) {
    const node = template.content.firstElementChild.cloneNode(true);
    const image = node.querySelector('img'); image.alt = skin.filename;
    image.onload = () => drawSkinThumbnail(node.querySelector('.skin-preview'), image);
    image.src = skinUrl(skin.filename);
    node.querySelector('.file-name').textContent = skin.filename.split('/').pop();
    skin.tags.slice(0, 3).forEach((tag) => node.querySelector('.card-tags').append(chip(tag)));
    node.querySelector('button').onclick = () => openViewer(skin);
    grid.append(node);
  }
  $('#result-title').textContent = state.query || state.tags.size ? `Найдено: ${total.toLocaleString('ru-RU')}` : 'Случайные скины';
}

async function loadSkins() {
  const params = new URLSearchParams({ limit: '48' });
  if (state.query) params.set('q', state.query);
  if (state.tags.size) params.set('tags', [...state.tags].join(','));
  grid.style.opacity = '.45';
  try { const response = await fetch(`/api/skins?${params}`); const data = await response.json(); renderSkins(data.skins, data.total); }
  finally { grid.style.opacity = '1'; }
}

async function bootstrap() {
  const data = await (await fetch('/api/popular-tags?limit=40')).json();
  $('#total-count').textContent = data.skin_count.toLocaleString('ru-RU');
  $('#tag-count').textContent = data.tag_count.toLocaleString('ru-RU');
  data.tags.slice(0, 24).forEach(({ tag, count }) => { const button = chip(tag, true); button.title = `${count.toLocaleString('ru-RU')} скинов`; $('#popular-tags').append(button); });
  state.analyticsTags = data.tags; renderTagMatrix(); showRelations(data.tags[0].tag);
  loadSkins();
}

let searchTimer;
search.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.query = search.value; loadSkins(); }, 280); });
$('#randomize').onclick = loadSkins; $('#more').onclick = loadSkins;
$('#add-analysis-tag').onclick = () => {
  if (!state.analysisTag) return;
  state.tags.add(state.analysisTag); loadSkins(); renderActiveFilters(); renderPopularTags();
};

// ---- 3D Minecraft skin viewer -------------------------------------------------
// skinview3d знает официальную UV-развёртку Minecraft, дополнительные слои и slim-модель.
const canvasWrap = $('#canvas-wrap');
const canvas = document.createElement('canvas'); canvasWrap.append(canvas);
const viewer3d = new skinview3d.SkinViewer({ canvas, width: 500, height: 500, background: 0x292840, zoom: .72 });
viewer3d.controls.enableRotate = true; viewer3d.controls.enableZoom = true; viewer3d.controls.enablePan = false;
viewer3d.animation = new skinview3d.IdleAnimation();
function resize() { viewer3d.width = canvasWrap.clientWidth; viewer3d.height = canvasWrap.clientHeight; }
new ResizeObserver(resize).observe(canvasWrap);
function openViewer(skin) {
  $('#viewer-name').textContent = skin.filename.split('/').pop(); const tags = $('#viewer-tags'); tags.replaceChildren(); skin.tags.forEach((tag) => tags.append(chip(tag)));
  $('#viewer').classList.add('open'); $('#viewer').setAttribute('aria-hidden', 'false'); resize();
  viewer3d.loadSkin(skinUrl(skin.filename));
  const download = $('#download-skin'); download.href = skinUrl(skin.filename); download.download = skin.filename.split('/').pop();
}
$('#close-viewer').onclick = () => { $('#viewer').classList.remove('open'); $('#viewer').setAttribute('aria-hidden', 'true'); };
$('#run-toggle').onclick = (event) => { state.running = !state.running; const animation = state.running ? new skinview3d.RunningAnimation() : new skinview3d.IdleAnimation(); animation.speed = state.running ? .42 : 1; viewer3d.animation = animation; event.currentTarget.classList.toggle('active', state.running); event.currentTarget.textContent = state.running ? 'Ⅱ Остановить бег' : '▷ Включить бег'; };
bootstrap();
