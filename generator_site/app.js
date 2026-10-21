const $ = s => document.querySelector(s);
const state = {tags: [], selected: new Set(), viewers: [], observers: [], suggestions: [], recommendations: [], focused: -1, busy: false, modelId: null};
let recommendationTimer, recommendationController, recommendationVersion = 0;
const input = $('#tag-input');
let uploadedUrl, uploadController, uploadVersion = 0, recognizedTags = [], descriptionRows = [];
function renderDescription() {
  $('#recognized-tags').replaceChildren();
  const allowed = new Set(state.tags.map(row => row.tag));
  recognizedTags = descriptionRows.filter(row => allowed.has(row.tag)).map(row => row.tag);
  for (const row of descriptionRows) {
    const supported = allowed.has(row.tag);
    const node = document.createElement(supported ? 'button' : 'span');
    node.className = 'chip'; node.textContent = `${row.tag} · ${Math.round(row.score * 100)}%`;
    if (supported) { node.type = 'button'; node.title = 'Добавить тег в генерацию'; node.onclick = () => addTag(row.tag); }
    else node.title = 'Распознанный тег отсутствует в словаре генератора';
    $('#recognized-tags').append(node);
  }
  $('#use-recognized-tags').hidden = !recognizedTags.length;
}
async function loadModel() {
  state.modelId = $('#generator-model').value;
  $('#generator-model').disabled = true; $('#generate-button').disabled = true;
  try {
    const data = await api(`/api/tags?model=${encodeURIComponent(state.modelId)}`);
    state.tags = data.tags;
    const allowed = new Set(data.tags.map(row => row.tag));
    const removed = [...state.selected].filter(tag => !allowed.has(tag));
    removed.forEach(tag => state.selected.delete(tag));
    input.value = ''; renderTags(); suggest(); renderDescription();
    $('#model-hint').textContent = `${data.tags.length.toLocaleString('ru-RU')} доступных тегов`;
    if (removed.length) status(`Теги недоступны этой модели и удалены: ${removed.join(', ')}`);
    localStorage.setItem('skinlab-model', state.modelId);
    $('#generate-button').disabled = state.busy;
  } catch (error) { status(error.message, 'error'); }
  finally { $('#generator-model').disabled = state.busy; }
}
$('#generator-model').onchange = loadModel;
$('#skin-upload').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  const version = ++uploadVersion; uploadController?.abort(); uploadController = new AbortController();
  recognizedTags = []; descriptionRows = []; $('#recognized-tags').replaceChildren(); $('#use-recognized-tags').hidden = true;
  $('#uploaded-preview').hidden = true;
  if (uploadedUrl) { URL.revokeObjectURL(uploadedUrl); uploadedUrl = null; }
  if (file.size > 1024 * 1024) { $('#upload-status').textContent = 'Выберите PNG размером до 1 МБ.'; return; }
  $('#upload-status').textContent = 'Распознаём скин…';
  try {
    const data = await api(`/api/describe-skin?model=${encodeURIComponent(state.modelId)}`, {method: 'POST', headers: {'Content-Type': 'image/png'}, body: file, signal: uploadController.signal});
    if (version !== uploadVersion) return;
    uploadedUrl = URL.createObjectURL(file); $('#uploaded-image').src = uploadedUrl;
    $('#uploaded-name').textContent = file.name; $('#uploaded-preview').hidden = false;
    descriptionRows = data.tags; renderDescription();
    $('#upload-status').textContent = data.low_confidence ? 'Распознаватель не уверен. Ниже наиболее вероятные теги.' : 'Описание готово. Нажми тег, чтобы использовать его.';
    $('#use-recognized-tags').hidden = !recognizedTags.length;
  } catch (error) { if (error.name !== 'AbortError' && version === uploadVersion) $('#upload-status').textContent = error.message; }
  finally { if (version === uploadVersion) event.target.value = ''; }
};
$('#use-recognized-tags').onclick = () => {
  const available = Math.max(0, 32 - state.selected.size);
  recognizedTags.filter(tag => !state.selected.has(tag)).slice(0, available).forEach(tag => state.selected.add(tag));
  input.value = ''; renderTags(); suggest(); status('Распознанные теги добавлены. Можно изменить их и сгенерировать скины.');
};
async function api(url, options) { const response = await fetch(url, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Ошибка сервера'); return data; }
function status(text, kind = '') { $('#status').textContent = text; $('#status').className = kind; }
function addTag(tag) { state.selected.add(tag); input.value = ''; renderTags(); suggest(); input.focus(); }
function chip(tag, selected = false) { const button = document.createElement('button'); button.type = 'button'; button.className = 'chip' + (selected ? ' active' : ''); button.textContent = tag; if (selected) button.append(document.createTextNode(' ×')); button.onclick = () => { state.selected.has(tag) ? state.selected.delete(tag) : state.selected.add(tag); renderTags(); suggest(); }; return button; }
function renderTags() {
  $('#selected-tags').replaceChildren(...[...state.selected].map(tag => chip(tag, true)));
  updateRecommendations();
}
function showRecommendations() {
  $('#suggestion-heading').textContent = state.selected.size ? 'Следующий тег' : 'Популярные теги';
  const rows = state.selected.size ? state.recommendations : state.tags.slice(0, 12);
  $('#popular-tags').replaceChildren(...rows.map(row => chip(row.tag)));
}
function updateRecommendations() {
  clearTimeout(recommendationTimer);
  recommendationController?.abort();
  const version = ++recommendationVersion;
  state.recommendations = []; showRecommendations();
  if (!state.selected.size) { $('#recommendation-status').textContent = ''; return; }
  $('#recommendation-status').textContent = 'Подбираем следующий тег…';
  recommendationTimer = setTimeout(async () => {
    recommendationController = new AbortController();
    try {
      const data = await api('/api/suggest', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tags: [...state.selected], model: state.modelId}), signal: recommendationController.signal});
      if (version !== recommendationVersion) return;
      state.recommendations = data.tags.filter(row => !state.selected.has(row.tag));
      showRecommendations(); suggest();
      $('#recommendation-status').textContent = state.recommendations.length ? 'Предложения по выбранным тегам' : 'Нет дополнительных предложений.';
    } catch (error) {
      if (error.name === 'AbortError' || version !== recommendationVersion) return;
      $('#recommendation-status').textContent = 'Не удалось получить предложения модели.';
      console.error(error);
    }
  }, 150);
}
function suggest() {
  const query = input.value.trim().toLowerCase(); state.focused = -1;
  // Search only after three characters. Empty input shows contextual model results.
  state.suggestions = query.length >= 3
    ? state.tags.filter(row => !state.selected.has(row.tag) && row.tag.includes(query)).slice(0, 8)
    : (!query && state.selected.size ? state.recommendations : []);
  input.removeAttribute('aria-activedescendant');
  const list = $('#suggestions'); list.replaceChildren(); list.hidden = document.activeElement !== input || !state.suggestions.length; input.setAttribute('aria-expanded', String(!list.hidden));
  state.suggestions.forEach((row, index) => { const button = document.createElement('button'); button.type = 'button'; button.role = 'option'; button.id = `suggestion-${index}`; const name = document.createElement('span'); name.textContent = row.tag; const count = document.createElement('small'); count.textContent = query ? row.count.toLocaleString('ru-RU') : 'рекомендация'; button.append(name, count); button.onmousedown = event => event.preventDefault(); button.onclick = () => addTag(row.tag); list.append(button); });
}
function commitInput() {
  const parts = input.value.split(',').map(t => t.trim().toLowerCase()).filter(Boolean);
  const unknown = parts.filter(t => !state.tags.some(row => row.tag === t));
  if (unknown.length) throw new Error(`Выбери тег из подсказок: ${unknown.join(', ')}`);
  parts.forEach(t => state.selected.add(t)); input.value = ''; renderTags(); suggest();
}
input.addEventListener('input', () => { if (input.value.includes(',')) { try { const parts = input.value.split(','); const tail = parts.pop(); const completed = parts.map(t => t.trim().toLowerCase()).filter(Boolean); if (completed.every(t => state.tags.some(row => row.tag === t))) { completed.forEach(t => state.selected.add(t)); input.value = tail; renderTags(); } } catch {} } suggest(); });
input.addEventListener('focus', suggest);
input.addEventListener('blur', () => { $('#suggestions').hidden = true; input.setAttribute('aria-expanded', 'false'); });
input.addEventListener('keydown', event => {
  if (['ArrowDown', 'ArrowUp'].includes(event.key) && state.suggestions.length) { event.preventDefault(); state.focused = (state.focused + (event.key === 'ArrowDown' ? 1 : -1) + state.suggestions.length) % state.suggestions.length; [...$('#suggestions').children].forEach((button, i) => button.classList.toggle('focused', i === state.focused)); input.setAttribute('aria-activedescendant', `suggestion-${state.focused}`); }
  if (event.key === 'Enter') { event.preventDefault(); try { const exact = state.tags.find(row => row.tag === input.value.trim().toLowerCase()); if (state.focused >= 0) addTag(state.suggestions[state.focused].tag); else if (exact) addTag(exact.tag); else if (state.suggestions.length && !input.value.includes(',')) addTag(state.suggestions[0].tag); else commitInput(); } catch (error) { status(error.message, 'error'); } }
  if (event.key === 'Escape') $('#suggestions').hidden = true;
});
$('#pick-tags').onclick = () => { const choose = options => { const present = options.filter(tag => state.tags.some(row => row.tag === tag)); return present[Math.floor(Math.random() * present.length)]; }; const tags = [choose(['girl', 'boy']), choose(['pink', 'blue', 'red', 'black', 'purple']), choose(['hoodie', 'cute', 'aesthetic'])].filter(Boolean); state.selected = new Set(tags.length ? tags : state.tags.slice(0, 3).map(row => row.tag)); renderTags(); };
function clearViewers() { state.observers.forEach(observer => observer.disconnect()); state.observers = []; state.viewers.forEach(viewer => viewer.dispose()); state.viewers = []; }
function emptyCards(pending = false) { clearViewers(); $('#skin-grid').replaceChildren(); for (let i = 0; i < 4; i++) { const card = document.createElement('article'); card.className = 'skin-card' + (pending ? ' pending' : ''); card.innerHTML = `<div class="skin-stage"><span class="variant">0${i + 1}</span><span class="slim-label">SLIM</span><div class="placeholder"></div></div><div class="card-info"><p class="seed">${pending ? 'Создаём новый образ…' : 'Твой следующий скин'}</p><div class="empty-label">${pending ? 'Генерация' : 'Здесь появится результат'}</div></div>`; $('#skin-grid').append(card); } }
async function renderResults(results) {
  clearViewers(); $('#skin-grid').replaceChildren();
  for (const [index, item] of results.slice(0, 4).entries()) {
    const card = document.createElement('article'); card.className = 'skin-card';
    card.innerHTML = `<div class="skin-stage"><span class="variant">0${index + 1}</span><span class="slim-label">SLIM</span><canvas aria-label="3D-просмотр скина ${index + 1}"></canvas></div><div class="card-info"><p class="seed"></p><a class="download">↓ Скачать PNG</a></div>`;
    card.querySelector('.seed').textContent = `СИД ${item.seed}`; const link = card.querySelector('.download'); link.href = item.url; link.download = item.name;
    $('#skin-grid').append(card);
    const stage = card.querySelector('.skin-stage');
    try {
      const viewer = new skinview3d.SkinViewer({canvas: card.querySelector('canvas'), width: stage.clientWidth, height: stage.clientHeight, zoom: .85, fov: 40});
      state.viewers.push(viewer); viewer.controls.enablePan = false; viewer.controls.enableZoom = true;
      viewer.camera.position.set(24, 12, 50); viewer.controls.update();
      const reset = document.createElement('button'); reset.type = 'button'; reset.className = 'reset-view'; reset.textContent = '↺'; reset.setAttribute('aria-label', `Сбросить ракурс скина ${index + 1}`); reset.title = 'Сбросить ракурс'; reset.onclick = () => { viewer.camera.position.set(24, 12, 50); viewer.controls.target.set(0, 0, 0); viewer.controls.update(); }; stage.append(reset);
      await viewer.loadSkin(item.url, {model: 'slim'});
      const body = viewer.playerObject.skin;
      body.rightArm.rotation.x = -.22; body.leftArm.rotation.x = .22; body.rightLeg.rotation.x = .15; body.leftLeg.rotation.x = -.15;
      const observer = new ResizeObserver(() => { viewer.width = stage.clientWidth; viewer.height = stage.clientHeight; }); observer.observe(stage); state.observers.push(observer);
    } catch (error) { console.error(error); const image = document.createElement('img'); image.src = item.url; image.className = 'fallback'; image.alt = 'Развёртка скина'; stage.append(image); const message = document.createElement('span'); message.className = 'preview-error'; message.textContent = '3D недоступен. PNG можно скачать.'; stage.append(message); }
  }
}
function busy(value) { state.busy = value; $('#generate-button').disabled = value; $('#history-button').disabled = value; $('#generator-model').disabled = value; }
async function poll(id) {
  busy(true); const start = Date.now();
  try { while (true) { const job = await api(`/api/jobs/${encodeURIComponent(id)}`); if (job.state === 'failed') throw new Error(job.error); if (job.state === 'done') { await renderResults(job.results); status('Готово! Четыре скина сохранены. Можно скачать любой.'); localStorage.removeItem('skinlab-job'); break; } status(`Генерируем четыре скина… ${Math.floor((Date.now() - start) / 1000)} с\nМожно оставить эту вкладку открытой.`, 'busy'); await new Promise(resolve => setTimeout(resolve, 1500)); } }
  catch (error) { status(error.message, 'error'); localStorage.removeItem('skinlab-job'); }
  finally { busy(false); $('#skin-grid').querySelectorAll('.pending').forEach(card => card.classList.remove('pending')); }
}
$('#generate-form').onsubmit = async event => { event.preventDefault(); if (state.busy) return; try { commitInput(); if (!state.selected.size) throw new Error('Добавь хотя бы один тег.'); busy(true); const job = await api('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({tags: [...state.selected], steps: Number($('#steps').value), model: state.modelId})}); localStorage.setItem('skinlab-job', job.id); emptyCards(true); await poll(job.id); } catch (error) { status(error.message, 'error'); busy(false); } };
$('#history-button').onclick = async () => { try { const data = await api('/api/history'); if (!data.results.length) return status('Пока нет сохранённых скинов. Создай первую четвёрку.'); await renderResults(data.results); status('Показаны четыре последних сохранённых скина.'); } catch (error) { status(error.message, 'error'); } };
async function init() { emptyCards(); try { const data = await api('/api/models'); const select = $('#generator-model'); for (const row of data.models) { const option = document.createElement('option'); option.value = row.id; option.textContent = row.name + (row.ready ? '' : ' · ещё обучается'); option.disabled = !row.ready; select.append(option); } const savedModel = localStorage.getItem('skinlab-model'); select.value = data.models.some(row => row.id === savedModel && row.ready) ? savedModel : data.default_model; await loadModel(); const saved = localStorage.getItem('skinlab-job'); if (saved) { emptyCards(true); await poll(saved); } else { const history = await api('/api/history'); if (history.results.length) await renderResults(history.results); } } catch (error) { status(error.message, 'error'); $('#generate-button').disabled = true; } }
init();
