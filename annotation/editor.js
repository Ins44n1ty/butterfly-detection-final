(function (global) {
  'use strict';
  function mount(root, input, saved, save) {
    const image = root.querySelector('.bf-image');
    const viewport = root.querySelector('.bf-viewport');
    const surface = root.querySelector('.bf-surface');
    const svg = root.querySelector('.bf-overlay');
    const status = root.querySelector('.bf-status');
    const loader = root.querySelector('.bf-loading');
    const loadError = root.querySelector('.bf-load-error');
    let boxes = JSON.parse(JSON.stringify(saved.boxes || []));
    let answer = saved.status || '';
    let selected = -1;
    let loaded = false;
    let failed = false;
    let gesture = null;
    const history = [];
    const copy = () => JSON.parse(JSON.stringify({ boxes, status: answer }));
    const value = () => ({ status: answer, boxes: answer === 'empty' ? [] : JSON.parse(JSON.stringify(boxes)) });
    function remember() { history.push(copy()); }
    function message(text, error) {
      status.textContent = text;
      status.classList.toggle('error', !!error);
    }
    function valid() {
      return loaded && (answer === 'empty' || (answer === 'boxes' && boxes.length > 0)) || answer === 'load_error' && failed && boxes.length === 0;
    }
    function emit() {
      save(value());
      paint();
    }
    function paint() {
      svg.replaceChildren();
      root.querySelector('.bf-count').textContent = answer === 'empty' ? 'Без бабочек' : 'Рамок: ' + boxes.length;
      root.classList.toggle('bf-empty-state', answer === 'empty');
      root.querySelector('[data-action="empty"]').setAttribute('aria-pressed', String(answer === 'empty'));
      root.querySelector('[data-action="empty"]').disabled = !loaded;
      root.querySelector('[data-action="delete"]').disabled = selected < 0 || answer === 'empty';
      root.querySelector('[data-action="undo"]').disabled = !history.length;
      (answer === 'empty' ? [] : boxes).forEach((b, index) => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        for (const [key, value] of Object.entries({x:b.left * 1000, y:b.top * 1000, width:b.width * 1000, height:b.height * 1000, fill:'#00b8a010', stroke:index === selected ? '#ffb000' : '#00aa88', 'stroke-width':2, 'vector-effect':'non-scaling-stroke', 'data-index':index,cursor:'move'})) {
          rect.setAttribute(key, value);
        }
        svg.appendChild(rect);
        if (index === selected) {
          for (const [corner, x, y] of [['nw',b.left,b.top],['ne',b.left+b.width,b.top],['sw',b.left,b.top+b.height],['se',b.left+b.width,b.top+b.height]]) {
            const handle = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            const size = 10 / surface.clientWidth * 1000;
            const height = 10 / surface.clientHeight * 1000;
            for (const [key,value] of Object.entries({x:x*1000-size/2,y:y*1000-height/2,width:size,height,fill:'#ffb000','data-index':index,'data-corner':corner})) { handle.setAttribute(key,value); }
            svg.appendChild(handle);
          }
        }
      });
    }
    function fit() {
      if (!loaded) { return; }
      const ratio = Math.min(viewport.clientWidth / image.naturalWidth, viewport.clientHeight / image.naturalHeight);
      surface.style.width = Math.max(1, image.naturalWidth * ratio) + 'px';
      surface.style.height = Math.max(1, image.naturalHeight * ratio) + 'px';
      paint();
    }
    function point(event) {
      const r = svg.getBoundingClientRect();
      return { x:Math.min(1,Math.max(0,(event.clientX-r.left)/r.width)), y:Math.min(1,Math.max(0,(event.clientY-r.top)/r.height)) };
    }
    function action(name) {
      if (name === 'delete' && selected >= 0 && answer !== 'empty') {
        remember(); boxes.splice(selected,1); selected = -1; answer = boxes.length ? 'boxes' : ''; emit();
      }
      if (name === 'undo' && history.length) {
        const old = history.pop(); boxes = old.boxes; answer = old.status; selected = -1; emit();
      }
      if (name === 'empty' && loaded) {
        remember(); answer = answer === 'empty' ? (boxes.length ? 'boxes' : '') : 'empty'; selected = -1; emit();
        message(answer === 'empty' ? (boxes.length ? 'Выбрано: бабочек нет. Повторный клик вернёт рамки' : 'Выбрано: бабочек нет. Можно отправлять') : (boxes.length ? 'Рамки восстановлены. Можно продолжать разметку' : 'Обведите бабочек или отметьте, что их нет'));
      }
      if (name === 'error' && failed) {
        if (loaded) { message('Фото загрузилось. Разметьте его или выберите «Бабочек нет»',true); return; }
        remember(); boxes = []; answer = 'load_error'; emit(); message('Ошибка загрузки отмечена. Можно отправлять');
      }
      if (name === 'retry') { failed = false; loaded = false; loader.hidden = false; loadError.hidden = true; image.src = input.image_url; paint(); }
    }
    const buttons = Array.from(root.querySelectorAll('[data-action]'));
    const buttonHandlers = buttons.map(button => {
      const handler = () => action(button.dataset.action);
      button.addEventListener('click',handler);
      return handler;
    });
    function down(event) {
      if (!loaded || answer === 'empty' || event.button !== 0) { return; }
      root.focus({preventScroll:true});
      const start = point(event);
      const index = event.target.getAttribute('data-index');
      if (index !== null && !event.shiftKey) {
        selected = index === null ? -1 : Number(index);
        if (selected < 0) { paint(); return; }
        remember();
        gesture = { start, index:selected, before:{...boxes[selected]}, corner:event.target.getAttribute('data-corner') };
      } else {
        remember(); answer = 'boxes';
        boxes.push({ left:start.x, top:start.y, width:0, height:0 });
        selected = boxes.length-1;
        gesture = { start, index:selected, drawing:true };
      }
      svg.setPointerCapture(event.pointerId);
      paint();
      event.preventDefault();
    }
    function move(event) {
      if (!gesture) { return; }
      const p = point(event);
      const g = gesture;
      let x1, y1, x2, y2;
      if (g.drawing) { x1=g.start.x; y1=g.start.y; x2=p.x; y2=p.y; }
      else if (g.corner) {
        const b=g.before;
        x1=g.corner.includes('w') ? p.x : b.left;
        x2=g.corner.includes('e') ? p.x : b.left+b.width;
        y1=g.corner.includes('n') ? p.y : b.top;
        y2=g.corner.includes('s') ? p.y : b.top+b.height;
      } else {
        const b=g.before;
        const left=Math.max(0,Math.min(1-b.width,b.left+p.x-g.start.x));
        const top=Math.max(0,Math.min(1-b.height,b.top+p.y-g.start.y));
        boxes[g.index]={...b,left,top}; paint(); return;
      }
      boxes[g.index]={left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1)};
      paint();
    }
    function up() {
      if (!gesture) { return; }
      const b=boxes[gesture.index];
      if (b.width*surface.clientWidth < 3 || b.height*surface.clientHeight < 3) {
        const old=history.pop(); boxes=old.boxes; answer=old.status;
        selected=-1;
      }
      gesture=null;
      emit();
      message(valid() ? 'Разметка готова. Проверьте весь кадр и отправьте ответ' : 'Обведите бабочек или нажмите «На фото нет бабочек»');
    }
    function key(event) {
      if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) { return; }
      const key=event.key.toLowerCase();
      const map={n:'empty','т':'empty',delete:'delete',backspace:'delete'};
      const name=(event.ctrlKey || event.metaKey) && (key==='z' || key==='я') ? 'undo' : map[key];
      if (name) { event.preventDefault(); action(name); }
    }
    const resize = new ResizeObserver(fit);
    resize.observe(viewport);
    root.addEventListener('keydown',key);
    svg.addEventListener('pointerdown',down);
    svg.addEventListener('pointermove',move);
    svg.addEventListener('pointerup',up);
    svg.addEventListener('pointercancel',up);
    image.onload=()=>{
      loaded=true; failed=false; loader.hidden=true; loadError.hidden=true;
      if (answer==='load_error') { answer=''; emit(); }
      fit(); message(valid() ? 'Ответ восстановлен. Можно отправлять' : 'Рисуйте рамки прямо на фото. Моль тоже считается бабочкой — обведите её. «На фото нет бабочек» выбирайте, только если нет ни бабочек, ни молей');
    };
    image.onerror=()=>{ loaded=false; failed=true; loader.hidden=true; loadError.hidden=false; paint(); message('Фото не открылось. Это не значит, что бабочек нет',true); };
    image.src=input.image_url;
    paint();
    return { valid, value, action, destroy() { resize.disconnect(); root.removeEventListener('keydown',key); image.onload=null; image.onerror=null; buttons.forEach((b,i)=>b.removeEventListener('click',buttonHandlers[i])); svg.removeEventListener('pointerdown',down); svg.removeEventListener('pointermove',move); svg.removeEventListener('pointerup',up); svg.removeEventListener('pointercancel',up); } };
  }
  global.ButterflyEditor={mount};
})(window);
