
exports.Task = extend(TolokaHandlebarsTask, function (options) {
  TolokaHandlebarsTask.call(this, options);
}, {
  onRender: function () {
    var task = this.getTask();
    var input = (task && task.input_values) ? task.input_values : {};
    var boxes = input.boxes || [];
    if (typeof boxes === 'string') {
      try { boxes = JSON.parse(boxes); } catch (e) { boxes = []; }
    }

    var root = this.getDOMElement();
    var overlay = root.querySelector('.bbox-overlay-layer');
    var img = root.querySelector('.review-main-img');
    var errBanner = root.querySelector('.img-load-error');

    if (img && errBanner) {
      img.onerror = function () {
        errBanner.style.display = 'block';
      };
    }

    if (overlay && Array.isArray(boxes)) {
      overlay.innerHTML = '';
      for (var i = 0; i < boxes.length; i++) {
        var b = boxes[i];
        if (!b || b.left === undefined) continue;
        var rect = document.createElement('div');
        rect.className = 'bbox-rect';
        rect.style.left = (Number(b.left) * 100).toFixed(4) + '%';
        rect.style.top = (Number(b.top) * 100).toFixed(4) + '%';
        rect.style.width = (Number(b.width) * 100).toFixed(4) + '%';
        rect.style.height = (Number(b.height) * 100).toFixed(4) + '%';

        var tag = document.createElement('span');
        tag.className = 'bbox-tag';
        tag.textContent = '#' + (i + 1);
        rect.appendChild(tag);
        overlay.appendChild(rect);
      }
    }
  }
});

function extend(ParentClass, constructorFunction, prototypeHash) {
  constructorFunction = constructorFunction || function () {};
  prototypeHash = prototypeHash || {};
  if (ParentClass) {
    constructorFunction.prototype = Object.create(ParentClass.prototype);
  }
  for (var i in prototypeHash) {
    constructorFunction.prototype[i] = prototypeHash[i];
  }
  return constructorFunction;
}
