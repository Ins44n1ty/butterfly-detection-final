exports.Task = function (options) {
  TolokaHandlebarsTask.call(this, options);
};
exports.Task.prototype = Object.create(TolokaHandlebarsTask.prototype);
exports.Task.prototype.constructor = exports.Task;
exports.Task.prototype.onRender = function () {
  var saved = this.getSolution().output_values || {};
  this.editor = window.ButterflyEditor.mount(this.getDOMElement().querySelector('.bf-task'), this.getTask().input_values, saved, function (value) {
    this.setSolutionOutputValue('boxes', value.boxes);
    this.setSolutionOutputValue('status', value.status);
    this.setSolution(this.getSolution());
  }.bind(this));
};
exports.Task.prototype.validate = function (solution) {
  if (!this.editor || !this.editor.valid()) {
    return { task_id:this.getTask().id, errors:{__TASK__:{code:'VALUE_REQUIRED',message:'Обведите бабочек или нажмите «На фото нет бабочек»'}} };
  }
  return TolokaHandlebarsTask.prototype.validate.call(this, solution);
};
exports.Task.prototype.onDestroy = function () {
  if (this.editor) { this.editor.destroy(); }
};
exports.Assignment = function (options) {
  TolokaAssignment.call(this, options);
};
exports.Assignment.prototype = Object.create(TolokaAssignment.prototype);
exports.Assignment.prototype.constructor = exports.Assignment;
exports.Assignment.prototype.onValidationDone = function (event) {
  if (event && event.data) {
    return TolokaAssignment.prototype.onValidationDone.call(this, event);
  }
};
