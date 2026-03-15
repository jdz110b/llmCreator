/**
 * 全局提示消息
 */
function showAlert(message, type) {
    type = type || 'info';
    var el = document.getElementById('global-alert');
    el.className = 'alert alert-' + type;
    el.textContent = message;
    el.classList.remove('d-none');
    setTimeout(function() {
        el.classList.add('d-none');
    }, 5000);
}
