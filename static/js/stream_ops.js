// === 批量执行选中的操作（流式版本）===
function runBatchOperations() {
    var configId = document.getElementById('config-select').value;
    if (!configId) { showAlert('请先选择大模型', 'danger'); return; }

    var checkedOps = [];
    document.querySelectorAll('.op-check:checked').forEach(function(cb) {
        checkedOps.push(cb.value);
    });

    if (checkedOps.length === 0) {
        showAlert('请至少选择一个操作', 'danger');
        return;
    }

    var categories = document.getElementById('custom-categories').value;
    if (checkedOps.indexOf('category') !== -1 && !categories.trim()) {
        showAlert('自定义分类需要输入分类类型', 'danger');
        return;
    }

    var selectedIds = getSelectedIds();
    var promptId = document.getElementById('prompt-select').value;

    // 分离 score 和分类操作
    var classifyOps = [];
    var hasScore = false;
    checkedOps.forEach(function(op) {
        if (op === 'score') {
            hasScore = true;
        } else {
            classifyOps.push(op);
        }
    });

    showProgress('准备中...');
    disableButtons();

    // 使用流式 API
    if (classifyOps.length > 0) {
        var eventSource = new EventSourcePolyfill('/api/classify-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                file_id: FILE_ID,
                config_id: parseInt(configId),
                classify_type: 'combined',
                classify_types: classifyOps,
                categories: categories,
                prompt_id: promptId ? parseInt(promptId) : null,
                item_ids: selectedIds.length > 0 ? selectedIds : [],
            })
        });

        eventSource.onmessage = function(event) {
            var data = JSON.parse(event.data);
            if (data.error) {
                showAlert(data.error, 'danger');
                eventSource.close();
                enableButtons();
                hideProgress();
                return;
            }

            // 更新进度条
            var percent = Math.round((data.progress / data.total) * 100);
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-text').textContent = 
                '正在处理第 ' + data.progress + '/' + data.total + ' 条...';

            // 实时更新表格行
            if (data.status === 'ok' && data.data) {
                updateTableRow(data.item_id, data.data);
            } else if (data.status === 'error') {
                console.error('处理失败:', data.item_id, data.error);
            }

            // 完成
            if (data.progress === data.total) {
                eventSource.close();
                enableButtons();
                hideProgress();
                showAlert('分类完成!', 'success');
                // 最后刷新页面确保一致性
                setTimeout(function() { location.reload(); }, 1000);
            }
        };

        eventSource.onerror = function(err) {
            console.error('SSE Error:', err);
            eventSource.close();
            enableButtons();
            hideProgress();
            showAlert('连接中断，请重试', 'danger');
        };
    }
}

function updateTableRow(itemId, data) {
    // 根据返回的数据更新对应行的显示
    // 这里可以根据实际需要更新 DOM
    // 简单起见，我们让页面最后整体刷新
}

function disableButtons() {
    document.querySelector('button[onclick="runBatchOperations()"]').disabled = true;
}

function enableButtons() {
    document.querySelector('button[onclick="runBatchOperations()"]').disabled = false;
}

// EventSource polyfill for older browsers
var EventSourcePolyfill = window.EventSource || function(url, options) {
    // 简化实现，实际项目建议引入完整的 polyfill
    var xhr = new XMLHttpRequest();
    xhr.open(options.method || 'GET', url, true);
    for (var key in options.headers) {
        xhr.setRequestHeader(key, options.headers[key]);
    }
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                var lines = xhr.responseText.split('\n');
                lines.forEach(function(line) {
                    if (line.startsWith('data: ')) {
                        var data = line.substring(6);
                        if (this.onmessage) {
                            this.onmessage({data: data});
                        }
                    }
                }.bind(this));
            }
        }
    }.bind(this);
    if (options.body) {
        xhr.send(options.body);
    } else {
        xhr.send();
    }
    this.close = function() { xhr.abort(); };
};
