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

    var allErrors = [];

    function doClassify() {
        if (classifyOps.length === 0) return Promise.resolve();

        return new Promise(function(resolve, reject) {
            fetch('/api/classify-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_id: FILE_ID,
                    config_id: parseInt(configId),
                    classify_type: classifyOps.length > 1 ? 'combined' : classifyOps[0],
                    classify_types: classifyOps,
                    categories: categories,
                    prompt_id: promptId ? parseInt(promptId) : null,
                    item_ids: selectedIds.length > 0 ? selectedIds : [],
                })
            }).then(function(response) {
                if (!response.ok) {
                    throw new Error('服务器返回错误 (HTTP ' + response.status + ')');
                }
                var reader = response.body.getReader();
                var decoder = new TextDecoder();
                var buffer = '';

                function processChunk(result) {
                    if (result.done) {
                        // 处理缓冲区中剩余的数据
                        if (buffer.trim()) {
                            processLines(buffer);
                        }
                        resolve();
                        return;
                    }

                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop(); // 保留未完成的行

                    processLines(lines.join('\n'));
                    reader.read().then(processChunk).catch(reject);
                }

                function processLines(text) {
                    var lines = text.split('\n');
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i].trim();
                        if (!line.startsWith('data: ')) continue;
                        try {
                            var data = JSON.parse(line.substring(6));
                            if (data.error) {
                                showAlert(data.error, 'danger');
                                console.error('[分类流] 错误:', data.error);
                                return;
                            }
                            if (data.status === 'done') {
                                console.log('[分类流] 完成:', data.message);
                                return;
                            }
                            if (data.progress && data.total) {
                                var percent = Math.round((data.progress / data.total) * 100);
                                document.getElementById('progress-bar').style.width = percent + '%';
                                document.getElementById('progress-text').textContent =
                                    '正在处理第 ' + data.progress + '/' + data.total + ' 条...';
                            }
                            if (data.status === 'ok' && data.data) {
                                updateTableRow(data.item_id, data.data);
                            } else if (data.status === 'error') {
                                console.error('[分类流] 条目处理失败:', data.item_id, data.error);
                                allErrors.push(data);
                            }
                        } catch (e) {
                            console.error('[分类流] 解析 SSE 数据失败:', line, e);
                        }
                    }
                }

                reader.read().then(processChunk).catch(reject);
            }).catch(reject);
        });
    }

    function doScore() {
        if (!hasScore) return Promise.resolve();

        return new Promise(function(resolve, reject) {
            fetch('/api/score-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_id: FILE_ID,
                    config_id: parseInt(configId),
                    prompt_id: promptId ? parseInt(promptId) : null,
                    item_ids: selectedIds.length > 0 ? selectedIds : [],
                })
            }).then(function(response) {
                if (!response.ok) {
                    // 回退到非流式打分接口
                    return fetch('/api/score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            file_id: FILE_ID,
                            config_id: parseInt(configId),
                            prompt_id: promptId ? parseInt(promptId) : null,
                            item_ids: selectedIds.length > 0 ? selectedIds : [],
                        })
                    }).then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.errors) allErrors = allErrors.concat(data.errors);
                        resolve();
                    });
                }
                return response.json().then(function(data) {
                    if (data.errors) allErrors = allErrors.concat(data.errors);
                    resolve();
                });
            }).catch(function() {
                // 回退到非流式打分
                fetch('/api/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_id: FILE_ID,
                        config_id: parseInt(configId),
                        prompt_id: promptId ? parseInt(promptId) : null,
                        item_ids: selectedIds.length > 0 ? selectedIds : [],
                    })
                }).then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.errors) allErrors = allErrors.concat(data.errors);
                    resolve();
                }).catch(reject);
            });
        });
    }

    // 串行执行：先分类，再打分
    doClassify()
        .then(doScore)
        .then(function() {
            enableButtons();
            hideProgress();
            var opNames = classifyOps.length > 0 ? classifyOps.join('+') : '';
            if (hasScore) opNames += (opNames ? '+' : '') + '打分';
            var msg = '操作完成 (' + opNames + ')';
            if (allErrors.length > 0) {
                msg += ', ' + allErrors.length + ' 条处理失败';
            }
            showAlert(msg, allErrors.length > 0 ? 'warning' : 'success');
            setTimeout(function() { location.reload(); }, 1500);
        })
        .catch(function(err) {
            enableButtons();
            hideProgress();
            console.error('[批量操作] 失败:', err);
            showAlert('操作失败: ' + (err.message || err), 'danger');
        });
}

function updateTableRow(itemId, data) {
    // 根据返回的数据更新对应行的显示
    // 页面最后会整体刷新
}

function disableButtons() {
    var btn = document.querySelector('button[onclick="runBatchOperations()"]');
    if (btn) btn.disabled = true;
}

function enableButtons() {
    var btn = document.querySelector('button[onclick="runBatchOperations()"]');
    if (btn) btn.disabled = false;
}
