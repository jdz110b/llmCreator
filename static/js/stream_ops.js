// === 流式操作模块（支持并行调用、暂停/恢复） ===

// 任务状态管理
var _abortController = null;       // 当前 AbortController
var _processedItemIds = new Set(); // 已处理的 item IDs（跨暂停保留）
var _allItemIds = [];              // 全部 item IDs（从 init 事件获取）
var _taskState = 'idle';           // idle | running | paused
var _taskConfig = {};              // 当前任务配置（恢复时复用）
var _progressBase = 0;             // 恢复时的进度基数
var _allErrors = [];               // 错误收集
var _currentPhase = '';            // 当前阶段: 'classify' | 'score'

// 获取并行数配置
function getConcurrency() {
    var sel = document.getElementById('concurrency-select');
    return sel ? parseInt(sel.value) || 3 : 3;
}

// === 批量执行选中的操作（流式版本，支持暂停/恢复） ===
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
    var concurrency = getConcurrency();

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

    // 重置状态
    _processedItemIds = new Set();
    _allItemIds = [];
    _progressBase = 0;
    _allErrors = [];
    _taskState = 'running';

    // 保存配置（用于恢复）
    _taskConfig = {
        configId: configId,
        classifyOps: classifyOps,
        hasScore: hasScore,
        categories: categories,
        promptId: promptId,
        selectedIds: selectedIds,
        concurrency: concurrency
    };

    showProgress('准备中...');
    disableButtons();
    showPauseBtn();

    // 串行执行：先分类，再打分
    _doClassifyPhase(classifyOps, hasScore);
}

function _doClassifyPhase(classifyOps, hasScore) {
    if (classifyOps.length === 0) {
        if (hasScore) {
            _currentPhase = 'score';
            _processedItemIds = new Set();
            _allItemIds = [];
            _progressBase = 0;
            _doScorePhase();
        } else {
            _onAllDone();
        }
        return;
    }

    _currentPhase = 'classify';
    var cfg = _taskConfig;

    _abortController = new AbortController();

    var requestBody = {
        file_id: FILE_ID,
        config_id: parseInt(cfg.configId),
        classify_type: classifyOps.length > 1 ? 'combined' : classifyOps[0],
        classify_types: classifyOps,
        categories: cfg.categories,
        prompt_id: cfg.promptId ? parseInt(cfg.promptId) : null,
        item_ids: cfg.selectedIds.length > 0 ? cfg.selectedIds : [],
        concurrency: cfg.concurrency
    };

    // 恢复时用剩余 item_ids
    if (_allItemIds.length > 0) {
        var remaining = _allItemIds.filter(function(id) { return !_processedItemIds.has(id); });
        if (remaining.length === 0) {
            if (hasScore) {
                _currentPhase = 'score';
                _processedItemIds = new Set();
                _allItemIds = [];
                _progressBase = 0;
                _doScorePhase();
            } else {
                _onAllDone();
            }
            return;
        }
        requestBody.item_ids = remaining;
    }

    _streamFetch('/api/classify-stream', requestBody, {
        onInit: function(data) {
            if (_allItemIds.length === 0) {
                _allItemIds = data.item_ids || [];
            }
        },
        onProgress: function(data) {
            var processed = _processedItemIds.size;
            var total = _allItemIds.length || data.total;
            var percent = Math.round((processed / total) * 100);
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-text').textContent =
                '分类中 ' + processed + '/' + total + ' 条...';
        },
        onOk: function(data) {
            _processedItemIds.add(data.item_id);
            if (data.data) updateTableRow(data.item_id, data.data);
        },
        onError: function(data) {
            _processedItemIds.add(data.item_id);
            _allErrors.push(data);
            console.error('[分类流] 条目处理失败:', data.item_id, data.error);
        },
        onDone: function() {
            if (_taskConfig.hasScore) {
                _currentPhase = 'score';
                _processedItemIds = new Set();
                _allItemIds = [];
                _progressBase = 0;
                _doScorePhase();
            } else {
                _onAllDone();
            }
        },
        onAbort: function() {
            // 暂停时不做额外处理，状态已在 pauseTask 中设置
        },
        onFatalError: function(err) {
            _onFatalError(err);
        }
    });
}

function _doScorePhase() {
    var cfg = _taskConfig;

    _abortController = new AbortController();

    var requestBody = {
        file_id: FILE_ID,
        config_id: parseInt(cfg.configId),
        prompt_id: cfg.promptId ? parseInt(cfg.promptId) : null,
        item_ids: cfg.selectedIds.length > 0 ? cfg.selectedIds : [],
        concurrency: cfg.concurrency
    };

    // 恢复时用剩余 item_ids
    if (_allItemIds.length > 0) {
        var remaining = _allItemIds.filter(function(id) { return !_processedItemIds.has(id); });
        if (remaining.length === 0) {
            _onAllDone();
            return;
        }
        requestBody.item_ids = remaining;
    }

    _streamFetch('/api/score-stream', requestBody, {
        onInit: function(data) {
            if (_allItemIds.length === 0) {
                _allItemIds = data.item_ids || [];
            }
        },
        onProgress: function(data) {
            var processed = _processedItemIds.size;
            var total = _allItemIds.length || data.total;
            var percent = Math.round((processed / total) * 100);
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-text').textContent =
                '打分中 ' + processed + '/' + total + ' 条...';
        },
        onOk: function(data) {
            _processedItemIds.add(data.item_id);
            if (data.data) updateTableRow(data.item_id, data.data);
        },
        onError: function(data) {
            _processedItemIds.add(data.item_id);
            _allErrors.push(data);
            console.error('[打分流] 条目处理失败:', data.item_id, data.error);
        },
        onDone: function() {
            _onAllDone();
        },
        onAbort: function() {},
        onFatalError: function(err) {
            _onFatalError(err);
        }
    });
}

// 通用 SSE 流式请求
function _streamFetch(url, body, callbacks) {
    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: _abortController.signal
    }).then(function(response) {
        if (!response.ok) {
            throw new Error('服务器返回错误 (HTTP ' + response.status + ')');
        }
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        function processChunk(result) {
            if (result.done) {
                if (buffer.trim()) processLines(buffer);
                return;
            }
            buffer += decoder.decode(result.value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop();
            processLines(lines.join('\n'));
            reader.read().then(processChunk).catch(function(err) {
                if (err.name === 'AbortError') {
                    callbacks.onAbort();
                } else {
                    callbacks.onFatalError(err);
                }
            });
        }

        function processLines(text) {
            var lines = text.split('\n');
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i].trim();
                if (!line.startsWith('data: ')) continue;
                try {
                    var data = JSON.parse(line.substring(6));
                    if (data.error && !data.status) {
                        showAlert(data.error, 'danger');
                        console.error('[SSE] 致命错误:', data.error);
                        return;
                    }
                    if (data.status === 'init') {
                        callbacks.onInit(data);
                        continue;
                    }
                    if (data.status === 'done') {
                        console.log('[SSE] 完成:', data.message);
                        callbacks.onDone();
                        return;
                    }
                    if (data.status === 'ok') {
                        callbacks.onOk(data);
                        callbacks.onProgress(data);
                    } else if (data.status === 'error') {
                        callbacks.onError(data);
                        callbacks.onProgress(data);
                    }
                } catch (e) {
                    console.error('[SSE] 解析数据失败:', line, e);
                }
            }
        }

        reader.read().then(processChunk).catch(function(err) {
            if (err.name === 'AbortError') {
                callbacks.onAbort();
            } else {
                callbacks.onFatalError(err);
            }
        });
    }).catch(function(err) {
        if (err.name === 'AbortError') {
            callbacks.onAbort();
        } else {
            callbacks.onFatalError(err);
        }
    });
}

// 操作完成
function _onAllDone() {
    _taskState = 'idle';
    enableButtons();
    hideProgress();
    hidePauseBtn();
    hideResumeBtn();
    var cfg = _taskConfig;
    var opNames = cfg.classifyOps.length > 0 ? cfg.classifyOps.join('+') : '';
    if (cfg.hasScore) opNames += (opNames ? '+' : '') + '打分';
    var msg = '操作完成 (' + opNames + ')';
    if (_allErrors.length > 0) {
        msg += ', ' + _allErrors.length + ' 条处理失败';
    }
    showAlert(msg, _allErrors.length > 0 ? 'warning' : 'success');
    setTimeout(function() { location.reload(); }, 1500);
}

// 致命错误
function _onFatalError(err) {
    _taskState = 'idle';
    enableButtons();
    hideProgress();
    hidePauseBtn();
    hideResumeBtn();
    console.error('[批量操作] 失败:', err);
    showAlert('操作失败: ' + (err.message || err), 'danger');
}

// === 暂停任务 ===
function pauseTask() {
    if (_taskState !== 'running') return;
    _taskState = 'paused';
    if (_abortController) {
        _abortController.abort();
    }
    hidePauseBtn();
    showResumeBtn();
    var processed = _processedItemIds.size;
    var total = _allItemIds.length || '?';
    document.getElementById('progress-text').textContent =
        '已暂停 (' + processed + '/' + total + ')';
    // 停止进度条动画
    var bar = document.getElementById('progress-bar');
    bar.classList.remove('progress-bar-animated');
}

// === 恢复任务 ===
function resumeTask() {
    if (_taskState !== 'paused') return;
    _taskState = 'running';
    hideResumeBtn();
    showPauseBtn();
    // 恢复进度条动画
    var bar = document.getElementById('progress-bar');
    bar.classList.add('progress-bar-animated');

    if (_currentPhase === 'classify') {
        _doClassifyPhase(_taskConfig.classifyOps, _taskConfig.hasScore);
    } else if (_currentPhase === 'score') {
        _doScorePhase();
    }
}

// === UI 辅助函数 ===
function updateTableRow(itemId, data) {
    // 根据返回的数据更新对应行的显示
    // 页面最后会整体刷新
}

function disableButtons() {
    var btn = document.getElementById('btn-run');
    if (!btn) btn = document.querySelector('button[onclick="runBatchOperations()"]');
    if (btn) btn.disabled = true;
}

function enableButtons() {
    var btn = document.getElementById('btn-run');
    if (!btn) btn = document.querySelector('button[onclick="runBatchOperations()"]');
    if (btn) btn.disabled = false;
}

function showPauseBtn() {
    var btn = document.getElementById('btn-pause');
    if (btn) btn.classList.remove('d-none');
}

function hidePauseBtn() {
    var btn = document.getElementById('btn-pause');
    if (btn) btn.classList.add('d-none');
}

function showResumeBtn() {
    var btn = document.getElementById('btn-resume');
    if (btn) btn.classList.remove('d-none');
}

function hideResumeBtn() {
    var btn = document.getElementById('btn-resume');
    if (btn) btn.classList.add('d-none');
}
