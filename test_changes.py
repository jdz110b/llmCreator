"""
测试脚本：验证并行调用、重试、暂停恢复等改动点
运行方式: cd 到项目根目录，然后执行 python test_changes.py
"""
import sys
import time
import threading
import traceback

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1


# ============================================================
# 1. 测试 config.py 新增配置项
# ============================================================
print("\n=== 1. 测试 config.py ===")

def test_config_concurrency():
    from config import LLM_CONCURRENCY
    assert LLM_CONCURRENCY == 3, f"期望 3，实际 {LLM_CONCURRENCY}"

def test_config_retries():
    from config import LLM_MAX_RETRIES
    assert LLM_MAX_RETRIES == 1, f"期望 1，实际 {LLM_MAX_RETRIES}"

test("LLM_CONCURRENCY 存在且为 3", test_config_concurrency)
test("LLM_MAX_RETRIES 存在且为 1", test_config_retries)


# ============================================================
# 2. 测试 task_executor.py 并行批处理
# ============================================================
print("\n=== 2. 测试 task_executor (并行执行器) ===")

def test_executor_basic():
    from services.task_executor import execute_parallel_batch
    items = [1, 2, 3, 4, 5]
    results = list(execute_parallel_batch(items, lambda x: x * 10, concurrency=2))
    assert len(results) == 5, f"期望 5 个结果，实际 {len(results)}"
    for idx, item, result, error in results:
        assert error is None, f"item {item} 不应有错误"
        assert result == item * 10, f"item {item} 期望 {item*10}，实际 {result}"

def test_executor_order():
    """验证结果按原始顺序返回"""
    from services.task_executor import execute_parallel_batch
    import random
    def slow_fn(x):
        time.sleep(random.uniform(0.01, 0.05))
        return x
    items = list(range(10))
    results = list(execute_parallel_batch(items, slow_fn, concurrency=3))
    indices = [r[0] for r in results]
    assert indices == list(range(1, 11)), f"顺序不正确: {indices}"

def test_executor_error_handling():
    """验证异常被正确捕获"""
    from services.task_executor import execute_parallel_batch
    def fail_fn(x):
        if x == 3:
            raise ValueError("模拟失败")
        return x
    items = [1, 2, 3, 4, 5]
    results = list(execute_parallel_batch(items, fail_fn, concurrency=3))
    assert len(results) == 5
    for idx, item, result, error in results:
        if item == 3:
            assert error is not None, "item 3 应该有错误"
            assert result is None, "item 3 result 应为 None"
        else:
            assert error is None, f"item {item} 不应有错误"
            assert result == item

def test_executor_concurrency():
    """验证确实是并行执行"""
    from services.task_executor import execute_parallel_batch
    timestamps = {}
    lock = threading.Lock()
    def record_fn(x):
        with lock:
            timestamps[x] = time.time()
        time.sleep(0.2)
        return x
    items = [1, 2, 3]
    start = time.time()
    list(execute_parallel_batch(items, record_fn, concurrency=3))
    elapsed = time.time() - start
    # 3 个任务并行执行，总时间应 < 0.5s（串行需要 0.6s）
    assert elapsed < 0.5, f"并行执行耗时 {elapsed:.2f}s，期望 < 0.5s（可能未并行）"

def test_executor_single():
    """验证 concurrency=1 时退化为串行"""
    from services.task_executor import execute_parallel_batch
    items = [1, 2, 3]
    start = time.time()
    def slow_fn(x):
        time.sleep(0.1)
        return x
    list(execute_parallel_batch(items, slow_fn, concurrency=1))
    elapsed = time.time() - start
    assert elapsed >= 0.25, f"串行耗时 {elapsed:.2f}s，期望 >= 0.25s"

test("基本功能：处理结果正确", test_executor_basic)
test("顺序保证：结果按原始顺序返回", test_executor_order)
test("异常处理：错误被正确捕获", test_executor_error_handling)
test("并行验证：3个任务确实并行执行", test_executor_concurrency)
test("串行退化：concurrency=1 时串行执行", test_executor_single)


# ============================================================
# 3. 测试 llm_service.py 重试逻辑
# ============================================================
print("\n=== 3. 测试 LLMService 重试逻辑 ===")

def test_chat_retry_signature():
    """验证 chat() 接受 max_retries 参数"""
    from services.llm_service import LLMService
    import inspect
    sig = inspect.signature(LLMService.chat)
    params = list(sig.parameters.keys())
    assert 'max_retries' in params, f"chat() 缺少 max_retries 参数，现有: {params}"

def test_chat_json_retry_signature():
    """验证 chat_json() 接受 max_retries 参数"""
    from services.llm_service import LLMService
    import inspect
    sig = inspect.signature(LLMService.chat_json)
    params = list(sig.parameters.keys())
    assert 'max_retries' in params, f"chat_json() 缺少 max_retries 参数，现有: {params}"

def test_chat_retry_behavior():
    """验证 chat() 在失败时重试"""
    from services.llm_service import LLMService
    llm = LLMService(api_url="http://127.0.0.1:1/v1", api_key="test", model="test")
    call_count = [0]
    original_post = llm.session.post
    def mock_post(*args, **kwargs):
        call_count[0] += 1
        raise ConnectionError("模拟连接失败")
    llm.session.post = mock_post
    try:
        llm.chat("sys", "user", max_retries=1)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "连接" in str(e) or "connect" in str(e).lower(), f"错误信息不包含连接相关: {e}"
    # max_retries=1 意味着最多 2 次调用（1 次初始 + 1 次重试）
    assert call_count[0] == 2, f"期望调用 2 次（1次初始+1次重试），实际 {call_count[0]} 次"

def test_chat_json_retry_no_double():
    """验证 chat_json 调 chat 时传 max_retries=0 避免重复重试"""
    from services.llm_service import LLMService
    import inspect
    source = inspect.getsource(LLMService.chat_json)
    assert 'max_retries=0' in source, "chat_json 中应该用 max_retries=0 调用 chat"

test("chat() 签名含 max_retries 参数", test_chat_retry_signature)
test("chat_json() 签名含 max_retries 参数", test_chat_json_retry_signature)
test("chat() 失败时重试 1 次", test_chat_retry_behavior)
test("chat_json() 调用 chat 时传 max_retries=0", test_chat_json_retry_no_double)


# ============================================================
# 4. 测试 app.py 端点注册和导入
# ============================================================
print("\n=== 4. 测试 app.py 导入和端点注册 ===")

def test_app_import():
    """验证 app.py 能正常导入（所有依赖都正确）"""
    import app as main_app
    assert hasattr(main_app, 'app'), "app.py 中应有 app 对象"

def test_endpoints_registered():
    """验证关键端点已注册"""
    import app as main_app
    rules = {rule.rule for rule in main_app.app.url_map.iter_rules()}
    expected = [
        '/api/classify-stream',
        '/api/similarity-compare-stream',
        '/api/score-stream',
        '/api/score',
    ]
    for ep in expected:
        assert ep in rules, f"端点 {ep} 未注册。已注册: {sorted(rules)}"

def test_classify_stream_accepts_concurrency():
    """验证 classify-stream 端点读取 concurrency 参数"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.classify_items_stream)
    assert 'concurrency' in source, "classify_items_stream 中应使用 concurrency"
    assert 'execute_parallel_batch' in source, "classify_items_stream 中应使用 execute_parallel_batch"

def test_score_stream_exists():
    """验证 score-stream 端点存在"""
    import app as main_app
    assert hasattr(main_app, 'score_items_stream'), "app.py 中应有 score_items_stream 函数"

test("app.py 能正常导入", test_app_import)
test("关键端点已注册", test_endpoints_registered)
test("classify-stream 使用并行执行", test_classify_stream_accepts_concurrency)
test("score-stream 端点存在", test_score_stream_exists)


# ============================================================
# 5. 测试 Flask 端点请求响应（使用 test client）
# ============================================================
print("\n=== 5. 测试端点请求响应 ===")

def test_classify_stream_returns_sse():
    """验证 classify-stream 返回 SSE 格式"""
    import app as main_app
    client = main_app.app.test_client()
    with main_app.app.app_context():
        resp = client.post('/api/classify-stream',
                           json={'file_id': 99999, 'config_id': 99999,
                                 'classify_type': 'difficulty', 'concurrency': 2})
        assert resp.content_type == 'text/event-stream', f"期望 text/event-stream，实际 {resp.content_type}"

def test_score_stream_returns_sse():
    """验证 score-stream 返回 SSE 格式"""
    import app as main_app
    client = main_app.app.test_client()
    with main_app.app.app_context():
        resp = client.post('/api/score-stream',
                           json={'file_id': 99999, 'config_id': 99999, 'concurrency': 3})
        assert resp.content_type == 'text/event-stream', f"期望 text/event-stream，实际 {resp.content_type}"

def test_similarity_stream_returns_sse():
    """验证 similarity-compare-stream 返回 SSE 格式"""
    import app as main_app
    client = main_app.app.test_client()
    with main_app.app.app_context():
        resp = client.post('/api/similarity-compare-stream',
                           json={'task_id': 99999, 'config_id': 99999, 'concurrency': 2})
        assert resp.content_type == 'text/event-stream', f"期望 text/event-stream，实际 {resp.content_type}"

test("classify-stream 返回 SSE", test_classify_stream_returns_sse)
test("score-stream 返回 SSE", test_score_stream_returns_sse)
test("similarity-compare-stream 返回 SSE", test_similarity_stream_returns_sse)


# ============================================================
# 6. 测试代码审查修复点
# ============================================================
print("\n=== 6. 测试代码审查修复点 ===")

def test_traceback_format_exception():
    """验证并行错误路径使用 traceback.format_exception 而非 format_exc"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.classify_items_stream)
    assert 'traceback.format_exception(' in source, "classify_items_stream 应使用 traceback.format_exception"
    assert source.count('traceback.format_exc()') <= 1, "classify_items_stream 并行路径不应使用 format_exc()"

    source_score = inspect.getsource(main_app.score_items_stream)
    assert 'traceback.format_exception(' in source_score, "score_items_stream 应使用 traceback.format_exception"

    source_sim = inspect.getsource(main_app.similarity_compare_stream)
    assert 'traceback.format_exception(' in source_sim, "similarity_compare_stream 应使用 traceback.format_exception"

def test_orm_precache_classify():
    """验证 classify-stream 中预缓存了 ORM 属性 (item._q)"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.classify_items_stream)
    assert 'item._q = item.question' in source, "classify_items_stream 应预缓存 item._q"
    assert 'item._q' in source.split('do_classify')[1], "do_classify 闭包应使用 item._q"

def test_orm_precache_score():
    """验证 score-stream 中预缓存了 ORM 属性 (item._q, item._a)"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.score_items_stream)
    assert 'item._q = item.question' in source, "score_items_stream 应预缓存 item._q"
    assert 'item._a = item.answer' in source, "score_items_stream 应预缓存 item._a"

def test_orm_precache_similarity():
    """验证 similarity-compare-stream 中预缓存了 ORM 属性 (item._q)"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.similarity_compare_stream)
    assert 'item._q = item.question' in source, "similarity_compare_stream 应预缓存 item._q"

def test_similarity_init_uses_processable():
    """验证 similarity init 事件只包含可处理条目（不含跳过的）"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.similarity_compare_stream)
    assert 'for item in processable_items]' in source, "similarity init 应使用 processable_items 而非 items"

def test_concurrency_defensive():
    """验证 concurrency 参数有 try/except 防御"""
    import inspect
    import app as main_app
    source = inspect.getsource(main_app.classify_items_stream)
    assert 'except (TypeError, ValueError)' in source, "classify_items_stream 应对 concurrency 做防御性转换"

def test_similarity_html_response_check():
    """验证 similarity_detail.html 中检查了 response.ok"""
    with open('templates/similarity_detail.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'response.ok' in content, "similarity_detail.html 应检查 response.ok"

test("并行错误路径使用 format_exception", test_traceback_format_exception)
test("classify-stream 预缓存 ORM 属性", test_orm_precache_classify)
test("score-stream 预缓存 ORM 属性", test_orm_precache_score)
test("similarity-compare-stream 预缓存 ORM 属性", test_orm_precache_similarity)
test("similarity init 仅包含可处理条目", test_similarity_init_uses_processable)
test("concurrency 参数防御性转换", test_concurrency_defensive)
test("similarity_detail.html 检查 HTTP 状态", test_similarity_html_response_check)


# ============================================================
# 结果汇总
# ============================================================
print(f"\n{'='*50}")
print(f"测试完成: {passed} 通过, {failed} 失败")
if failed > 0:
    print("请检查上面 [FAIL] 的条目！")
    sys.exit(1)
else:
    print("全部通过！改动验证成功。")
    sys.exit(0)
