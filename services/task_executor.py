"""并行任务执行器：使用 ThreadPoolExecutor 分批并行处理"""
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def execute_parallel_batch(items, process_fn, concurrency=3):
    """
    将 items 分批并行处理。

    采用"分批并行，批内并发，批间串行"模式：
    - 每批最多 concurrency 个任务同时执行
    - 等待该批全部完成后，按原始顺序 yield 结果
    - 主线程可在收到结果后安全地操作数据库

    Args:
        items: 待处理列表
        process_fn: callable，接受 (item,)，返回结果或抛异常。在子线程中执行，不应操作数据库。
        concurrency: 并行数

    Yields:
        (global_index, item, result, error) 元组
        成功时 error=None，失败时 result=None
    """
    if concurrency < 1:
        concurrency = 1

    total = len(items)
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        for batch_start in range(0, total, concurrency):
            batch = items[batch_start:batch_start + concurrency]

            # 提交该批任务，保留顺序信息
            futures_ordered = []
            for offset, item in enumerate(batch):
                global_idx = batch_start + offset + 1
                future = executor.submit(process_fn, item)
                futures_ordered.append((global_idx, item, future))

            # 等待该批全部完成，按原始顺序返回结果
            for global_idx, item, future in futures_ordered:
                try:
                    result = future.result()  # 阻塞等待完成
                    yield (global_idx, item, result, None)
                except Exception as e:
                    logger.error(f"[并行执行] 第 {global_idx}/{total} 条处理失败: {str(e)}")
                    yield (global_idx, item, None, e)
    finally:
        executor.shutdown(wait=False)
