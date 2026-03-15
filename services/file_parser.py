"""文件解析服务：支持 CSV 和 TXT 格式"""
import csv
import io
import os


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def parse_csv(filepath, corpus_type='question'):
    """
    解析 CSV 文件
    corpus_type='question': 只需要 question 列
    corpus_type='qa': 需要 question 和 answer 列
    """
    items = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        # 尝试检测分隔符
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]

        # 找到 question 列
        q_col = None
        for name in fieldnames:
            if name in ('question', 'q', '问题', 'query', 'prompt', 'input'):
                q_col = reader.fieldnames[fieldnames.index(name)]
                break

        # 找到 answer 列
        a_col = None
        if corpus_type == 'qa':
            for name in fieldnames:
                if name in ('answer', 'a', '答案', 'response', 'output', 'reply'):
                    a_col = reader.fieldnames[fieldnames.index(name)]
                    break

        if q_col is None:
            # 如果没有列头匹配，尝试用第一列作为 question
            f.seek(0)
            reader = csv.reader(f, dialect=dialect)
            rows = list(reader)
            if len(rows) > 0:
                has_header = not _looks_like_data(rows[0][0]) if rows[0] else False
                start = 1 if has_header else 0
                for row in rows[start:]:
                    if not row or not row[0].strip():
                        continue
                    item = {'question': row[0].strip()}
                    if corpus_type == 'qa' and len(row) > 1:
                        item['answer'] = row[1].strip()
                    items.append(item)
            return items

        for row in reader:
            q = row.get(q_col, '').strip()
            if not q:
                continue
            item = {'question': q}
            if corpus_type == 'qa' and a_col:
                item['answer'] = row.get(a_col, '').strip()
            items.append(item)

    return items


def parse_txt(filepath, corpus_type='question'):
    """
    解析 TXT 文件
    每行一个 question；如果是 QA 对，用 \\t 或 ||| 分隔
    """
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if corpus_type == 'qa':
                # 尝试不同分隔符
                parts = None
                for sep in ['\t', '|||', '|', '::']:
                    if sep in line:
                        parts = line.split(sep, 1)
                        break
                if parts and len(parts) == 2:
                    items.append({
                        'question': parts[0].strip(),
                        'answer': parts[1].strip()
                    })
                else:
                    items.append({'question': line})
            else:
                items.append({'question': line})

    return items


def parse_file(filepath, corpus_type='question'):
    """根据文件扩展名选择解析器"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return parse_csv(filepath, corpus_type)
    elif ext == '.txt':
        return parse_txt(filepath, corpus_type)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _looks_like_data(value):
    """简单判断是否像数据行（而非表头）"""
    if len(value) > 50:
        return True
    if value.replace(' ', '').isdigit():
        return True
    return False
