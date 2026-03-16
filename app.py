"""语料评测平台 - 主应用"""
import os
import json
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, DATABASE_URI
from models import db, LLMConfig, CorpusFile, CorpusItem, PromptTemplate
from services.file_parser import parse_file, allowed_file
from services.llm_service import LLMService
from services.classifier import (
    classify_subjective_objective, classify_difficulty,
    classify_category, generate_objective_answer,
    evaluate_quality, classify_domain, classify_intent,
    classify_combined,
    DEFAULT_CLASSIFY_SUBJ_OBJ_PROMPT, DEFAULT_CLASSIFY_DIFFICULTY_PROMPT,
    DEFAULT_CLASSIFY_CATEGORY_PROMPT, DEFAULT_GENERATE_ANSWER_PROMPT,
    DEFAULT_QUALITY_EVAL_PROMPT, DEFAULT_DOMAIN_CLASSIFY_PROMPT,
    DEFAULT_INTENT_CLASSIFY_PROMPT, DEFAULT_COMBINED_CLASSIFY_PROMPT,
)
from services.scorer import score_answer, DEFAULT_SCORE_PROMPT

app = Flask(__name__)
app.config['SECRET_KEY'] = 'corpus-eval-platform-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

db.init_app(app)

def _auto_migrate(engine):
    """自动检测并添加缺失的数据库列，避免因 schema 变更需要删库重建"""
    from sqlalchemy import inspect as sa_inspect, text
    inspector = sa_inspect(engine)
    with engine.connect() as conn:
        for table_name, table_obj in db.metadata.tables.items():
            if not inspector.has_table(table_name):
                continue
            existing_cols = {col['name'] for col in inspector.get_columns(table_name)}
            for col in table_obj.columns:
                if col.name not in existing_cols:
                    col_type = col.type.compile(engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}'))
        conn.commit()


with app.app_context():
    os.makedirs(os.path.dirname(DATABASE_URI.replace('sqlite:///', '')), exist_ok=True)
    db.create_all()
    _auto_migrate(db.engine)
    # 初始化 / 同步默认 prompt 模板（代码更新后 DB 自动同步）
    defaults = [
        ('主观/客观分类', 'classify_subj_obj', DEFAULT_CLASSIFY_SUBJ_OBJ_PROMPT),
        ('难度等级分类', 'classify_difficulty', DEFAULT_CLASSIFY_DIFFICULTY_PROMPT),
        ('自定义分类', 'classify_category', DEFAULT_CLASSIFY_CATEGORY_PROMPT),
        ('客观题答案生成', 'generate_answer', DEFAULT_GENERATE_ANSWER_PROMPT),
        ('Answer 打分', 'score_answer', DEFAULT_SCORE_PROMPT),
        ('语料质量评估', 'quality_eval', DEFAULT_QUALITY_EVAL_PROMPT),
        ('领域/场景分类', 'domain_classify', DEFAULT_DOMAIN_CLASSIFY_PROMPT),
        ('意图识别分类', 'intent_classify', DEFAULT_INTENT_CLASSIFY_PROMPT),
        ('综合分类(一次调用)', 'combined_classify', DEFAULT_COMBINED_CLASSIFY_PROMPT),
    ]
    for name, ptype, content in defaults:
        existing = PromptTemplate.query.filter_by(prompt_type=ptype, is_default=True).first()
        if not existing:
            tpl = PromptTemplate(name=name, prompt_type=ptype, content=content, is_default=True)
            db.session.add(tpl)
        else:
            # 同步更新默认模板内容（仅当用户未自行修改过时）
            existing.content = content
            existing.name = name
    db.session.commit()


def get_llm_service(config_id=None):
    """获取 LLM 服务实例"""
    if config_id:
        cfg = LLMConfig.query.get(config_id)
    else:
        cfg = LLMConfig.query.filter_by(is_default=True).first()
    if not cfg:
        cfg = LLMConfig.query.first()
    if not cfg:
        return None
    return LLMService(cfg.api_url, cfg.api_key, cfg.model)


# ==================== 页面路由 ====================

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


@app.route('/')
def index():
    files = CorpusFile.query.order_by(CorpusFile.created_at.desc()).all()
    configs = LLMConfig.query.all()
    return render_template('index.html', files=files, configs=configs)


@app.route('/config')
def config_page():
    configs = LLMConfig.query.order_by(LLMConfig.created_at.desc()).all()
    prompts = PromptTemplate.query.order_by(PromptTemplate.prompt_type).all()
    return render_template('config.html', configs=configs, prompts=prompts)


@app.route('/corpus/<int:file_id>')
def corpus_detail(file_id):
    corpus = CorpusFile.query.get_or_404(file_id)
    items = CorpusItem.query.filter_by(file_id=file_id).all()
    configs = LLMConfig.query.all()
    prompts = PromptTemplate.query.all()
    return render_template('corpus_detail.html', corpus=corpus, items=items,
                           configs=configs, prompts=prompts)


# ==================== API 路由 ====================

# ----- LLM 配置 -----
@app.route('/api/llm-config', methods=['POST'])
def save_llm_config():
    data = request.json
    name = data.get('name', '').strip()
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', '').strip()
    is_default = data.get('is_default', False)

    if not all([name, api_url, api_key, model]):
        return jsonify({'error': '所有字段都是必填的'}), 400

    if is_default:
        LLMConfig.query.update({'is_default': False})

    config = LLMConfig(name=name, api_url=api_url, api_key=api_key,
                       model=model, is_default=is_default)
    db.session.add(config)
    db.session.commit()
    return jsonify({'id': config.id, 'message': '保存成功'})


@app.route('/api/llm-config/<int:config_id>', methods=['DELETE'])
def delete_llm_config(config_id):
    config = LLMConfig.query.get_or_404(config_id)
    db.session.delete(config)
    db.session.commit()
    return jsonify({'message': '删除成功'})


@app.route('/api/llm-config/<int:config_id>/test', methods=['POST'])
def test_llm_config(config_id):
    config = LLMConfig.query.get_or_404(config_id)
    try:
        llm = LLMService(config.api_url, config.api_key, config.model)
        result = llm.chat("你是一个助手。", "请回复'连接成功'这四个字。", temperature=0)
        return jsonify({'message': '连接测试成功', 'response': result})
    except Exception as e:
        return jsonify({'error': f'连接失败: {str(e)}'}), 500


# ----- Prompt 模板 -----
@app.route('/api/prompt', methods=['POST'])
def save_prompt():
    data = request.json
    name = data.get('name', '').strip()
    prompt_type = data.get('prompt_type', '').strip()
    content = data.get('content', '').strip()

    if not all([name, prompt_type, content]):
        return jsonify({'error': '所有字段都是必填的'}), 400

    prompt = PromptTemplate(name=name, prompt_type=prompt_type, content=content)
    db.session.add(prompt)
    db.session.commit()
    return jsonify({'id': prompt.id, 'message': '保存成功'})


@app.route('/api/prompt/<int:prompt_id>', methods=['PUT'])
def update_prompt(prompt_id):
    prompt = PromptTemplate.query.get_or_404(prompt_id)
    data = request.json
    prompt.content = data.get('content', prompt.content)
    prompt.name = data.get('name', prompt.name)
    db.session.commit()
    return jsonify({'message': '更新成功'})


@app.route('/api/prompt/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    prompt = PromptTemplate.query.get_or_404(prompt_id)
    if prompt.is_default:
        return jsonify({'error': '不能删除默认模板'}), 400
    db.session.delete(prompt)
    db.session.commit()
    return jsonify({'message': '删除成功'})


# ----- 文件上传 -----
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    corpus_type = request.form.get('corpus_type', 'question')

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({'error': f'不支持的文件格式，仅支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        items = parse_file(filepath, corpus_type)
        if not items:
            os.remove(filepath)
            return jsonify({'error': '文件解析失败或文件为空'}), 400

        corpus = CorpusFile(
            filename=filename,
            original_name=file.filename,
            file_type=ext,
            corpus_type=corpus_type,
            record_count=len(items),
        )
        db.session.add(corpus)
        db.session.flush()

        for item_data in items:
            item = CorpusItem(
                file_id=corpus.id,
                question=item_data['question'],
                answer=item_data.get('answer', ''),
            )
            db.session.add(item)

        db.session.commit()
        return jsonify({
            'id': corpus.id,
            'message': f'上传成功，解析到 {len(items)} 条语料',
            'count': len(items),
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'解析失败: {str(e)}'}), 500


@app.route('/api/corpus/<int:file_id>', methods=['DELETE'])
def delete_corpus(file_id):
    corpus = CorpusFile.query.get_or_404(file_id)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], corpus.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    db.session.delete(corpus)
    db.session.commit()
    return jsonify({'message': '删除成功'})


# ----- 分类与评测 -----
@app.route('/api/classify-stream', methods=['POST'])
def classify_items_stream():
    """流式分类语料（每处理一行就返回结果）"""
    from flask import Response, stream_with_context
    import json

    def generate():
        data = request.json
        file_id = data.get('file_id')
        config_id = data.get('config_id')
        classify_type = data.get('classify_type')
        classify_types = data.get('classify_types', [])
        categories = data.get('categories', '')
        prompt_id = data.get('prompt_id')
        item_ids = data.get('item_ids', [])

        llm = get_llm_service(config_id)
        if not llm:
            yield f"data: {json.dumps({'error': '请先配置大模型'})}\n\n"
            return

        custom_prompt = None
        if prompt_id:
            tpl = PromptTemplate.query.get(prompt_id)
            if tpl:
                custom_prompt = tpl.content

        if item_ids:
            items = CorpusItem.query.filter(CorpusItem.id.in_(item_ids)).all()
        else:
            items = CorpusItem.query.filter_by(file_id=file_id).all()

        total = len(items)
        for i, item in enumerate(items, 1):
            try:
                if classify_type == 'combined':
                    res = classify_combined(llm, item.question, categories, custom_prompt)
                    _apply_combined_result(item, res, classify_types, categories)
                    db.session.commit()
                    yield f"data: {json.dumps({'status': 'ok', 'item_id': item.id, 'progress': i, 'total': total, 'data': res})}\n\n"
                else:
                    # 其他分类类型保持原有逻辑（简化处理）
                    yield f"data: {json.dumps({'status': 'skip', 'item_id': item.id, 'progress': i, 'total': total})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'item_id': item.id, 'error': str(e), 'progress': i, 'total': total})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/classify', methods=['POST'])
def classify_items():
    """批量分类语料"""
    data = request.json
    file_id = data.get('file_id')
    config_id = data.get('config_id')
    classify_type = data.get('classify_type')  # subj_obj / difficulty / category / quality / domain / intent / combined
    classify_types = data.get('classify_types', [])  # combined 模式下指定哪些维度
    categories = data.get('categories', '')  # 自定义分类类型（逗号分隔）
    prompt_id = data.get('prompt_id')
    item_ids = data.get('item_ids', [])  # 可选，指定条目

    llm = get_llm_service(config_id)
    if not llm:
        return jsonify({'error': '请先配置大模型'}), 400

    custom_prompt = None
    if prompt_id:
        tpl = PromptTemplate.query.get(prompt_id)
        if tpl:
            custom_prompt = tpl.content

    if item_ids:
        items = CorpusItem.query.filter(CorpusItem.id.in_(item_ids)).all()
    else:
        items = CorpusItem.query.filter_by(file_id=file_id).all()

    results = []
    errors = []
    for item in items:
        try:
            if classify_type == 'combined':
                # 一次调用返回所有维度
                res = classify_combined(llm, item.question, categories, custom_prompt)
                _apply_combined_result(item, res, classify_types, categories)
                results.append({'id': item.id, 'status': 'ok'})

            elif classify_type == 'subj_obj':
                res = classify_subjective_objective(llm, item.question, custom_prompt)
                item.subj_obj = res.get('type', '')
                if item.subj_obj == 'objective':
                    try:
                        ans_res = generate_objective_answer(llm, item.question)
                        item.objective_answer = ans_res.get('answer', '')
                    except Exception:
                        pass
                results.append({'id': item.id, 'subj_obj': item.subj_obj})

            elif classify_type == 'difficulty':
                res = classify_difficulty(llm, item.question, custom_prompt)
                item.difficulty = res.get('difficulty', '')
                results.append({'id': item.id, 'difficulty': item.difficulty})

            elif classify_type == 'category':
                if not categories:
                    return jsonify({'error': '自定义分类需要提供分类类型'}), 400
                res = classify_category(llm, item.question, categories, custom_prompt)
                item.category = res.get('category', '')
                results.append({'id': item.id, 'category': item.category})

            elif classify_type == 'quality':
                res = evaluate_quality(llm, item.question, custom_prompt)
                item.quality_score = float(res.get('quality_score', 0))
                item.quality_label = res.get('quality_label', '')
                item.quality_detail = json.dumps(res, ensure_ascii=False)
                results.append({'id': item.id, 'quality_score': item.quality_score})

            elif classify_type == 'domain':
                res = classify_domain(llm, item.question, custom_prompt)
                item.domain = res.get('domain', '')
                item.sub_domain = res.get('sub_domain', '')
                results.append({'id': item.id, 'domain': item.domain})

            elif classify_type == 'intent':
                res = classify_intent(llm, item.question, custom_prompt)
                item.intent = res.get('intent', '')
                item.intent_cn = res.get('intent_cn', '')
                item.intent_confidence = float(res.get('confidence', 0))
                results.append({'id': item.id, 'intent': item.intent})

        except Exception as e:
            errors.append({'id': item.id, 'question': item.question[:50], 'error': str(e)})

    db.session.commit()
    return jsonify({
        'message': f'分类完成: 成功 {len(results)} 条, 失败 {len(errors)} 条',
        'results': results,
        'errors': errors,
    })


def _apply_combined_result(item, res, classify_types, categories):
    """将综合分类结果写入 CorpusItem 对应字段"""
    types = set(classify_types) if classify_types else {'subj_obj', 'difficulty', 'quality', 'domain', 'intent', 'category'}

    if 'subj_obj' in types:
        item.subj_obj = res.get('subj_obj', '')
        obj_answer = res.get('objective_answer', '')
        if item.subj_obj == 'objective' and obj_answer:
            item.objective_answer = obj_answer

    if 'difficulty' in types:
        item.difficulty = res.get('difficulty', '')

    if 'quality' in types:
        item.quality_score = float(res.get('quality_score', 0))
        item.quality_label = res.get('quality_label', '')
        item.quality_detail = json.dumps(res, ensure_ascii=False)

    if 'domain' in types:
        item.domain = res.get('domain', '')
        item.sub_domain = res.get('sub_domain', '')

    if 'intent' in types:
        item.intent = res.get('intent', '')
        item.intent_cn = res.get('intent_cn', '')
        item.intent_confidence = float(res.get('intent_confidence', res.get('confidence', 0)))

    if 'category' in types and categories:
        item.category = res.get('category', '')


@app.route('/api/score', methods=['POST'])
def score_items():
    """批量对 Answer 打分"""
    data = request.json
    file_id = data.get('file_id')
    config_id = data.get('config_id')
    prompt_id = data.get('prompt_id')
    item_ids = data.get('item_ids', [])

    llm = get_llm_service(config_id)
    if not llm:
        return jsonify({'error': '请先配置大模型'}), 400

    custom_prompt = None
    if prompt_id:
        tpl = PromptTemplate.query.get(prompt_id)
        if tpl:
            custom_prompt = tpl.content

    if item_ids:
        items = CorpusItem.query.filter(CorpusItem.id.in_(item_ids)).all()
    else:
        items = CorpusItem.query.filter_by(file_id=file_id).all()

    results = []
    errors = []
    for item in items:
        if not item.answer:
            continue
        try:
            res = score_answer(llm, item.question, item.answer, custom_prompt)
            item.answer_score = float(res.get('score', 0))
            item.score_reason = res.get('reason', '')
            results.append({
                'id': item.id,
                'score': item.answer_score,
                'reason': item.score_reason,
            })
        except Exception as e:
            errors.append({'id': item.id, 'question': item.question[:50], 'error': str(e)})

    db.session.commit()
    return jsonify({
        'message': f'打分完成: 成功 {len(results)} 条, 失败 {len(errors)} 条',
        'results': results,
        'errors': errors,
    })


@app.route('/api/export/<int:file_id>')
def export_results(file_id):
    """导出评测结果为 Excel"""
    import io
    from flask import Response
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    corpus = CorpusFile.query.get_or_404(file_id)
    items = CorpusItem.query.filter_by(file_id=file_id).all()

    wb = Workbook()
    ws = wb.active
    ws.title = '评测结果'

    # 表头
    headers = ['ID', '问题', '答案', '主观/客观', '难度', '分类',
               '客观题答案', '质量评分', '质量等级', '领域', '子领域',
               '意图', '意图(中文)', '意图置信度', '答案评分', '评分理由']
    ws.append(headers)

    # 表头样式
    header_font = Font(bold=True, color='FFFFFF', size=11)
    header_fill = PatternFill(start_color='344955', end_color='344955', fill_type='solid')
    thin_border = Border(
        bottom=Side(style='thin', color='CCCCCC'),
    )
    for col_idx, cell in enumerate(ws[1], 1):
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # 数据行
    for idx, item in enumerate(items, 1):
        difficulty_display = ''
        if item.difficulty == 'L1':
            difficulty_display = 'L1(简单)'
        elif item.difficulty == 'L2':
            difficulty_display = 'L2(中等)'
        elif item.difficulty == 'L3':
            difficulty_display = 'L3(困难)'
        elif item.difficulty:
            difficulty_display = item.difficulty

        subj_obj_display = ''
        if item.subj_obj == 'objective':
            subj_obj_display = '客观'
        elif item.subj_obj == 'subjective':
            subj_obj_display = '主观'

        ws.append([
            idx, item.question, item.answer or '',
            subj_obj_display, difficulty_display, item.category or '',
            item.objective_answer or '',
            item.quality_score or '', item.quality_label or '',
            item.domain or '', item.sub_domain or '',
            item.intent or '', item.intent_cn or '',
            item.intent_confidence or '',
            item.answer_score or '', item.score_reason or '',
        ])

    # 数据行边框
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(vertical='center', wrap_text=True)

    # 列宽
    col_widths = [6, 40, 30, 10, 12, 12, 30, 10, 10, 12, 12, 18, 12, 10, 10, 30]
    from openpyxl.utils import get_column_letter
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    from urllib.parse import quote
    filename = os.path.splitext(corpus.original_name)[0] + '_评测结果.xlsx'
    encoded = quote(filename)
    return Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{encoded}"}
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
