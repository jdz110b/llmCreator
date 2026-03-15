from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class LLMConfig(db.Model):
    """大模型配置"""
    __tablename__ = 'llm_config'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    api_url = db.Column(db.String(500), nullable=False)
    api_key = db.Column(db.String(500), nullable=False)
    model = db.Column(db.String(200), nullable=False)
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CorpusFile(db.Model):
    """语料文件"""
    __tablename__ = 'corpus_file'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(500), nullable=False)
    original_name = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # csv / txt
    corpus_type = db.Column(db.String(20), nullable=False)  # question / qa
    record_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    items = db.relationship('CorpusItem', backref='corpus_file', lazy=True, cascade='all, delete-orphan')


class CorpusItem(db.Model):
    """语料条目"""
    __tablename__ = 'corpus_item'
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('corpus_file.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)  # QA 对时有值

    # 分类结果
    subj_obj = db.Column(db.String(20), nullable=True)  # subjective / objective
    difficulty = db.Column(db.String(20), nullable=True)  # easy / medium / hard
    category = db.Column(db.String(200), nullable=True)  # 自定义分类
    objective_answer = db.Column(db.Text, nullable=True)  # 客观题答案

    # 质量评估
    quality_score = db.Column(db.Float, nullable=True)  # 综合质量分 1-10
    quality_label = db.Column(db.String(20), nullable=True)  # high / medium / low
    quality_detail = db.Column(db.Text, nullable=True)  # 质量评估详情 JSON

    # 领域分类
    domain = db.Column(db.String(100), nullable=True)  # 领域
    sub_domain = db.Column(db.String(100), nullable=True)  # 子领域

    # 意图分类
    intent = db.Column(db.String(50), nullable=True)  # 意图英文标识
    intent_cn = db.Column(db.String(50), nullable=True)  # 意图中文名
    intent_confidence = db.Column(db.Float, nullable=True)  # 置信度

    # 打分结果
    answer_score = db.Column(db.Float, nullable=True)
    score_reason = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PromptTemplate(db.Model):
    """Prompt 模板"""
    __tablename__ = 'prompt_template'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    prompt_type = db.Column(db.String(50), nullable=False)  # classify_subj_obj / classify_difficulty / classify_category / score_answer / generate_answer
    content = db.Column(db.Text, nullable=False)
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
