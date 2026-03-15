# 语料评测平台 - 部署文档

## 环境要求

- Python 3.9+
- pip3

## 目录结构

```
llmCreator/
├── app.py                 # 主应用
├── config.py              # 配置文件
├── models.py              # 数据库模型
├── requirements.txt       # Python 依赖
├── start.sh               # 启动脚本
├── stop.sh                # 停止脚本
├── services/              # 业务逻辑
│   ├── classifier.py      # 分类服务
│   ├── file_parser.py     # 文件解析
│   ├── llm_service.py     # LLM API 调用
│   └── scorer.py          # 打分服务
├── templates/             # HTML 模板
├── static/                # 静态资源
├── data/                  # SQLite 数据库（自动创建）
└── uploads/               # 上传文件（自动创建）
```

## 快速部署

### 1. 安装依赖

```bash
# 建议使用虚拟环境
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

### 2. 启动服务

```bash
./start.sh
```

默认监听 `0.0.0.0:5000`，可通过环境变量自定义：

```bash
HOST=127.0.0.1 PORT=8080 WORKERS=8 ./start.sh
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 5000 | 监听端口 |
| WORKERS | 4 | gunicorn worker 数量 |

### 3. 停止服务

```bash
./stop.sh
```

### 4. 查看日志

```bash
tail -f app.log
```

## 使用流程

1. 浏览器访问 `http://<服务器IP>:5000`
2. 进入 **配置管理** 页面，添加 LLM 模型配置（API URL、API Key、模型名称）
3. 在首页上传 CSV 或 TXT 格式的语料文件
4. 进入语料详情页，选择操作后点击 **执行选中操作**
5. 评测完成后点击 **导出结果** 下载 Excel 文件

## 数据持久化

- 数据库文件：`data/corpus.db`（SQLite）
- 上传文件：`uploads/` 目录
- 备份时保留以上两个目录即可，服务重启不会丢失数据
