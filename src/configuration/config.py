from pathlib import Path
import os
import dotenv

dotenv.load_dotenv()

# 1. 目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / 'data'
NER_DIR = 'ner'
RAW_DATA_DIR = DATA_DIR / NER_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / 'processed'

LOG_DIR = ROOT_DIR / 'logs'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

# 2. 数据文件名 和 模型名称
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')
MODEL_NAME = f'D:\cache\huggingface\hub\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea'

# 3. 超参数
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5

SAVE_STEPS = 20

# 4. NER任务分类标签
LABELS = ['B', 'I', 'O']

# 5. 数据库连接
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': 'gmall',
}

NEO4J_CONFIG = {
    'uri': "neo4j://localhost:7687",
    'auth': ("neo4j", os.getenv('NEO4J_PASSWORD')),
}
