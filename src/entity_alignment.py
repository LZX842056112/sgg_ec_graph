import pymysql
import hashlib
import chromadb
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from configuration.config import MYSQL_CONFIG, EMBEDDING_MODEL_PATH, VECTOR_STORE_DIR

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device="cuda")
        print("加载嵌入模型")
    return _embedding_model


# 初始化 MySQL 同义词表
def init_mysql_table():
    """初始化 MySQL 表结构，若缺失列则自动添加"""
    with pymysql.connect(**MYSQL_CONFIG) as conn:
        with conn.cursor() as cursor:
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE 'entity_mapping'")
            table_exists = cursor.fetchone()
            if not table_exists:
                # 创建全新表
                create_table_sql = """
                CREATE TABLE entity_mapping (
                    id VARCHAR(255) NOT NULL COMMENT '实体ID',
                    synonym VARCHAR(255) NOT NULL COLLATE utf8mb4_bin COMMENT '同义词',
                    std_name VARCHAR(255) NOT NULL COMMENT '标准词',
                    entity_schema VARCHAR(255) NOT NULL COMMENT '实体类型',
                    is_reviewed INT DEFAULT 0 NOT NULL COMMENT '是否已审核',
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    update_time TIMESTAMP DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    PRIMARY KEY (synonym, entity_schema)
                ) COMMENT '实体映射表';
                """
                cursor.execute(create_table_sql)
                print("entity_mapping 表创建成功")
            else:
                # 表已存在，检查必要列是否存在
                cursor.execute("DESCRIBE entity_mapping")
                columns = [row[0] for row in cursor.fetchall()]
                required_columns = {
                    'id': 'VARCHAR(255)',
                    'synonym': 'VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin',
                    'std_name': 'VARCHAR(255)',
                    'entity_schema': 'VARCHAR(255)',
                    'is_reviewed': 'INT DEFAULT 0',
                    'create_time': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'update_time': 'TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP',
                }
                for col, col_def in required_columns.items():
                    if col not in columns:
                        try:
                            # 添加缺失列
                            alter_sql = f"ALTER TABLE entity_mapping ADD COLUMN {col} {col_def}"
                            cursor.execute(alter_sql)
                            print(f"添加缺失列: {col}")
                        except Exception as e:
                            print(f"添加列 {col} 失败: {e}")
                # 检查主键是否正确（可选）
                cursor.execute("SHOW KEYS FROM entity_mapping WHERE Key_name = 'PRIMARY'")
                pk_columns = [row[4] for row in cursor.fetchall()]
                if set(pk_columns) != {'synonym', 'entity_schema'}:
                    print("警告：主键可能不正确，建议手动检查")
        conn.commit()


init_mysql_table()


def entity_alignment(datas, entity_schema, field_name, embed_batch_size=128):
    """离线批量实体对齐"""
    embedding_model = get_embedding_model()

    # 加载已有映射
    with pymysql.connect(**MYSQL_CONFIG) as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT id, synonym, std_name FROM entity_mapping WHERE entity_schema=%s AND is_reviewed=1",
                (entity_schema,)
            )
            old_mappings = cursor.fetchall()

    old_entities = []
    old_std_entities = []
    old_mapping_dict = {}
    if old_mappings:
        old_entities = list(set([m["synonym"] for m in old_mappings]))
        old_std_entities = list(set([m["std_name"] for m in old_mappings]))
        old_mapping_dict = {m["synonym"]: m["std_name"] for m in old_mappings}

    # 收集新实体及频率
    new_entity_freq = defaultdict(int)
    for item in datas:
        value = item.get(field_name)
        if not value:
            continue
        if isinstance(value, str):
            value = [value]
        for v in value:
            if v:
                new_entity_freq[v] += 1

    new_entities = list(set(new_entity_freq.keys()) - set(old_entities))
    new_mapping = {}

    if new_entities:
        print(f"发现 {len(new_entities)} 个新 {entity_schema} 实体")
        new_embeddings = embedding_model.encode(
            new_entities, batch_size=embed_batch_size, normalize_embeddings=True
        )
        cluster_ids = DBSCAN(eps=0.15, min_samples=1, metric="cosine").fit_predict(new_embeddings)
        cluster_dict = defaultdict(list)
        for ent, cid in zip(new_entities, cluster_ids):
            if cid >= 0:
                cluster_dict[cid].append(ent)

        if not old_entities:
            for cid, ents in cluster_dict.items():
                std = max(ents, key=lambda x: new_entity_freq[x])
                for e in ents:
                    new_mapping[e] = std
        else:
            temp_std_to_ents = {}
            for cid, ents in cluster_dict.items():
                std = max(ents, key=lambda x: new_entity_freq[x])
                temp_std_to_ents[std] = ents

            temp_stds = list(temp_std_to_ents.keys())
            temp_emb = embedding_model.encode(temp_stds, batch_size=embed_batch_size, normalize_embeddings=True)
            old_emb = embedding_model.encode(old_std_entities, batch_size=embed_batch_size, normalize_embeddings=True)

            sim_matrix = cosine_similarity(temp_emb, old_emb)
            threshold = 0.85
            for i, temp_std in enumerate(temp_stds):
                max_idx = sim_matrix[i].argmax()
                max_sim = sim_matrix[i][max_idx]
                if max_sim >= threshold:
                    matched_std = old_std_entities[max_idx]
                    for e in temp_std_to_ents[temp_std]:
                        new_mapping[e] = matched_std
                else:
                    for e in temp_std_to_ents[temp_std]:
                        new_mapping[e] = temp_std

        # 写入 MySQL
        with pymysql.connect(**MYSQL_CONFIG) as conn:
            with conn.cursor() as cursor:
                for syn, std in new_mapping.items():
                    uid = f"{entity_schema}_{hashlib.md5(std.encode()).hexdigest()[:16]}"
                    cursor.execute(
                        "INSERT IGNORE INTO entity_mapping (id, synonym, std_name, entity_schema, is_reviewed) VALUES (%s,%s,%s,%s,1)",
                        (uid, syn, std, entity_schema)
                    )
                conn.commit()
        print(f"新增 {len(new_mapping)} 条 {entity_schema} 映射")

    # 合并映射并替换原数据
    full_mapping = {**old_mapping_dict, **new_mapping}
    for item in datas:
        value = item.get(field_name)
        if not value:
            continue
        if isinstance(value, str):
            item[field_name] = full_mapping.get(value, value)
        elif isinstance(value, list):
            item[field_name] = [full_mapping.get(v, v) for v in value]


def vector_indexing(datas, embed_batch_size=128, add_batch_size=256):
    """构建向量索引 (Chroma)"""
    embedding_model = get_embedding_model()
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection("ecommerce")

    # 收集实体 (品牌、分类、SPU、SKU、标签等)
    items = []
    for data in datas:
        for field, etype in [("name", "trademark"), ("name", "category1"), ("name", "category2"),
                             ("name", "category3"), ("name", "spu"), ("name", "sku"), ("name", "tag")]:
            val = data.get(field)
            if not val:
                continue
            if isinstance(val, str):
                val = [val]
            for v in val:
                if v:
                    uid = f"{etype}_{hashlib.md5(v.encode()).hexdigest()[:16]}"
                    items.append({"id": uid, "type": etype, "text": v})

    # 去重
    seen_ids = set(collection.get()["ids"])
    new_items = []
    seen = set()
    for it in items:
        if it["id"] not in seen_ids and it["id"] not in seen:
            seen.add(it["id"])
            new_items.append(it)

    if not new_items:
        print("无新增向量数据")
        return

    documents = [it["text"] for it in new_items]
    metadatas = [{"type": it["type"]} for it in new_items]
    ids = [it["id"] for it in new_items]

    embeddings = embedding_model.encode(documents, batch_size=embed_batch_size, normalize_embeddings=True,
                                        show_progress_bar=True)

    for i in tqdm(range(0, len(ids), add_batch_size), desc="写入Chroma"):
        collection.add(
            ids=ids[i:i + add_batch_size],
            documents=documents[i:i + add_batch_size],
            metadatas=metadatas[i:i + add_batch_size],
            embeddings=embeddings[i:i + add_batch_size]
        )
    print(f"添加 {len(new_items)} 条向量数据")


class EntityAlignment:
    """运行时实体对齐"""

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    def entity_mapping(self, text, entity_schema):
        with pymysql.connect(**MYSQL_CONFIG) as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(
                    "SELECT std_name FROM entity_mapping WHERE is_reviewed=1 AND synonym=%s AND entity_schema=%s",
                    (text, entity_schema)
                )
                res = cursor.fetchone()
                return res["std_name"] if res else None

    def vector_retrieve(self, text, entity_schema, threshold=0.5):
        embedding = self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        collection = self.chroma_client.get_collection("ecommerce")
        res = collection.query(query_embeddings=[embedding], n_results=1, where={"type": entity_schema})
        if res["distances"][0] and res["distances"][0][0] < threshold:
            return res["documents"][0][0]
        return None

    def __call__(self, text, entity_schema):
        std = self.entity_mapping(text, entity_schema)
        if std:
            return std
        std = self.vector_retrieve(text, entity_schema)
        if std:
            with pymysql.connect(**MYSQL_CONFIG) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT IGNORE INTO entity_mapping (synonym, std_name, entity_schema, is_reviewed) VALUES (%s,%s,%s,1)",
                        (text, std, entity_schema)
                    )
                conn.commit()
        return std
