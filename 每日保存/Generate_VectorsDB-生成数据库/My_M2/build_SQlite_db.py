# gemini3生成，将.tsv文件转化为sqlite数据库

import sqlite3
import pandas as pd
import csv
from tqdm import tqdm
import os

def build_database():
    # 路径配置
    tsv_path = "/home/lab/ybw/Generate_VectorsDB/DB/raw_data/psgs_w100.tsv"
    db_path = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"
    
    # 1. 连接/创建数据库
    if os.path.exists(db_path):
        os.remove(db_path) # 为了演示，先删旧的，实际使用请小心
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating table...")
    # 关键：id 设置为 PRIMARY KEY，SQLite 会自动为它建立 B-Tree 索引，查找速度极快
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            text TEXT,
            title TEXT
        )
    ''')
    
    print(f"Reading {tsv_path} and inserting into SQLite...")
    
    chunk_size = 100000
    # 估算总行数用于进度条
    total_rows = 21015325 
    
    reader = pd.read_csv(
        tsv_path, 
        sep='\t', 
        header=None, 
        names=['id', 'text', 'title'], 
        chunksize=chunk_size,
        quoting=csv.QUOTE_NONE
    )
    
    count = 0
    for chunk in tqdm(reader, total=total_rows//chunk_size):
        # 数据清洗
        chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
        chunk = chunk.dropna(subset=['id'])
        chunk['id'] = chunk['id'].astype(int)
        
        # 准备数据：转为 tuple 列表 [(id, text, title), ...]
        # 填充空 title 为空字符串
        chunk['title'] = chunk['title'].fillna('')
        chunk['text'] = chunk['text'].fillna('')
        
        data_to_insert = list(zip(chunk['id'], chunk['text'], chunk['title']))
        
        # 批量插入
        cursor.executemany('INSERT OR IGNORE INTO documents (id, text, title) VALUES (?, ?, ?)', data_to_insert)
        conn.commit()
        count += len(data_to_insert)

    print(f"Indexing completed. Creating final optimization...")
    # 这一步不是必须的，但可以让数据库更紧凑
    cursor.execute("VACUUM") 
    
    conn.close()
    print(f"Done! {count} rows inserted into {db_path}")

if __name__ == "__main__":
    build_database()