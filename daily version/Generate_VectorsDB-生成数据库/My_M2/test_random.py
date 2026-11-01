import sqlite3

db_path = "/home/lab/ybw/Generate_VectorsDB/DB/DB/wiki_knowledge.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("正在查询数据库中是否包含 'RLHF'...")
# 模糊查询标题包含 RLHF 的文章
cursor.execute("SELECT id, title FROM documents WHERE title LIKE '%RLHF%'")
rows = cursor.fetchall()

if len(rows) == 0:
    print("❌ 结论：数据库里没有 RLHF！这是由于数据版本过旧导致的。")
else:
    print(f"✅ 结论：找到了 {len(rows)} 条相关数据：")
    for r in rows:
        print(r)

conn.close()