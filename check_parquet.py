import duckdb
conn = duckdb.connect(':memory:')
conn.execute("CREATE VIEW avg_sales_menu AS SELECT * FROM read_parquet('C:/Users/Hafez/Desktop/Project C.O.C.O/cleaned/avg_sales_menu.parquet')")
print("=== SCHEMA ===")
print(conn.execute('DESCRIBE avg_sales_menu').fetchdf())
print("\n=== SAMPLE DATA ===")
print(conn.execute('SELECT * FROM avg_sales_menu LIMIT 5').fetchdf())
conn.close()
