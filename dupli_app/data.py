import sqlite3, os, pandas as pd
from datetime import date

DB_PATH = "production.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_log (
            date        TEXT PRIMARY KEY,
            output      INTEGER NOT NULL,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()

def save_entry(entry_date: str, output: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO daily_log (date, output)
        VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET output=excluded.output
    """, (entry_date, output))
    con.commit()
    con.close()

def delete_entry(entry_date: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM daily_log WHERE date=?", (entry_date,))
    con.commit()
    con.close()

def load_all() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM daily_log ORDER BY date", con)
    con.close()
    return df

def get_monthly_summary() -> pd.DataFrame:
    df = load_all()
    if df.empty:
        return pd.DataFrame(columns=['month','total','days','avg_daily'])
    df['month'] = df['date'].str[:7]
    summary = df.groupby('month').agg(
        total=('output','sum'),
        days=('output','count'),
        avg_daily=('output','mean')
    ).reset_index()
    return summary
