from tinydb import TinyDB

db = TinyDB("production_log.json")
db.truncate()  # clear existing data first

sample = [
    {"log_date": "2026-02-01", "m1_output": 24000, "m2_output": 8000,  "notes": "Normal day"},
    {"log_date": "2026-02-02", "m1_output": 28000, "m2_output": 9000,  "notes": ""},
    {"log_date": "2026-02-03", "m1_output": 31000, "m2_output": 10000, "notes": "Good run"},
    {"log_date": "2026-02-04", "m1_output": 22000, "m2_output": 7000,  "notes": "Machine issue"},
    {"log_date": "2026-02-05", "m1_output": 35000, "m2_output": 12000, "notes": ""},
    {"log_date": "2026-02-06", "m1_output": 40000, "m2_output": 14000, "notes": "Best day"},
    {"log_date": "2026-02-07", "m1_output": 33000, "m2_output": 11000, "notes": ""},
    {"log_date": "2026-02-10", "m1_output": 38000, "m2_output": 13000, "notes": ""},
    {"log_date": "2026-02-11", "m1_output": 29000, "m2_output": 9500,  "notes": "Short shift"},
    {"log_date": "2026-02-12", "m1_output": 42000, "m2_output": 15000, "notes": ""},
    {"log_date": "2026-03-03", "m1_output": 48000, "m2_output": 18000, "notes": "After packing person"},
    {"log_date": "2026-03-04", "m1_output": 52000, "m2_output": 19000, "notes": ""},
    {"log_date": "2026-03-05", "m1_output": 55000, "m2_output": 20000, "notes": "Target hit"},
    {"log_date": "2026-03-06", "m1_output": 50000, "m2_output": 18500, "notes": ""},
    {"log_date": "2026-03-07", "m1_output": 58000, "m2_output": 21000, "notes": "Best week"},
]

for row in sample:
    db.insert(row)

db.close()
print(f"Seeded {len(sample)} entries into production_log.json")
