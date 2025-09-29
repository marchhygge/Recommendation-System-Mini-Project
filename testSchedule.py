from datetime import datetime
import psycopg2 # PostgreSQL database adapter for Python
from psycopg2 import sql

now = datetime.now()
print("Current date and time:", now)

# connect to your postgres DB
conn = psycopg2.connect(
    dbname="your_db_name",
    user="your_username",
    password="your_password",
    host="your_host",
    port="your_port"
)
# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM Restaurants;")

# Retrieve query results
records = cur.fetchall()
print(records)

# Close communication with the database
cur.close()
