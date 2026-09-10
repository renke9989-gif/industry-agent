from mysql_run_store import MySQLRunStore

if __name__ == "__main__":
    store = MySQLRunStore()
    print("MySQL Run Store schema is ready")
    store.close()
