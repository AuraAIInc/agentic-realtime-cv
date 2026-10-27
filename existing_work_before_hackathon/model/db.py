import pymysql
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
load_dotenv()
import os

class MysqlDB(object):
    pool: PooledDB

    def __init__(self):
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=200,
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DB'),
            port=int(os.getenv('MYSQL_PORT')),
        )

    def _get_conn(self):
        conn = self.pool.connection()
        cursor = conn.cursor()
        return conn, cursor

    def get_connection(self):
        return self.pool.connection()

    def execute_query(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            results = cursor.fetchall()
        except Exception as e:
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return results

    def execute_query_one(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            result = cursor.fetchone()
        except Exception as e:
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    def execute_update(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            conn.commit()
            result = cursor.rowcount
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    """
    This method executes an SQL delete statement and returns the number of affected rows.
    """

    def execute_delete(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            conn.commit()
            result = cursor.rowcount
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    """
    This method executes multiple SQL delete statements and returns the number of affected rows.
    """

    def execute_many_delete(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            result = cursor.executemany(query, args)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    def execute_many(self, query, args_list):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, args_list)
        num_rows = cursor.rowcount
        cursor.close()
        conn.close()
        return num_rows

    """This method executes a SQL query and returns the results as a list of dictionaries, where each dictionary 
    represents a row and the keys correspond to the column names.
    """

    def execute_query_dict(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(query, args)
            results = cursor.fetchall()
        except Exception as e:
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return results

    """
    This method executes an SQL insert statement and returns the last inserted ID (if the table has an auto-incrementing primary key).
python
    """

    def execute_insert(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            conn.commit()
            last_id = cursor.lastrowid
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return last_id

    """
    This method executes a list of SQL statements as a single transaction.
    """

    def execute_transaction(self, queries):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for query in queries:
                cursor.execute(query)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(query)
            raise e
        finally:
            cursor.close()
            conn.close()

    """
    This method executes an SQL script containing multiple statements and returns the number of affected rows.
    """

    def execute_script(self, script):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for statement in script.split(';'):
                if statement.strip():
                    cursor.execute(statement)
            conn.commit()
            result = cursor.rowcount
        except Exception as e:
            conn.rollback()
            print(script)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    """
    This method executes multiple SQL insert statements and returns the number of affected rows.
    """

    def execute_many_insert(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            result = cursor.executemany(query, args)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result

    """
    This method executes multiple SQL update statements and returns the number of affected rows.
    """

    def execute_many_update(self, query, args=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            result = cursor.executemany(query, args)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(query)
            print(args)
            raise e
        finally:
            cursor.close()
            conn.close()
        return result
