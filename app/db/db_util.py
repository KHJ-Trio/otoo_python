import aiomysql
import os
import asyncio

async def get_db_connection():
    try:
        connection = await aiomysql.connect(
            host=os.getenv('DB_HOST'),
            db=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return connection
    except aiomysql.Error as e:
        print("Error while connecting to MySQL", e)
        return None

async def get_model_name():
    connection = await get_db_connection()
    if connection is None:
        raise Exception("Failed to connect to database")
    async with connection.cursor() as cursor:
        await cursor.execute("SELECT model_name FROM model_config ORDER BY model_code DESC LIMIT 1")
        result = await cursor.fetchone()
    connection.close()
    model_name = result[0] if result else None
    return model_name

async def get_db_connection2():
    try:
        connection = await aiomysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            db=os.getenv('DB_NAME'),
            charset='utf8mb4',
            cursorclass=aiomysql.cursors.DictCursor
        )
        return connection
    except aiomysql.Error as e:
        print("Error while connecting to MySQL:", e)
        return None
