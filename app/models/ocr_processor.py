import hashlib
import json
import os
from paddleocr import PaddleOCR
from db.db_util import get_db_connection2

class MyPaddleOCR:
    def __init__(self, lang: str = "korean", **kwargs):
        self.lang = lang
        self._ocr = PaddleOCR(lang=lang)

    def get_image_hash(self, img_bytes):
        return hashlib.md5(img_bytes).hexdigest()

    async def get_ocr_result(self, image_hash):
        try:
            connection = await get_db_connection2()
            if connection is None:
                print("Failed to connect to database")
                return None
            async with connection.cursor() as cursor:
                if cursor is None:
                    print("Failed to create cursor")
                    return None
                await cursor.execute("SELECT ocr_result FROM ocr_cache WHERE image_hash=%s", (image_hash,))
                result = await cursor.fetchone()
            await connection.ensure_closed()
            print("succes")
            return json.loads(result['ocr_result']) if result else None
        except Exception as e:
            print(f"Error fetching OCR result from database: {e}")
            return None

    async def save_ocr_result_to_cache(self, image_hash, ocr_result):
        try:
            connection = await get_db_connection2()
            if connection is None:
                print("Failed to connect to database")
                return
            async with connection.cursor() as cursor:
                if cursor is None:
                    print("Failed to create cursor")
                    return
                await cursor.execute(
                    "REPLACE INTO ocr_cache (image_hash, ocr_result) VALUES (%s, %s)",
                    (image_hash, json.dumps(ocr_result, ensure_ascii=False))
                )
                await connection.commit()
            await connection.ensure_closed()
        except Exception as e:
            print(f"Error saving OCR result to database: {e}")

    async def run_ocr(self, img_bytes: bytes):
        image_hash = self.get_image_hash(img_bytes)
        try:
            ocr_result = await self.get_ocr_result(image_hash)
            if ocr_result:
                return ocr_result

            result = self._ocr.ocr(img_bytes, cls=False)
            ocr_result = result[0]
            await self.save_ocr_result_to_cache(image_hash, ocr_result)
            return ocr_result
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return None
