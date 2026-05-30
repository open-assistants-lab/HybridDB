"""Async wrappers for HybridDB."""

import asyncio
from typing import Any

from hybriddb.types import Column


class AsyncMixin:
    async def acreate_table(self, table: str, columns: dict[str, str | Column]) -> None:
        await asyncio.to_thread(self.create_table, table, columns)

    async def aadd_column(self, table: str, column: str, col_type: str) -> None:
        await asyncio.to_thread(self.add_column, table, column, col_type)

    async def adrop_column(self, table: str, column: str) -> None:
        await asyncio.to_thread(self.drop_column, table, column)

    async def arename_column(self, table: str, old_name: str, new_name: str) -> None:
        await asyncio.to_thread(self.rename_column, table, old_name, new_name)

    async def ainsert(self, table: str, data: dict, sync: bool = True) -> int | str:
        return await asyncio.to_thread(self.insert, table, data, sync)

    async def ainsert_batch(self, table: str, rows: list[dict], sync: bool = True) -> list[int | str]:
        return await asyncio.to_thread(self.insert_batch, table, rows, sync)

    async def aupdate(self, table: str, row_id: int | str, data: dict, sync: bool = True) -> bool:
        return await asyncio.to_thread(self.update, table, row_id, data, sync)

    async def adelete(self, table: str, row_id: int | str, sync: bool = True) -> bool:
        return await asyncio.to_thread(self.delete, table, row_id, sync)

    async def aget(self, table: str, row_id: int | str) -> dict | None:
        return await asyncio.to_thread(self.get, table, row_id)

    async def aquery(
        self, table: str, where: str = "", params: tuple = (),
        order_by: str = "", limit: int = 100,
    ) -> list[dict]:
        return await asyncio.to_thread(self.query, table, where, params, order_by, limit)

    async def araw_query(self, sql: str, params: tuple = ()) -> list[dict]:
        return await asyncio.to_thread(self.raw_query, sql, params)

    async def aread_query(self, sql: str, params: tuple = ()) -> list[dict]:
        return await asyncio.to_thread(self.read_query, sql, params)

    async def acount(self, table: str, where: str = "", params: tuple = ()) -> int:
        return await asyncio.to_thread(self.count, table, where, params)

    async def asearch(self, table: str, column: str, query: str | None = None, **kwargs: Any) -> list[dict]:
        return await asyncio.to_thread(self.search, table, column, query, **kwargs)

    async def asearch_all(self, table: str, query: str, **kwargs: Any) -> list[dict]:
        return await asyncio.to_thread(self.search_all, table, query, **kwargs)

    async def ahealth(self, table: str) -> dict:
        return await asyncio.to_thread(self.health, table)

    async def areconcile(self, table: str) -> dict:
        return await asyncio.to_thread(self.reconcile, table)

    async def aprocess_journal(self, limit: int = 5000) -> int:
        return await asyncio.to_thread(self.process_journal, limit)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
