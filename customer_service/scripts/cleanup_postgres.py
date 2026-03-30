from __future__ import annotations

from customer_service.persistence.postgres import AsyncConversationRepository


def main() -> None:
    repository = AsyncConversationRepository()
    try:
        deleted_count = repository.cleanup_expired_turns()
        print(f"PostgreSQL 清理完成，删除 {deleted_count} 条过期会话记录。")
    finally:
        repository.close()


if __name__ == "__main__":
    main()