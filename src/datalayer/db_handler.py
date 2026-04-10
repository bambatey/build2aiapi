from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from config import app_config

class DBHandler:

    def __init__(self):
        self.db_engine = None
        self.async_session_local = None

    async def start_db_engine(self):
        self.db_engine = create_async_engine(
            f"postgresql+asyncpg://{app_config.pg_config.user}:{app_config.pg_config.password}@{app_config.pg_config.host}:{app_config.pg_config.port}/{app_config.pg_config.database}",
            echo=False
        )
        self.async_session_local = async_sessionmaker(
            self.db_engine, class_=AsyncSession, expire_on_commit=False
        )

    async def close_db_engine(self):
        if self.db_engine:
            await self.db_engine.dispose()


db_handler_instance = DBHandler()

# Create a lazy session factory that will be initialized when DB starts
AsyncSessionLocal = None


async def get_db_session() -> AsyncSession:
    if db_handler_instance.async_session_local is None:
        raise RuntimeError("Database not initialized. Make sure app_lifespan started the DB engine.")

    async with db_handler_instance.async_session_local() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"Error committing session: {e}")
            raise
        finally:
            await session.close()
