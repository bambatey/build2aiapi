from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Union, Any, Dict
from contextlib import asynccontextmanager, contextmanager
import asyncio
from datetime import datetime

# SQLAlchemy imports
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from ._repository_abc import RepositoryABC, AsyncRepositoryABC
# Type variables
T = TypeVar('T')
PrimaryKeyType = Union[int, str]

class BaseRepository(RepositoryABC[T]):
    """Synchronous SQLAlchemy repository implementation"""
    
    def __init__(self, session: Session, model_class: type[T]):
        self.session = session
        self.model_class = model_class
    
    def get_by_id(self, id: PrimaryKeyType) -> Optional[T]:
        """Get entity by primary key"""
        stmt = select(self.model_class).where(self.model_class.id == id)
        return self.session.execute(stmt).scalar_one_or_none()
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """Get all entities with optional pagination"""
        stmt = select(self.model_class)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return self.session.execute(stmt).scalars().all()
    
    def find_by(self, **filters) -> List[T]:
        """Find entities by filter criteria"""
        stmt = select(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                if isinstance(value, (list, tuple)):
                    stmt = stmt.where(getattr(self.model_class, key).in_(value))
                elif isinstance(value, dict) and 'op' in value:
                    # Support for complex operations: {'op': 'like', 'value': '%test%'}
                    op = value['op']
                    val = value['value']
                    attr = getattr(self.model_class, key)
                    if op == 'like':
                        stmt = stmt.where(attr.like(val))
                    elif op == 'ilike':
                        stmt = stmt.where(attr.ilike(val))
                    elif op == 'gt':
                        stmt = stmt.where(attr > val)
                    elif op == 'gte':
                        stmt = stmt.where(attr >= val)
                    elif op == 'lt':
                        stmt = stmt.where(attr < val)
                    elif op == 'lte':
                        stmt = stmt.where(attr <= val)
                    elif op == 'ne':
                        stmt = stmt.where(attr != val)
                else:
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        return self.session.execute(stmt).scalars().all()
    
    def find_one_by(self, **filters) -> Optional[T]:
        """Find single entity by filter criteria"""
        results = self.find_by(**filters)
        if len(results) > 1:
            raise MultipleResultsFound(f"Multiple results found for {filters}")
        return results[0] if results else None
    
    def save(self, entity: T) -> T:
        """Save (insert or update) entity"""
        self.session.add(entity)
        self.session.flush()
        self.session.refresh(entity)
        return entity
    
    def save_all(self, entities: List[T]) -> List[T]:
        """Save multiple entities"""
        self.session.add_all(entities)
        self.session.flush()
        for entity in entities:
            self.session.refresh(entity)
        return entities
    
    def delete(self, entity: T) -> None:
        """Delete entity"""
        self.session.delete(entity)
        self.session.flush()
    
    def delete_by_id(self, id: PrimaryKeyType) -> bool:
        """Delete entity by ID, returns True if deleted"""
        entity = self.get_by_id(id)
        if entity:
            self.delete(entity)
            return True
        return False
    
    def exists(self, id: PrimaryKeyType) -> bool:
        """Check if entity exists by ID"""
        stmt = select(1).where(self.model_class.id == id)
        return self.session.execute(stmt).scalar() is not None
    
    def count(self, **filters) -> int:
        """Count entities with optional filters"""
        from sqlalchemy import func
        stmt = select(func.count(self.model_class.id))
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                stmt = stmt.where(getattr(self.model_class, key) == value)
        return self.session.execute(stmt).scalar()
    
    @contextmanager
    def transaction(self):
        """Context manager for handling transactions"""
        try:
            yield self
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise


class AsyncBaseRepository(AsyncRepositoryABC[T]):
    """Asynchronous SQLAlchemy repository implementation"""
    
    def __init__(self, session: AsyncSession, model_class: type[T]):
        self.session = session
        self.model_class = model_class
    
    async def get_by_id(self, id: PrimaryKeyType) -> Optional[T]:
        """Get entity by primary key"""
        stmt = select(self.model_class).where(self.model_class.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """Get all entities with optional pagination"""
        stmt = select(self.model_class)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def find_by(self, **filters) -> List[T]:
        """Find entities by filter criteria"""
        stmt = select(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                if isinstance(value, (list, tuple)):
                    stmt = stmt.where(getattr(self.model_class, key).in_(value))
                elif isinstance(value, dict) and 'op' in value:
                    # Support for complex operations
                    op = value['op']
                    val = value['value']
                    attr = getattr(self.model_class, key)
                    if op == 'like':
                        stmt = stmt.where(attr.like(val))
                    elif op == 'ilike':
                        stmt = stmt.where(attr.ilike(val))
                    elif op == 'gt':
                        stmt = stmt.where(attr > val)
                    elif op == 'gte':
                        stmt = stmt.where(attr >= val)
                    elif op == 'lt':
                        stmt = stmt.where(attr < val)
                    elif op == 'lte':
                        stmt = stmt.where(attr <= val)
                    elif op == 'ne':
                        stmt = stmt.where(attr != val)
                else:
                    stmt = stmt.where(getattr(self.model_class, key) == value)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def find_one_by(self, **filters) -> Optional[T]:
        """Find single entity by filter criteria"""
        results = await self.find_by(**filters)
        if len(results) > 1:
            raise MultipleResultsFound(f"Multiple results found for {filters}")
        return results[0] if results else None
    
    async def save(self, entity: T) -> T:
        """Save (insert or update) entity"""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def save_all(self, entities: List[T]) -> List[T]:
        """Save multiple entities"""
        self.session.add_all(entities)
        await self.session.flush()
        for entity in entities:
            await self.session.refresh(entity)
        return entities
    
    async def delete(self, entity: T) -> None:
        """Delete entity"""
        await self.session.delete(entity)
        await self.session.flush()
    
    async def delete_by_id(self, id: PrimaryKeyType) -> bool:
        """Delete entity by ID, returns True if deleted"""
        entity = await self.get_by_id(id)
        if entity:
            await self.delete(entity)
            return True
        return False
    
    async def exists(self, id: PrimaryKeyType) -> bool:
        """Check if entity exists by ID"""
        stmt = select(1).where(self.model_class.id == id)
        result = await self.session.execute(stmt)
        return result.scalar() is not None
    
    async def count(self, **filters) -> int:
        """Count entities with optional filters"""
        from sqlalchemy import func
        stmt = select(func.count(self.model_class.id))
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                stmt = stmt.where(getattr(self.model_class, key) == value)
        result = await self.session.execute(stmt)
        return result.scalar()
    
    @asynccontextmanager
    async def transaction(self):
        """Async context manager for handling transactions"""
        try:
            yield self
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise


# =============================================================================
# Repository Factory Classes
# =============================================================================

class RepositoryFactory:
    """Factory for creating synchronous repositories"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_repository(self, model_class: type[T]) -> BaseRepository[T]:
        """Create a repository for the given model class"""
        return BaseRepository(self.session, model_class)


class AsyncRepositoryFactory:
    """Factory for creating asynchronous repositories"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    def get_repository(self, model_class: type[T]) -> AsyncBaseRepository[T]:
        """Create an async repository for the given model class"""
        return AsyncBaseRepository(self.session, model_class)