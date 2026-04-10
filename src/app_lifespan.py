from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from datalayer import (
    get_opensearch_client,
    opensearch_handler_instance,
    pg_handler_instance,
    rabbitmq_handler,
    db_handler_instance,
)
import dspy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events for resources like database connections.

    Startup: Initialize database engine
    Shutdown: Close and dispose database engine
    """
    # ---! Startup: Initialize resources
    logger.info("Starting up application...")
    try:
        await pg_handler_instance.start_db_engine()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise

    try:
        # Initialize database
        await db_handler_instance.start_db_engine()
        logger.info("Database engine initialized")

        # Configure DSPy with LM via OpenRouter
        lm_model = app_config.dspy_lm_model
        openrouter_key = app_config.openrouter_api_key

        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        logger.info(f"Configuring DSPy with LM model: {lm_model}")
        dspy.configure(lm=dspy.LM(
            model=lm_model,
            api_key=openrouter_key,
            api_base="https://openrouter.ai/api/v1",
        ))
        logger.info("DSPy LM configured successfully with OpenRouter")

        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise

    try:
        await opensearch_handler_instance.start_opensearch_client()
        logger.info("OpenSearch client initialized successfully")
    except Exception as e:
        logger.error(f"Error during OpenSearch client initialization: {e}")
        raise

    try:
        await rabbitmq_handler.connect()
        logger.info("RabbitMQ connected successfully")
    except Exception as e:
        logger.warning(f"RabbitMQ connection failed (continuing without it): {e}")

    yield  # ---! Application runs here

    # ---! Shutdown: Cleanup resources
    logger.info("Shutting down application...")
    try:
        await pg_handler_instance.close_db_engine()
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")
        raise

    try:
        await db_handler_instance.close_db_engine()
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")
        raise 

    if opensearch_handler_instance.opensearch_client:
        await opensearch_handler_instance.close_opensearch_client()
        logger.info("OpenSearch client closed")

    if rabbitmq_handler and rabbitmq_handler.connection:
        try:
            await rabbitmq_handler.close()
            logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.warning(f"Error closing RabbitMQ connection: {e}")

