import logging
from uuid import UUID

import dspy
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from datalayer import get_db_session
from datalayer.model.dto import BusinessLogicDtoGeneric
from datalayer.model.dto.analysis_dto import AnalysisResponseDto, MessageTagAnalysis
from datalayer.repository import MessageRepository, ConversationRepository
from signature.state_signature import StateSignature, MessagesToTags
from services.state_config_service import StateConfigManager

logger = logging.getLogger(__name__)


class AnalysisService:
    """Business logic layer for conversation analysis with message tagging"""

    def __init__(
        self,
        session: AsyncSession = Depends(get_db_session),
    ):
        """
        Initialize service with database session

        Args:
            session: AsyncSession for database operations
        """
        self.message_repository = MessageRepository(session)
        self.conversation_repository = ConversationRepository(session)
        self.session = session

    async def analyze_conversation_messages(
        self,
        conversation_id: UUID,
        project_info: str,
        company_info: str
    ) -> BusinessLogicDtoGeneric:
        """
        Analyze conversation messages and assign tags to each message using DSPy with BusinessLogicDtoGeneric wrapper.

        Args:
            conversation_id: ID of the conversation to analyze
            project_info: Project information for context
            company_info: Company information for context

        Returns:
            BusinessLogicDtoGeneric with analysis results
        """
    
            # Get messages for conversation
        messages_db = await self.message_repository.get_messages_by_conversation_id(
                conversation_id
            )


            # Prepare messages for DSPy analysis
        messages_to_analyze = [
                MessagesToTags(
                    role=msg.message_role,
                    content=msg.message_content
                )
                for msg in messages_db
            ]

            # Fetch states from SharePoint via StateConfigManager
        logger.info("Fetching states from SharePoint...")
        state_config_manager = StateConfigManager()
        states = await state_config_manager.get_states()

        

        logger.info(f"Fetched {len(states)} states from SharePoint")

            # Create DSPy predictor for state analysis
        predictor = dspy.ChainOfThought(StateSignature)

            # Run DSPy analysis
        dspy_result = predictor(
                company_info=company_info,
                project_info=project_info,
                messages_to_tags=messages_to_analyze,
                messages_already_tagged=[],  # No previous tags for now
                states=states  # States from SharePoint
            )

            # Convert DSPy results to MessageTagAnalysis
        message_analyses = []
        state_analyses = dspy_result.state_analysis if hasattr(dspy_result, 'state_analysis') else []

        for idx, msg_db in enumerate(messages_db):
                # Get corresponding analysis result
                analysis_result = state_analyses[idx] if idx < len(state_analyses) else None

                if analysis_result:
                    tag = analysis_result.sub_tag_key if hasattr(analysis_result, 'sub_tag_key') else "unknown"
                    reasoning = analysis_result.tagReasoning if hasattr(analysis_result, 'tagReasoning') else ""
                    confidence = analysis_result.confidenceScore if hasattr(analysis_result, 'confidenceScore') else 0.0
                else:
                    tag = "unanalyzed"
                    reasoning = "Could not analyze this message"
                    confidence = 0.0

                analysis = MessageTagAnalysis(
                    tag=tag,
                    tag_reasoning=reasoning,
                    confidence_score=confidence
                )
                message_analyses.append(analysis)

        # Convert to AnalysisResponseDto (which is now a list)
        result = AnalysisResponseDto(message_analyses)

        logger.info(f"Analyzed {len(messages_db)} messages for conversation {conversation_id} using DSPy")

        return BusinessLogicDtoGeneric(
            data=result,
            is_success=True
        )

