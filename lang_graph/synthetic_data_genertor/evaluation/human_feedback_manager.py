"""
Human Feedback Manager for the Kimi-K2 Agentic Data Synthesis System

Manages the collection, storage, and retrieval of human feedback on simulation data.
"""

from typing import Dict, Optional, List
from ..models.evaluation import HumanFeedback
import logging

logger = logging.getLogger(__name__)


class HumanFeedbackManager:
    """
    Manages human feedback for simulation evaluation results.

    Responsibilities:
    - Storing and retrieving human feedback
    - Associating feedback with evaluation results
    - Providing statistics on human feedback
    """

    def __init__(self):
        self._feedback_storage: Dict[str, HumanFeedback] = {}
        logger.info("HumanFeedbackManager initialized.")

    def add_feedback(self, evaluation_id: str, feedback: HumanFeedback) -> str:
        """
        Adds human feedback for a given evaluation result.

        Args:
            evaluation_id: The ID of the evaluation result this feedback pertains to.
            feedback: The HumanFeedback object.

        Returns:
            The ID of the stored feedback object.
        """
        if evaluation_id in self._feedback_storage:
            logger.warning(f"Feedback for evaluation {evaluation_id} already exists. It will be overwritten.")
        
        self._feedback_storage[evaluation_id] = feedback
        logger.info(f"Added human feedback from reviewer '{feedback.reviewer_id}' for evaluation '{evaluation_id}'.")
        return feedback.id

    def get_feedback(self, evaluation_id: str) -> Optional[HumanFeedback]:
        """
        Retrieves human feedback for a given evaluation result.

        Args:
            evaluation_id: The ID of the evaluation result.

        Returns:
            The HumanFeedback object, or None if not found.
        """
        return self._feedback_storage.get(evaluation_id)

    def list_all_feedback(self) -> List[HumanFeedback]:
        """
        Returns a list of all stored human feedback.

        Returns:
            A list of HumanFeedback objects.
        """
        return list(self._feedback_storage.values())

    def get_feedback_statistics(self) -> Dict[str, float]:
        """
        Calculates statistics from the collected human feedback.

        Returns:
            A dictionary containing statistics like average rating.
        """
        if not self._feedback_storage:
            return {
                "total_feedback_count": 0,
                "average_rating": 0.0,
            }

        total_feedback = len(self._feedback_storage)
        average_rating = sum(fb.rating for fb in self._feedback_storage.values()) / total_feedback

        stats = {
            "total_feedback_count": total_feedback,
            "average_rating": average_rating,
        }
        
        logger.info(f"Calculated feedback statistics: {stats}")
        return stats
