"""
Sample data generation utilities for GraphRAG Agent

This module provides utilities for creating sample data for testing and demonstration purposes.
"""

import logging
from typing import Optional


def create_sample_data(logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Create sample data file for testing
    
    Args:
        logger: Optional logger instance
        
    Returns:
        str: Path to created sample file, or None if failed
    """
    try:
        import pandas as pd
        
        sample_data = {
            "id": [1, 2, 3, 4, 5],
            "document_id": ["doc_1", "doc_1", "doc_2", "doc_2", "doc_3"],
            "text_unit": [
                "Apple Inc. is a technology company based in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
                "The company is known for innovative products like the iPhone, iPad, and Mac computers. Tim Cook is the current CEO of Apple.",
                "Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. It was founded by Bill Gates and Paul Allen in 1975.",
                "Microsoft is best known for its Windows operating systems, Office productivity suite, and Azure cloud computing platform. Satya Nadella is the current CEO.",
                "Google LLC is an American multinational technology company that specializes in Internet-related services and products. It was founded by Larry Page and Sergey Brin while they were PhD students at Stanford University."
            ]
        }
        
        df = pd.DataFrame(sample_data)
        sample_file = "sample_data.csv"
        df.to_csv(sample_file, index=False)
        
        if logger:
            logger.info(f"Sample data created: {sample_file}")
        else:
            print(f"✅ Sample data created: {sample_file}")
        
        return sample_file
        
    except ImportError:
        error_msg = "pandas is required. Install with: pip install pandas"
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ Error: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"Error creating sample data: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ Error: {error_msg}")
        return None


def create_tech_sample_data(logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Create technology-focused sample data
    
    Args:
        logger: Optional logger instance
        
    Returns:
        str: Path to created sample file, or None if failed
    """
    try:
        import pandas as pd
        
        sample_data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["tech_1", "tech_1", "tech_2", "tech_2", "tech_3", "tech_3", "tech_4", "tech_4"],
            "text_unit": [
                "Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior.",
                "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                "Deep Learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "Natural Language Processing (NLP) focuses on the interaction between computers and human language.",
                "Computer Vision enables machines to interpret and understand visual information from the world.",
                "Robotics combines AI with mechanical engineering to create autonomous machines.",
                "Cloud Computing provides on-demand access to computing resources over the internet.",
                "Blockchain is a distributed ledger technology that maintains a continuously growing list of records."
            ]
        }
        
        df = pd.DataFrame(sample_data)
        sample_file = "tech_sample_data.csv"
        df.to_csv(sample_file, index=False)
        
        if logger:
            logger.info(f"Tech sample data created: {sample_file}")
        else:
            print(f"✅ Tech sample data created: {sample_file}")
        
        return sample_file
        
    except ImportError:
        error_msg = "pandas is required. Install with: pip install pandas"
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ Error: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"Error creating tech sample data: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ Error: {error_msg}")
        return None
