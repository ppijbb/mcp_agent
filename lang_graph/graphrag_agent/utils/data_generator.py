"""
Data Generator for GraphRAG Agent

This module provides utilities for generating sample data in various formats
to test the dynamic graph generation capabilities.
"""

import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path


class DataGenerator:
    """Generate sample data for testing graph generation"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_tech_company_data(self, output_path: str = "tech_companies.csv") -> str:
        """Generate technology company data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "document_id": ["tech_1", "tech_1", "tech_2", "tech_2", "tech_3", "tech_3", "tech_4", "tech_4", "tech_5", "tech_5"],
            "content": [
                "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
                "Apple is known for designing and manufacturing consumer electronics, software, and online services. Tim Cook is the current CEO of Apple.",
                "Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. It was founded by Bill Gates and Paul Allen in 1975.",
                "Microsoft develops, manufactures, licenses, supports, and sells computer software, consumer electronics, and personal computers. Satya Nadella is the current CEO.",
                "Google LLC is an American multinational technology company that specializes in Internet-related services and products. It was founded by Larry Page and Sergey Brin in 1998.",
                "Google's parent company is Alphabet Inc., and Sundar Pichai is the current CEO of both Google and Alphabet.",
                "Amazon.com Inc. is an American multinational technology company that focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence.",
                "Amazon was founded by Jeff Bezos in 1994 and is headquartered in Seattle, Washington. Andy Jassy is the current CEO.",
                "Tesla Inc. is an American electric vehicle and clean energy company founded by Elon Musk, JB Straubel, Martin Eberhard, and Marc Tarpenning in 2003.",
                "Tesla is headquartered in Austin, Texas, and is known for its electric vehicles, energy storage systems, and solar panels."
            ],
            "category": ["company", "company", "company", "company", "company", "company", "company", "company", "company", "company"],
            "year": [1976, 1976, 1975, 1975, 1998, 1998, 1994, 1994, 2003, 2003]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Tech company data generated: {output_path}")
        return output_path
    
    def generate_scientific_research_data(self, output_path: str = "scientific_research.csv") -> str:
        """Generate scientific research data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["research_1", "research_1", "research_2", "research_2", "research_3", "research_3", "research_4", "research_4"],
            "abstract": [
                "Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. Machine learning is a subset of AI that enables computers to learn from data.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
                "Quantum computing leverages quantum mechanical phenomena to perform calculations. Unlike classical computers, quantum computers use quantum bits (qubits) that can exist in superposition.",
                "Quantum algorithms like Shor's algorithm and Grover's algorithm offer exponential speedups for certain problems. IBM, Google, and Microsoft are leading quantum computing research.",
                "Climate change is one of the most pressing challenges of our time. Rising global temperatures are causing sea level rise, extreme weather events, and ecosystem disruption.",
                "Renewable energy sources like solar and wind power are crucial for mitigating climate change. The Paris Agreement aims to limit global warming to well below 2 degrees Celsius.",
                "CRISPR-Cas9 is a revolutionary gene-editing technology that allows precise modification of DNA sequences. It has potential applications in treating genetic diseases and improving crops.",
                "The Human Genome Project successfully mapped the entire human genome. This breakthrough has enabled personalized medicine and advanced our understanding of genetic diseases."
            ],
            "field": ["AI", "AI", "Quantum Computing", "Quantum Computing", "Climate Science", "Climate Science", "Biotechnology", "Genomics"],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Scientific research data generated: {output_path}")
        return output_path
    
    def generate_news_articles_data(self, output_path: str = "news_articles.csv") -> str:
        """Generate news articles data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "document_id": ["news_1", "news_2", "news_3", "news_4", "news_5", "news_6", "news_7", "news_8", "news_9", "news_10"],
            "headline": [
                "Apple Announces New iPhone with Advanced AI Features",
                "Microsoft Partners with OpenAI to Enhance Azure Services",
                "Google Launches New Quantum Computing Research Center",
                "Tesla Reports Record Electric Vehicle Sales in Q3",
                "Amazon Expands Cloud Computing Services Globally",
                "Meta Introduces New Virtual Reality Headset",
                "Netflix Invests in Korean Content Production",
                "SpaceX Successfully Launches Starlink Satellites",
                "NVIDIA Develops New AI Chip for Autonomous Vehicles",
                "Uber Implements Self-Driving Car Technology"
            ],
            "content": [
                "Apple Inc. unveiled its latest iPhone model featuring advanced artificial intelligence capabilities. The new device includes enhanced camera systems and improved machine learning processing.",
                "Microsoft Corporation announced a strategic partnership with OpenAI to integrate advanced AI models into its Azure cloud platform. This collaboration aims to accelerate AI adoption in enterprise environments.",
                "Google LLC opened a new quantum computing research center in California. The facility will focus on developing quantum algorithms and hardware for practical applications.",
                "Tesla Inc. reported record-breaking electric vehicle sales in the third quarter. The company delivered over 400,000 vehicles worldwide, exceeding previous records.",
                "Amazon.com Inc. announced the expansion of its cloud computing services to new regions. The AWS platform will now be available in additional countries across Asia and Europe.",
                "Meta Platforms Inc. introduced its latest virtual reality headset with improved display technology. The new device offers enhanced immersive experiences for users.",
                "Netflix Inc. announced significant investment in Korean content production. The streaming platform plans to produce over 50 new Korean series and films.",
                "SpaceX successfully launched another batch of Starlink satellites into orbit. The mission aims to expand global internet coverage through satellite constellation.",
                "NVIDIA Corporation developed a new AI chip specifically designed for autonomous vehicles. The processor offers improved performance for real-time decision making in self-driving cars.",
                "Uber Technologies Inc. implemented new self-driving car technology in select cities. The ride-sharing company is testing autonomous vehicles for commercial use."
            ],
            "category": ["Technology", "Technology", "Technology", "Automotive", "Cloud Computing", "Virtual Reality", "Entertainment", "Space", "AI Hardware", "Transportation"],
            "date": ["2023-09-15", "2023-09-20", "2023-09-25", "2023-10-01", "2023-10-05", "2023-10-10", "2023-10-15", "2023-10-20", "2023-10-25", "2023-10-30"]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"News articles data generated: {output_path}")
        return output_path
    
    def generate_social_media_data(self, output_path: str = "social_media.csv") -> str:
        """Generate social media posts data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "document_id": ["post_1", "post_2", "post_3", "post_4", "post_5", "post_6", "post_7", "post_8", "post_9", "post_10"],
            "message": [
                "Just tried the new iPhone 15 Pro and the camera quality is incredible! Apple really outdid themselves this time. #iPhone15 #Apple #Photography",
                "Microsoft's new AI features in Office 365 are game-changing for productivity. The writing suggestions are surprisingly accurate! #Microsoft #AI #Productivity",
                "Google's quantum computing breakthrough could revolutionize cryptography. The implications for cybersecurity are enormous. #Google #QuantumComputing #Security",
                "Tesla's autopilot feature saved me from a potential accident today. The technology is getting really impressive! #Tesla #Autopilot #Safety",
                "Amazon's same-day delivery never fails to amaze me. Ordered this morning, delivered this afternoon! #Amazon #Delivery #Convenience",
                "Meta's VR headset made me feel like I was actually in another world. The future of entertainment is here! #Meta #VR #Entertainment",
                "Netflix's Korean dramas are absolutely addictive. The storytelling and production quality are top-notch! #Netflix #KoreanDrama #Entertainment",
                "SpaceX's rocket launch was spectacular to watch. The future of space exploration is exciting! #SpaceX #Space #Innovation",
                "NVIDIA's new graphics card runs AI models like a dream. Perfect for machine learning projects! #NVIDIA #AI #Hardware",
                "Uber's self-driving car test ride was smooth and safe. The future of transportation is autonomous! #Uber #SelfDriving #Transportation"
            ],
            "platform": ["Twitter", "LinkedIn", "Twitter", "Instagram", "Facebook", "Twitter", "Instagram", "Twitter", "LinkedIn", "Facebook"],
            "likes": [150, 89, 203, 312, 67, 178, 245, 156, 134, 98],
            "shares": [23, 12, 45, 67, 8, 34, 56, 29, 19, 15]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Social media data generated: {output_path}")
        return output_path
    
    def generate_custom_data(self, data_config: Dict[str, Any], output_path: str) -> str:
        """Generate custom data based on configuration"""
        try:
            # Create DataFrame from configuration
            df = pd.DataFrame(data_config)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Custom data generated: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to generate custom data: {e}")
            raise
    
    def generate_medical_data(self, output_path: str = "medical_data.csv") -> str:
        """Generate medical and healthcare data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["medical_1", "medical_1", "medical_2", "medical_2", "medical_3", "medical_3", "medical_4", "medical_4"],
            "content": [
                "Dr. Sarah Johnson is a cardiologist at Johns Hopkins Hospital in Baltimore. She specializes in treating heart disease and has published over 50 research papers.",
                "The hospital recently acquired new MRI machines for advanced cardiac imaging. This technology helps doctors diagnose conditions more accurately.",
                "The American Heart Association recommends regular exercise and a healthy diet to prevent cardiovascular disease. These lifestyle changes can reduce risk by up to 30%.",
                "Dr. Michael Chen conducted a clinical trial on a new diabetes medication. The study involved 500 patients over two years and showed promising results.",
                "Mayo Clinic in Rochester, Minnesota is renowned for its cancer treatment programs. The facility uses cutting-edge immunotherapy techniques.",
                "The World Health Organization declared COVID-19 a pandemic in March 2020. This led to unprecedented global health measures and vaccine development.",
                "Dr. Emily Rodriguez works at the National Institutes of Health in Bethesda. Her research focuses on genetic disorders and personalized medicine.",
                "The FDA approved a new Alzheimer's drug after extensive clinical trials. This breakthrough offers hope for millions of patients worldwide."
            ],
            "category": ["medical", "medical", "medical", "medical", "medical", "medical", "medical", "medical"],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Medical data generated: {output_path}")
        return output_path
    
    def generate_art_culture_data(self, output_path: str = "art_culture.csv") -> str:
        """Generate art and culture data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["art_1", "art_1", "art_2", "art_2", "art_3", "art_3", "art_4", "art_4"],
            "content": [
                "Leonardo da Vinci painted the Mona Lisa in the early 16th century. This masterpiece is now housed in the Louvre Museum in Paris.",
                "The Renaissance period marked a cultural rebirth in Europe. Artists like Michelangelo and Raphael created timeless works of art.",
                "The Metropolitan Museum of Art in New York City houses one of the world's largest art collections. It attracts millions of visitors annually.",
                "Shakespeare wrote Romeo and Juliet in the late 16th century. This tragic love story remains one of the most performed plays worldwide.",
                "The Sydney Opera House in Australia is an architectural masterpiece designed by Jørn Utzon. It opened in 1973 and is a UNESCO World Heritage site.",
                "Pablo Picasso co-founded the Cubist movement in the early 20th century. His innovative style revolutionized modern art.",
                "The Broadway theater district in New York is famous for musical productions. Shows like Hamilton and The Lion King have achieved worldwide success.",
                "The British Museum in London contains artifacts from ancient civilizations. The Rosetta Stone is one of its most famous exhibits."
            ],
            "category": ["art", "art", "art", "art", "art", "art", "art", "art"],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Art and culture data generated: {output_path}")
        return output_path
    
    def generate_sports_data(self, output_path: str = "sports_data.csv") -> str:
        """Generate sports and athletics data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["sports_1", "sports_1", "sports_2", "sports_2", "sports_3", "sports_3", "sports_4", "sports_4"],
            "content": [
                "Lionel Messi won the FIFA World Cup with Argentina in 2022. This was his first World Cup victory after a legendary career.",
                "The Olympic Games were held in Tokyo in 2021. Despite the pandemic, athletes from around the world competed in various sports.",
                "Serena Williams retired from professional tennis in 2022. She won 23 Grand Slam singles titles during her career.",
                "The Super Bowl is the championship game of the National Football League. It's one of the most-watched television events in the United States.",
                "Usain Bolt holds the world record for the 100-meter dash. His time of 9.58 seconds was set at the 2009 World Championships in Berlin.",
                "The Tour de France is an annual cycling race held in France. It covers over 3,500 kilometers and lasts three weeks.",
                "Michael Phelps won 28 Olympic medals in swimming. He is the most decorated Olympian of all time.",
                "The FIFA World Cup is held every four years. The 2026 tournament will be hosted by the United States, Canada, and Mexico."
            ],
            "category": ["sports", "sports", "sports", "sports", "sports", "sports", "sports", "sports"],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Sports data generated: {output_path}")
        return output_path
    
    def generate_education_data(self, output_path: str = "education_data.csv") -> str:
        """Generate education and academic data"""
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "document_id": ["edu_1", "edu_1", "edu_2", "edu_2", "edu_3", "edu_3", "edu_4", "edu_4"],
            "content": [
                "Harvard University was founded in 1636 in Cambridge, Massachusetts. It is the oldest institution of higher education in the United States.",
                "Professor Jane Smith teaches computer science at Stanford University. Her research focuses on artificial intelligence and machine learning.",
                "The University of Oxford in England is one of the world's oldest universities. It has produced numerous Nobel Prize winners and world leaders.",
                "Online learning platforms like Coursera and edX have revolutionized education. They offer courses from top universities worldwide.",
                "The International Baccalaureate program is offered in schools across 150 countries. It provides a rigorous curriculum for students aged 3-19.",
                "Dr. Robert Johnson published a groundbreaking study on educational psychology. His work has influenced teaching methods globally.",
                "The Massachusetts Institute of Technology is renowned for its engineering programs. Many successful entrepreneurs graduated from MIT.",
                "The United Nations Educational, Scientific and Cultural Organization promotes education worldwide. UNESCO works to ensure quality education for all."
            ],
            "category": ["education", "education", "education", "education", "education", "education", "education", "education"],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Education data generated: {output_path}")
        return output_path
    
    def generate_all_sample_data(self, output_dir: str = ".") -> List[str]:
        """Generate all sample data files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        # Generate different types of sample data
        generators = [
            ("tech_companies.csv", self.generate_tech_company_data),
            ("scientific_research.csv", self.generate_scientific_research_data),
            ("news_articles.csv", self.generate_news_articles_data),
            ("social_media.csv", self.generate_social_media_data),
            ("medical_data.csv", self.generate_medical_data),
            ("art_culture.csv", self.generate_art_culture_data),
            ("sports_data.csv", self.generate_sports_data),
            ("education_data.csv", self.generate_education_data)
        ]
        
        for filename, generator_func in generators:
            output_path = output_dir / filename
            try:
                generated_file = generator_func(str(output_path))
                generated_files.append(generated_file)
            except Exception as e:
                self.logger.error(f"Failed to generate {filename}: {e}")
        
        self.logger.info(f"Generated {len(generated_files)} sample data files")
        return generated_files


def main():
    """Generate all sample data files"""
    generator = DataGenerator()
    files = generator.generate_all_sample_data()
    
    print("Generated sample data files:")
    for file in files:
        print(f"  - {file}")


if __name__ == "__main__":
    main()
