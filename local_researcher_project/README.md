# (prototype) SparkleForge ⚒️✨ 

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-orange.svg)](https://openrouter.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash%20Lite-purple.svg)](https://ai.google.dev/)

> **Where Ideas Sparkle and Get Forged** ⚒️✨
> 
> Revolutionary multi-agent system that forges sparkling insights through real-time collaboration, 
> creative AI, and 8 core innovations that make every idea sparkle.

## 🔥 What Makes SparkleForge Special?

Unlike traditional research tools, SparkleForge simulates a **team of master craftsmen** working together in a digital forge, each with specialized expertise. Watch as multiple AI agents collaborate like skilled artisans, forging raw information into pure knowledge with sparks of creativity flying everywhere.

### Key Features

- ⚒️ **Multi-Agent Forge**: 5+ specialized AI craftsmen working together
- ✨ **Real-Time Sparkling**: Watch ideas sparkle and get forged live
- 🧠 **Creative Synthesis**: AI generates novel solutions by combining ideas
- 🔍 **Source Validation**: Every claim is verified with credibility scores
- 📚 **Research Memory**: Learns from past forges to improve over time
- 🎯 **Production Ready**: Enterprise-grade reliability and error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sparkleforge.git
cd sparkleforge

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your OpenRouter API key
```

### Basic Usage

```bash
# Start the forge interface
streamlit run src/web/streamlit_app.py

# Or use command line
python main.py --request "Latest AI trends in 2025"
```

## ⚒️ The Forge Process

### 1. **Raw Material Collection** (Information Gathering)
- Multiple AI agents scour the web like prospectors
- Real-time streaming shows each agent's progress
- Raw information is collected and catalogued

### 2. **Heating & Melting** (Data Processing)
- Information is processed and analyzed
- Hierarchical compression removes impurities
- Multi-model orchestration ensures quality

### 3. **Forging & Shaping** (Synthesis)
- Creative AI agents hammer ideas together
- Cross-domain synthesis creates new alloys
- Continuous verification ensures purity

### 4. **Polishing & Finishing** (Final Output)
- Results are polished to perfection
- Citations and sources are properly attributed
- Final deliverable sparkles with quality

## ✨ Core Innovations

### 1. **Adaptive Forge Master** (혁신 1)
- Dynamically allocates 1-10 craftsmen based on complexity
- Real-time quality monitoring of each craftsman's work
- Fast-track mode for simple forging tasks
- Auto-retry mechanism for failed craftsmen
- Priority-based execution for important orders

### 2. **Hierarchical Refinement** (혁신 2)
- 3-stage refinement: Raw → Intermediate → Pure (minimizes loss)
- Importance-based preservation of core elements
- Refinement validation: Pre/post consistency verification
- Refinement history: Version storage for each stage

### 3. **Multi-Model Forge** (혁신 3)
- Role-based model selection for each task type
- Dynamic model switching based on material difficulty
- Cost optimization within budget constraints
- Weighted ensemble: Confidence-based combination

### 4. **Continuous Quality Control** (혁신 4)
- 3-stage verification:
  1. Self-Verification (internal consistency)
  2. Cross-Verification (cross-source validation)
  3. External-Verification (external database verification)
- Confidence scoring for every piece of information
- Early warning system for low-quality materials
- Fact-checking for major claims
- Uncertainty declaration for unclear parts

### 5. **Streaming Forge** (혁신 5)
- Real-time streaming of forging progress
- Progressive reporting with partial results
- Pipeline parallelization: Simultaneous processing
- Incremental save: Continuous saving of intermediate results

### 6. **Universal Tool Forge** (혁신 6)
- 100+ tools via Model Context Protocol
- OpenRouter + Gemini 2.5 Flash Lite integration
- Smart tool selection and rate limiting
- Health monitoring of all forge equipment

### 7. **Adaptive Workspace** (혁신 7)
- Dynamic adjustment from 2K to 1M tokens
- Importance-based information preservation
- Auto-compression of less important parts
- Long-term memory with searchable history

### 8. **Production-Grade Forge** (혁신 8)
- Circuit breaker pattern for fault tolerance
- Exponential backoff and state persistence
- Comprehensive logging and health monitoring
- Graceful degradation when some tools fail

## 🎨 Creative Forge Features

### Creative Synthesis Forge
The system includes a specialized **Creative Forge** that:

- **Analogical Reasoning**: Draws parallels from different domains
- **Cross-Domain Synthesis**: Combines principles from different fields
- **Lateral Thinking**: Challenges conventional approaches
- **Idea Combination**: Merges existing ideas into novel solutions

Example output:
```
✨ Nature-Inspired Forging Approach
   Apply evolutionary principles to research methodology, 
   allowing ideas to adapt and evolve through iterative refinement.
   Confidence: 85% | Novelty: 78% | Applicability: 82%
```

## 🖥️ Forge Interface

The Streamlit web interface provides:

- **Live Forge Dashboard**: Real-time craftsman activity monitoring
- **Creative Forge Page**: Explore AI-generated innovative solutions
- **Source Validation**: Credibility scores and fact-checking
- **Research Memory**: Past forge history and recommendations
- **Data Visualization**: Interactive charts and progress tracking

## 🛠️ Architecture

```
┌─────────────────────────────────────────┐
│           Forge Interface               │
│        (Streamlit + WebSocket)          │
├─────────────────────────────────────────┤
│         Forge Master                    │
│           (LangGraph Workflow)          │
├─────────────────────────────────────────┤
│  Research Craftsman │  Verification     │
│  Planning Craftsman │  Synthesis        │
│  Creative Forge     │  Memory Keeper    │
├─────────────────────────────────────────┤
│         Universal Tool Forge            │
│    (OpenRouter + 100+ Tools)            │
└─────────────────────────────────────────┘
```

## 📊 Performance (example)

| Feature | Traditional Tools | SparkleForge | Improvement |
|---------|------------------|--------------|-------------|
| Response Time | 5-10 minutes | 30-60 seconds | **10x faster** |
| Information Loss | 20-30% | <5% | **90% reduction** |
| Source Verification | Manual | Automatic | **100% automated** |
| Creative Insights | None | AI-generated | **New capability** |
| Real-time Updates | No | Yes | **Live monitoring** |

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
LLM_MODEL=google/gemini-2.5-flash-lite
MAX_SOURCES=20
ENABLE_STREAMING=true
ENABLE_CREATIVE_FORGE=true
```

### Advanced Settings

```python
# Customize craftsman behavior
CRAFTSMAN_MAX_RETRIES=3
CRAFTSMAN_TIMEOUT=300
ENABLE_CRAFTSMAN_COMMUNICATION=true

# Forge settings
MAX_SOURCES=20
SEARCH_TIMEOUT=30
ENABLE_ACADEMIC_FORGE=true
```

## 📈 Use Cases

- **Academic Research**: Comprehensive literature reviews with source validation
- **Business Intelligence**: Market research with creative insights
- **Content Creation**: Well-researched articles with citations
- **Decision Making**: Fact-checked information for important decisions
- **Learning**: Educational research with progressive complexity

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/yourusername/sparkleforge/issues)
- 💬 [Discussions](https://github.com/yourusername/sparkleforge/discussions)
- 📧 [Email Support](mailto:support@sparkleforge.ai)

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai/) for API access
- [Google Gemini](https://ai.google.dev/) for the AI models
- [Streamlit](https://streamlit.io/) for the web interface
- [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow orchestration

## 📊 Project Status

![GitHub stars](https://img.shields.io/github/stars/yourusername/sparkleforge?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/sparkleforge?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/sparkleforge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/sparkleforge)

---

**Ready to forge your ideas into sparkling insights?** ⚒️✨

[Get Started](#quick-start) | [View Demo](https://demo.sparkleforge.ai) | [Read Docs](docs/) | [Join Community](https://discord.gg/sparkleforge)