# DevOps Productivity Agent

**Production-level DevOps assistant with API integrations** - GitHub, Prometheus, Jenkins integration for enterprise infrastructure automation.

## 🎯 Features

- **🔍 GitHub Repository Analysis**: GitHub API integration for repository metrics, stars, languages
- **🚀 CI/CD Pipeline Monitoring**: GitHub Actions workflow status and results  
- **📊 Infrastructure Health**: Prometheus metrics for CPU, memory, alerts
- **⚡ Performance Analysis**: System performance data from monitoring APIs
- **🔒 Security Scanning**: GitHub security features and vulnerability detection
- **🚨 Incident Response**: Automated response based on monitoring data

## 🛠️ API Integrations

### GitHub API
- Repository analysis and metrics
- Pull request and issue tracking
- GitHub Actions workflow monitoring
- Security vulnerability scanning

### Prometheus API  
- CPU and memory metrics
- Alert monitoring and status
- Performance trend analysis
- Infrastructure health checks

## 🔧 Configuration

### Required Environment Variables

```bash
# GitHub API (Required)
# Set via environment variable: GITHUB_TOKEN
export GITHUB_TOKEN="${GITHUB_TOKEN}"

# Google Gemini API (Required)  
# Set via environment variable: GOOGLE_API_KEY
export GOOGLE_API_KEY="${GOOGLE_API_KEY}"

# Prometheus (Optional - defaults to localhost:9090)
export PROMETHEUS_URL="http://your-prometheus:9090"
```

### GitHub Token Setup
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create token with permissions:
   - `repo` (Full control of private repositories)
   - `read:org` (Read org and team membership)
   - `workflow` (Update GitHub Action workflows)

## 🚀 Installation

```bash
# Clone repository
cd srcs/devops_productivity_agent

# Install dependencies
pip install -r config/requirements.txt

# Set environment variables
export GITHUB_TOKEN="your-github-token"
export GOOGLE_API_KEY="your-google-api-key"
```

## 💻 Usage

### Interactive Mode
```bash
python scripts/run_devops_assistant.py
```

### Programmatic Usage
```python
from agents.devops_assistant_agent import DevOpsAssistantAgent

# Initialize agent
agent = DevOpsAssistantAgent()

# Analyze GitHub repositories
result = await agent.analyze_github_repositories(org="microsoft")

# Monitor CI/CD pipelines  
result = await agent.monitor_ci_cd_pipelines(owner="microsoft", repo="vscode")

# Check infrastructure health
result = await agent.check_infrastructure_health()

# Process natural language requests
result = await agent.process_request("Check our infrastructure health")
```

## 📊 Data Examples

### Repository Analysis
```json
{
  "total_repositories": 285,
  "languages": {
    "TypeScript": 89,
    "Python": 67, 
    "JavaScript": 45
  },
  "stars_total": 285420,
  "repositories": [...]
}
```

### Infrastructure Health
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "cpu_usage_percent": 23.5
  },
  "overall_status": "healthy"
}
```

### Pipeline Status
```json
{
  "total_runs": 156,
  "success_count": 142,
  "failure_count": 14,
  "success_rate": 91.0,
  "recent_runs": [...]
}
```

## 🏗️ Architecture

```
DevOpsAssistantAgent
├── GitHubClient          # GitHub API integration
├── PrometheusClient      # Metrics retrieval  
└── GoogleAugmentedLLM    # AI reasoning engine
```

## 📈 Monitoring Integration

The agent integrates with monitoring systems:

- **Prometheus**: CPU, memory, disk metrics
- **Grafana**: Dashboard data retrieval
- **Kubernetes**: Pod and service status
- **GitHub Actions**: Build and deployment status

## 🔒 Security

- Environment variable based authentication
- No hardcoded credentials
- GitHub token with minimal required permissions
- Secure API key management

## 🎯 Production Ready

This is a production-level implementation with:
- API integrations
- Error handling and retries
- Proper authentication
- Rate limiting compliance
- Structured logging
- Performance monitoring

## 🚨 Troubleshooting

### Common Issues

1. **GitHub API Rate Limiting**
   - Use authenticated requests with token
   - Implement proper rate limiting

2. **Prometheus Connection Issues**
   - Verify PROMETHEUS_URL is accessible
   - Check network connectivity

3. **Environment Variables Not Set**
   - Verify all required variables are exported
   - Check token permissions

### Debug Mode
```bash
export DEBUG=true
python scripts/run_devops_assistant.py
```

---

**Built for enterprise DevOps automation** 🚀 