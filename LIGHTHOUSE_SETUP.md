# 🚀 SEO Doctor Lighthouse 설정 가이드

이 가이드는 SEO Doctor에서 실제 Playwright-Lighthouse 분석을 사용하기 위한 설정 방법을 안내합니다.

## 📋 필수 요구사항

### 1. Node.js 설치
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS (Homebrew)
brew install node

# Windows
# https://nodejs.org에서 다운로드
```

### 2. Chrome/Chromium 설치
```bash
# Ubuntu/Debian
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt-get update
sudo apt-get install google-chrome-stable

# macOS
brew install --cask google-chrome

# Windows
# https://www.google.com/chrome에서 다운로드
```

## 🔧 Lighthouse 설치

### 전역 설치 (권장)
```bash
npm install -g lighthouse chrome-launcher
```

### 프로젝트별 설치
```bash
cd /path/to/mcp_agent
npm install lighthouse chrome-launcher
```

## 📦 Python 의존성 설치

```bash
cd /path/to/mcp_agent
pip install -r requirements.txt
```

## ✅ 설치 확인

### 1. Node.js 및 Lighthouse 확인
```bash
node --version  # v18.0.0 이상
lighthouse --version  # 10.0.0 이상
```

### 2. Chrome 헤드리스 모드 확인
```bash
google-chrome --headless --disable-gpu --no-sandbox --version
```

### 3. Python 모듈 확인
```python
python -c "
from srcs.seo_doctor.lighthouse_analyzer import analyze_website_with_lighthouse
print('Lighthouse 분석기 로드 성공!')
"
```

## 🚀 사용 방법

### 1. SEO Doctor 실행
```bash
cd /path/to/mcp_agent
streamlit run pages/seo_doctor.py
```

### 2. 웹 브라우저에서 접속
- URL: http://localhost:8501
- URL 입력 후 "실시간 SEO 진단 시작" 클릭

### 3. 분석 결과 확인
- **실시간 분석**: 30-60초 소요
- **상세 보고서**: Core Web Vitals, SEO 점수 등
- **개선 사항**: 구체적인 문제점과 해결 방안

## 🔍 문제 해결

### Chrome 실행 오류
```bash
# 권한 문제 해결
sudo chmod +x /usr/bin/google-chrome

# 샌드박스 비활성화 (Docker 환경)
export CHROME_ARGS="--no-sandbox --disable-dev-shm-usage"
```

### Node.js 권한 오류
```bash
# npm 전역 설치 권한 설정
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Lighthouse 실행 오류
```bash
# 포트 충돌 해결
sudo lsof -ti:9222 | xargs kill -9

# 임시 디렉토리 정리
rm -rf /tmp/lighthouse-*
```

## 🎯 성능 최적화

### 1. 시스템 리소스 설정
```bash
# 메모리 제한 증가
export NODE_OPTIONS="--max-old-space-size=4096"

# Chrome 프로세스 제한
export CHROME_ARGS="--max_old_space_size=4096 --disable-dev-shm-usage"
```

### 2. 분석 속도 향상
- **SSD 사용** 권장
- **8GB+ RAM** 권장  
- **안정적인 인터넷 연결** 필요

## 📊 기능 비교

| 기능 | 더미 데이터 | Lighthouse 실분석 |
|------|-------------|------------------|
| 분석 속도 | 즉시 | 30-60초 |
| 정확도 | 시뮬레이션 | Google 공식 |
| Core Web Vitals | ❌ | ✅ |
| SEO 점수 | 랜덤 | 실제 측정 |
| 접근성 진단 | ❌ | ✅ |
| Best Practices | ❌ | ✅ |

## 🎉 완료!

모든 설정이 완료되면 SEO Doctor에서 실제 Google Lighthouse 엔진을 사용한 정확한 SEO 분석을 수행할 수 있습니다.

더미 데이터 대신 실제 웹사이트 성능 데이터를 기반으로 한 신뢰할 수 있는 진단 결과를 제공합니다. 