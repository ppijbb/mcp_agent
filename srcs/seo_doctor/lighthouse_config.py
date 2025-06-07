"""
Lighthouse 설정 및 임계값 정의
Medium 글의 base.ts 파일을 Python으로 변환
"""

# 모바일 설정
MOBILE_CONFIG = {
    "setView": "lighthouse:default",
    "extends": "lighthouse:default", 
    "settings": {
        "onlyCategories": ["performance", "seo", "accessibility", "best-practices"],
        "emulatedFormFactor": "mobile",
        "throttling": {
            "rttMs": 40,
            "throughputKbps": 1024,
            "requestLatencyMs": 0,
            "downloadThroughputKbps": 0,
            "uploadThroughputKbps": 0,
            "cpuSlowdownMultiplier": 1,
        },
    },
}

# 데스크탑 설정
DESKTOP_CONFIG = {
    "setView": "lighthouse:default", 
    "extends": "lighthouse:default",
    "settings": {
        "onlyCategories": ["performance", "seo", "accessibility", "best-practices"],
        "emulatedFormFactor": "desktop",
        "throttling": {
            "rttMs": 40,
            "throughputKbps": 10240,
            "requestLatencyMs": 0,
            "downloadThroughputKbps": 0,
            "uploadThroughputKbps": 0,
            "cpuSlowdownMultiplier": 1,
        },
    },
}

# 성능 임계값 설정
THRESHOLDS = {
    "performance": 80,
    "accessibility": 70,
    "best-practices": 70,
    "seo": 80,
}

# 보고서 설정
REPORT_CONFIG = {
    "formats": {
        "json": True,
        "html": False,  # 메모리 절약을 위해 HTML 비활성화
        "csv": False,
    },
} 