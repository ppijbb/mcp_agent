import os
import json
from datetime import datetime
from typing import Dict, List, Any

def load_destination_options() -> List[str]:
    """목적지 옵션 로드"""
    return [
        # 아시아 주요 도시
        "Seoul (서울)",
        "Tokyo (도쿄)",
        "Osaka (오사카)",
        "Bangkok (방콕)",
        "Singapore (싱가포르)",
        "Hong Kong (홍콩)",
        "Shanghai (상하이)",
        "Beijing (베이징)",
        "Taipei (타이베이)",
        "Kuala Lumpur (쿠알라룸푸르)",
        "Manila (마닐라)",
        "Ho Chi Minh City (호치민)",
        "Jakarta (자카르타)",
        
        # 유럽 주요 도시
        "London (런던)",
        "Paris (파리)",
        "Rome (로마)",
        "Barcelona (바르셀로나)",
        "Amsterdam (암스테르담)",
        "Berlin (베를린)",
        "Vienna (비엔나)",
        "Prague (프라하)",
        "Zurich (취리히)",
        "Stockholm (스톡홀름)",
        "Copenhagen (코펜하겐)",
        "Oslo (오슬로)",
        
        # 북미 주요 도시
        "New York (뉴욕)",
        "Los Angeles (로스앤젤레스)",
        "San Francisco (샌프란시스코)",
        "Las Vegas (라스베이거스)",
        "Chicago (시카고)",
        "Miami (마이애미)",
        "Toronto (토론토)",
        "Vancouver (밴쿠버)",
        
        # 오세아니아
        "Sydney (시드니)",
        "Melbourne (멜버른)",
        "Auckland (오클랜드)",
        
        # 중동/아프리카
        "Dubai (두바이)",
        "Istanbul (이스탄불)",
        "Cairo (카이로)",
        "Cape Town (케이프타운)"
    ]

def load_origin_options() -> List[str]:
    """출발지 옵션 로드"""
    return [
        # 한국 주요 도시
        "Seoul (서울)",
        "Busan (부산)",
        "Incheon (인천)",
        "Daegu (대구)",
        "Gwangju (광주)",
        "Daejeon (대전)",
        "Ulsan (울산)",
        "Jeju (제주)",
        
        # 아시아 주요 출발지
        "Tokyo (도쿄)",
        "Osaka (오사카)",
        "Bangkok (방콕)",
        "Singapore (싱가포르)",
        "Hong Kong (홍콩)",
        "Shanghai (상하이)",
        "Beijing (베이징)",
        "Taipei (타이베이)",
        
        # 유럽 주요 출발지
        "London (런던)",
        "Paris (파리)",
        "Frankfurt (프랑크푸르트)",
        "Amsterdam (암스테르담)",
        "Rome (로마)",
        "Barcelona (바르셀로나)",
        
        # 북미 주요 출발지
        "New York (뉴욕)",
        "Los Angeles (로스앤젤레스)",
        "San Francisco (샌프란시스코)",
        "Toronto (토론토)",
        "Vancouver (밴쿠버)",
        
        # 오세아니아
        "Sydney (시드니)",
        "Melbourne (멜버른)",
        
        # 중동
        "Dubai (두바이)",
        "Doha (도하)"
    ]

def get_user_location() -> Dict[str, str]:
    """사용자 위치 기반 기본값 설정"""
    try:
        # 실제 환경에서는 IP 기반 위치 감지 또는 사용자 설정을 사용할 수 있음
        # 현재는 한국을 기본값으로 설정
        default_location = {
            "origin": "Seoul (서울)",
            "country": "South Korea",
            "timezone": "Asia/Seoul",
            "currency": "KRW",
            "language": "ko",
            "detected_method": "default_korean_user"
        }
        
        # 환경 변수나 설정 파일에서 사용자 기본 위치 읽기 시도
        user_origin = os.environ.get('TRAVEL_DEFAULT_ORIGIN', 'Seoul (서울)')
        user_country = os.environ.get('TRAVEL_DEFAULT_COUNTRY', 'South Korea')
        
        return {
            "origin": user_origin,
            "country": user_country,
            "timezone": os.environ.get('TRAVEL_DEFAULT_TIMEZONE', 'Asia/Seoul'),
            "currency": os.environ.get('TRAVEL_DEFAULT_CURRENCY', 'KRW'),
            "language": os.environ.get('TRAVEL_DEFAULT_LANGUAGE', 'ko'),
            "detected_method": "environment_variable" if user_origin != 'Seoul (서울)' else "default_korean_user",
            "available_origins": load_origin_options(),
            "available_destinations": load_destination_options()
        }
        
    except Exception as e:
        # 에러 발생 시 안전한 기본값 반환
        return {
            "origin": "Seoul (서울)",
            "country": "South Korea", 
            "timezone": "Asia/Seoul",
            "currency": "KRW",
            "language": "ko",
            "detected_method": "fallback_default",
            "error": str(e)
        }

def save_travel_report(content: str, filename: str, reports_dir: str = "travel_scout_reports") -> str:
    """여행 검색 보고서를 파일로 저장"""
    try:
        # 디렉토리 생성
        os.makedirs(reports_dir, exist_ok=True)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename.endswith('.md'):
            filename = f"{filename}_{timestamp}.md"
        
        file_path = os.path.join(reports_dir, filename)
        
        # 보고서 헤더 생성
        report_header = f"""# 🧳 Travel Scout Search Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Agent Type**: Travel Scout MCP Agent  
**Report ID**: travel_search_{timestamp}  
**Data Source**: MCP Browser + Real-time Travel Sites

---

"""
        
        # 메타데이터 생성
        metadata = {
            "report_id": f"travel_search_{timestamp}",
            "generated_at": datetime.now().isoformat(),
            "agent_type": "Travel Scout MCP Agent",
            "data_source": "MCP Browser + Real-time Travel Sites",
            "content_length": len(content),
            "file_path": file_path,
            "user_location": get_user_location(),
            "destination_options": load_destination_options(),
            "origin_options": load_origin_options(),
            "report_sections": [
                "Search Summary",
                "Hotel Results",
                "Flight Results", 
                "Price Analysis",
                "Recommendations",
                "Booking Strategy",
                "Total Cost Estimate"
            ]
        }
        
        # Markdown 보고서 저장
        full_content = report_header + content
        
        # 보고서 메타데이터 추가
        full_content += f"\n\n---\n\n### Report Metadata\n\n```json\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n```"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        # 메타데이터 JSON 저장
        metadata_file = file_path.replace('.md', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return file_path
        
    except Exception as e:
        raise Exception(f"여행 보고서 저장 실패: {str(e)}")

def generate_travel_report_content(results: dict, search_params: dict) -> str:
    """여행 검색 보고서 내용 생성"""
    try:
        # 기본 정보 추출
        hotels = results.get('hotels', [])
        flights = results.get('flights', [])
        recommendations = results.get('recommendations', {})
        analysis = results.get('analysis', {})

        # 보고서 내용 생성 (문자열 포매팅)
        content = f"## ✈️🌍 Travel Search Summary for {search_params.get('destination', 'N/A')}\n\n"
        content += f"- **Destination**: {search_params.get('destination', 'N/A')}\n"
        content += f"- **Origin**: {search_params.get('origin', 'N/A')}\n"
        content += f"- **Check-in**: {search_params.get('check_in', 'N/A')}\n"
        content += f"- **Check-out**: {search_params.get('check_out', 'N/A')}\n\n"

        # 호텔 결과
        content += "### 🏨 Hotel Results\n\n"
        if hotels:
            for hotel in hotels[:5]:
                content += f"- **{hotel.get('name', 'N/A')}**\n"
                content += f"  - Price: {hotel.get('price', 'N/A')}\n"
                content += f"  - Rating: {hotel.get('rating', 'N/A')}\n"
                content += f"  - Location: {hotel.get('location', 'N/A')}\n\n"
        else:
            content += "No hotel results found.\n\n"

        # 항공편 결과
        content += "### ✈️ Flight Results\n\n"
        if flights:
            for flight in flights[:5]:
                content += f"- **{flight.get('airline', 'N/A')}**\n"
                content += f"  - Price: {flight.get('price', 'N/A')}\n"
                content += f"  - Duration: {flight.get('duration', 'N/A')}\n"
                content += f"  - Stops: {flight.get('stops', 'N/A')}\n\n"
        else:
            content += "No flight results found.\n\n"
            
        return content
    except Exception as e:
        # Handle potential errors during report generation
        return f"An error occurred while generating the report content: {e}" 