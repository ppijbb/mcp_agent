#!/usr/bin/env python3
"""
Travel Search Utilities - Merged and Modernized
utils.py와 travel_utils.py 병합, fallback 코드 제거
"""

import os
import json
import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .config_loader import config

logger = logging.getLogger(__name__)


class TravelSearchError(Exception):
    """여행 검색 관련 커스텀 예외"""
    pass


class TravelSearchUtils:
    """여행 검색 유틸리티 - 통합 및 현대화"""
    
    @staticmethod
    def load_destination_options() -> List[str]:
        """목적지 옵션 로드 - 설정 파일에서"""
        return config.get_destination_options()
    
    @staticmethod
    def load_origin_options() -> List[str]:
        """출발지 옵션 로드 - 설정 파일에서"""
        return config.get_origin_options()
    
    @staticmethod
    def get_user_location() -> Dict[str, str]:
        """사용자 위치 정보 - 설정 파일에서"""
        return config.get_user_location()
    
    @staticmethod
    def validate_search_params(destination: str, check_in: str, check_out: str, 
                              departure_date: str, return_date: str) -> Tuple[bool, str]:
        """여행 검색 파라미터 검증 - 명시적 에러 처리"""
        try:
            # 필수 파라미터 검사
            required_params = {
                'destination': destination,
                'check_in': check_in, 
                'check_out': check_out,
                'departure_date': departure_date,
                'return_date': return_date
            }
            
            for param_name, param_value in required_params.items():
                if not param_value or not param_value.strip():
                    raise TravelSearchError(f"필수 파라미터 누락: {param_name}")
            
            # 날짜 형식 검증
            date_params = {
                'check_in': check_in,
                'check_out': check_out,
                'departure_date': departure_date,
                'return_date': return_date
            }
            
            parsed_dates = {}
            for date_name, date_str in date_params.items():
                try:
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    parsed_dates[date_name] = parsed_date
                except ValueError:
                    raise TravelSearchError(f"잘못된 날짜 형식 {date_name}: {date_str}. 형식: YYYY-MM-DD")
            
            # 날짜 로직 검증
            today = datetime.now().date()
            
            # 과거 날짜 검사
            for date_name, parsed_date in parsed_dates.items():
                if parsed_date.date() < today:
                    raise TravelSearchError(f"{date_name}은 과거 날짜일 수 없습니다: {date_str}")
            
            # 날짜 관계 검증
            if parsed_dates['check_out'] <= parsed_dates['check_in']:
                raise TravelSearchError("체크아웃 날짜는 체크인 날짜보다 늦어야 합니다")
            
            if parsed_dates['return_date'] <= parsed_dates['departure_date']:
                raise TravelSearchError("귀국 날짜는 출발 날짜보다 늦어야 합니다")
            
            # 합리적인 날짜 범위 검사 (1년 이내)
            max_future_date = today + timedelta(days=365)
            for date_name, parsed_date in parsed_dates.items():
                if parsed_date.date() > max_future_date:
                    raise TravelSearchError(f"{date_name}은 너무 먼 미래입니다 (최대 1년)")
            
            # 목적지 형식 검증
            if len(destination.strip()) < 3:
                raise TravelSearchError("목적지는 최소 3자 이상이어야 합니다")
            
            return True, "모든 파라미터가 유효합니다"
            
        except TravelSearchError:
            raise
        except Exception as e:
            logger.error(f"파라미터 검증 오류: {e}")
            raise TravelSearchError(f"검증 오류: {str(e)}")
    
    @staticmethod
    async def retry_operation(operation, max_retries: int = 3, delay: float = 1.0, 
                            backoff: float = 2.0) -> Any:
        """지수 백오프를 사용한 작업 재시도"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"시도 {attempt + 1} 실패: {e}. {wait_time:.1f}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"모든 {max_retries}번 시도 실패. 마지막 오류: {e}")
        
        raise TravelSearchError(f"{max_retries}번 시도 후 작업 실패: {last_exception}")
    
    @staticmethod
    def extract_price_from_text(text: str) -> float:
        """텍스트에서 숫자 가격 값 추출 - 향상된 패턴"""
        if not text:
            raise TravelSearchError("가격 텍스트가 비어있습니다")
        
        try:
            # 향상된 가격 패턴
            price_patterns = [
                r'[\$€£¥₩]\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*[\$€£¥₩]',
                r'([0-9,]+)\s*원',
                r'USD\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'EUR\s*([0-9,]+(?:\.[0-9]{1,2})?)',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*dollars?',
                r'([0-9,]+(?:\.[0-9]{1,2})?)\s*euros?'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    price_str = match.group(1) if match.group(1) else match.group(0)
                    cleaned = re.sub(r'[^\d.]', '', price_str)
                    if cleaned:
                        return float(cleaned)
            
            raise TravelSearchError(f"가격을 추출할 수 없습니다: {text}")
            
        except (ValueError, AttributeError) as e:
            logger.error(f"가격 추출 오류 '{text}': {e}")
            raise TravelSearchError(f"가격 추출 실패: {e}")
    
    @staticmethod
    def extract_rating_from_text(text: str) -> float:
        """텍스트에서 숫자 평점 값 추출 - 검증 포함"""
        if not text:
            raise TravelSearchError("평점 텍스트가 비어있습니다")
        
        try:
            rating_patterns = [
                r'([0-9]\.[0-9])\s*(?:/|\s*out\s+of)\s*([0-9])',  # 4.5/5 or 4.5 out of 5
                r'([0-9]\.[0-9])',  # Simple decimal rating
                r'([0-9])\s*stars?',  # 4 stars
                r'Rating:\s*([0-9]\.[0-9])',  # Rating: 4.5
                r'Score:\s*([0-9]\.[0-9])'  # Score: 4.5
            ]
            
            for pattern in rating_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    rating_value = float(match.group(1))
                    # 평점 범위 검증 (0-5 스케일)
                    if 0 <= rating_value <= 5:
                        return rating_value
                    # 10점 스케일인 경우 변환
                    elif 0 <= rating_value <= 10:
                        return rating_value / 2
                        
            raise TravelSearchError(f"유효한 평점을 찾을 수 없습니다: {text}")
            
        except (ValueError, AttributeError) as e:
            logger.error(f"평점 추출 오류 '{text}': {e}")
            raise TravelSearchError(f"평점 추출 실패: {e}")
    
    @staticmethod
    def format_search_url(base_url: str, params: Dict[str, str]) -> str:
        """파라미터로 검색 URL 포맷팅 및 검증"""
        try:
            import urllib.parse
            
            if not base_url:
                raise TravelSearchError("기본 URL이 비어있습니다")
            
            query_params = []
            for key, value in params.items():
                if value is not None and str(value).strip():
                    encoded_value = urllib.parse.quote(str(value))
                    query_params.append(f"{key}={encoded_value}")
            
            if query_params:
                formatted_url = f"{base_url}?{'&'.join(query_params)}"
            else:
                formatted_url = base_url
            
            # 최종 URL 검증
            parsed = urllib.parse.urlparse(formatted_url)
            if not parsed.scheme or not parsed.netloc:
                raise TravelSearchError(f"잘못된 URL 형식: {formatted_url}")
            
            return formatted_url
            
        except Exception as e:
            logger.error(f"URL 포맷팅 오류: {e}")
            raise TravelSearchError(f"URL 포맷팅 실패: {e}")
    
    @staticmethod
    def parse_hotel_data(content: str, platform: str) -> List[Dict]:
        """페이지 콘텐츠에서 호텔 데이터 파싱 - 향상된 에러 처리"""
        try:
            if not content or not content.strip():
                raise TravelSearchError(f"플랫폼 {platform}의 콘텐츠가 비어있습니다")
            
            hotels = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                try:
                    if 'hotel' in line.lower() or any(keyword in line.lower() for keyword in ['resort', 'inn', 'lodge']):
                        if any(keyword in line.lower() for keyword in ['rating', 'price', 'score', 'star']):
                            hotel_data = {
                                'name': TravelSearchUtils._extract_hotel_name(line),
                                'price': TravelSearchUtils._extract_price_pattern(line),
                                'rating': TravelSearchUtils._extract_rating_pattern(line),
                                'platform': platform,
                                'raw_text': line.strip(),
                                'line_number': line_num + 1
                            }
                            
                            # 의미있는 데이터가 있는 경우만 추가
                            if hotel_data['name'] or hotel_data['price']:
                                hotels.append(hotel_data)
                                
                except Exception as line_error:
                    logger.warning(f"라인 {line_num + 1} 파싱 오류 {platform}: {line_error}")
                    continue
            
            # 품질 점수로 정렬하고 결과 제한
            hotels = TravelSearchUtils._rank_hotels(hotels)
            
            logger.info(f"{platform}에서 {len(hotels)}개 호텔 파싱 성공")
            return hotels[:15]  # 상위 15개 결과 반환
            
        except TravelSearchError:
            raise
        except Exception as e:
            logger.error(f"호텔 데이터 파싱 오류 {platform}: {e}")
            raise TravelSearchError(f"호텔 데이터 파싱 실패: {e}")
    
    @staticmethod
    def parse_flight_data(content: str, platform: str) -> List[Dict]:
        """페이지 콘텐츠에서 항공편 데이터 파싱 - 향상된 에러 처리"""
        try:
            if not content or not content.strip():
                raise TravelSearchError(f"플랫폼 {platform}의 콘텐츠가 비어있습니다")
            
            flights = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                try:
                    if any(keyword in line.lower() for keyword in ['flight', 'airline', 'departure', 'arrival']):
                        if any(indicator in line for indicator in ['$', '€', '£', '₩', 'price', 'fare', 'USD']):
                            flight_data = {
                                'airline': TravelSearchUtils._extract_airline_name(line),
                                'price': TravelSearchUtils._extract_price_pattern(line),
                                'duration': TravelSearchUtils._extract_duration(line),
                                'departure_time': TravelSearchUtils._extract_time_pattern(line),
                                'platform': platform,
                                'raw_text': line.strip(),
                                'line_number': line_num + 1
                            }
                            
                            if flight_data['airline'] or flight_data['price']:
                                flights.append(flight_data)
                                
                except Exception as line_error:
                    logger.warning(f"항공편 라인 {line_num + 1} 파싱 오류 {platform}: {line_error}")
                    continue
            
            flights = TravelSearchUtils._rank_flights(flights)
            
            logger.info(f"{platform}에서 {len(flights)}개 항공편 파싱 성공")
            return flights[:15]
            
        except TravelSearchError:
            raise
        except Exception as e:
            logger.error(f"항공편 데이터 파싱 오류 {platform}: {e}")
            raise TravelSearchError(f"항공편 데이터 파싱 실패: {e}")
    
    @staticmethod
    def _extract_hotel_name(text: str) -> str:
        """텍스트에서 호텔명 추출"""
        hotel_patterns = [
            r'Hotel\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Hotel(?:\s|$|,|\||\.)',
            r'Resort\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Resort(?:\s|$|,|\||\.)',
            r'Inn\s+([A-Za-z\s\-&]+?)(?:\s|$|,|\||\.)',
            r'([A-Za-z\s\-&]+?)\s+Inn(?:\s|$|,|\||\.)'
        ]
        
        for pattern in hotel_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 3 and name.replace(' ', '').isalpha():
                    return name
        
        return ""
    
    @staticmethod
    def _extract_price_pattern(text: str) -> str:
        """다양한 패턴으로 텍스트에서 가격 추출"""
        price_patterns = [
            r'[\$€£¥₩]\s*([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)\s*[\$€£¥₩]',
            r'([0-9,]+\.?[0-9]*)\s*per\s*night',
            r'([0-9,]+\.?[0-9]*)\s*원',
            r'Price:\s*[\$€£¥₩]?\s*([0-9,]+\.?[0-9]*)',
            r'from\s*[\$€£¥₩]\s*([0-9,]+\.?[0-9]*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    @staticmethod
    def _extract_rating_pattern(text: str) -> str:
        """다양한 패턴으로 텍스트에서 평점 추출"""
        rating_patterns = [
            r'([0-9.]+)\s*\/\s*([0-9]+)',
            r'([0-9.]+)\s*stars?',
            r'Rating:\s*([0-9.]+)',
            r'([0-9.]+)\s*out\s*of\s*([0-9]+)',
            r'Score:\s*([0-9.]+)',
            r'([0-9.]+)\s*\/\s*10'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    @staticmethod
    def _extract_airline_name(text: str) -> str:
        """텍스트에서 항공사명 추출"""
        airlines = [
            'Korean Air', 'Asiana', 'Delta', 'United', 'American', 'Lufthansa', 
            'Emirates', 'Singapore Airlines', 'Cathay Pacific', 'JAL', 'ANA', 
            'Air France', 'KLM', 'British Airways', 'Qatar Airways', 'Turkish Airlines',
            'Southwest', 'JetBlue', 'Alaska Airlines', 'Spirit', 'Frontier'
        ]
        
        for airline in airlines:
            if airline.lower() in text.lower():
                return airline
        
        # 항공사 같은 패턴 추출 시도
        airline_patterns = [
            r'([A-Z][a-z]+\s+Air(?:lines?)?)',
            r'([A-Z][a-z]+\s+Airways?)',
            r'(Air\s+[A-Z][a-z]+)'
        ]
        
        for pattern in airline_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def _extract_duration(text: str) -> str:
        """텍스트에서 항공편 소요시간 추출"""
        duration_patterns = [
            r'([0-9]+h\s*[0-9]*m?)',
            r'([0-9]+\s*hours?\s*[0-9]*\s*minutes?)',
            r'Duration:\s*([0-9]+h\s*[0-9]*m?)',
            r'([0-9]+:[0-9]+)\s*(?:hours?|hrs?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def _extract_time_pattern(text: str) -> str:
        """텍스트에서 시간 추출"""
        time_patterns = [
            r'([0-9]{1,2}:[0-9]{2}\s*[AP]M)',
            r'([0-9]{1,2}:[0-9]{2})',
            r'Departure:\s*([0-9]{1,2}:[0-9]{2})',
            r'Depart:\s*([0-9]{1,2}:[0-9]{2})'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    @staticmethod
    def _rank_hotels(hotels: List[Dict]) -> List[Dict]:
        """품질과 가격으로 호텔 순위 매기기"""
        try:
            for hotel in hotels:
                quality_score = 0
                
                # 평점 점수
                rating_text = hotel.get('rating', '')
                if rating_text:
                    try:
                        rating_value = TravelSearchUtils.extract_rating_from_text(rating_text)
                        if rating_value >= 4.0:
                            quality_score += 3
                        elif rating_value >= 3.5:
                            quality_score += 2
                        elif rating_value >= 3.0:
                            quality_score += 1
                    except TravelSearchError:
                        pass
                
                # 가격 점수 (낮을수록 좋음)
                price_text = hotel.get('price', '')
                if price_text:
                    try:
                        price_value = TravelSearchUtils.extract_price_from_text(price_text)
                        if price_value != float('inf'):
                            quality_score += 1
                    except TravelSearchError:
                        pass
                
                # 이름 품질 (긴 이름이 보통 더 자세함)
                name = hotel.get('name', '')
                if len(name) > 10:
                    quality_score += 1
                
                hotel['quality_score'] = quality_score
            
            # 품질 점수로 정렬 (내림차순)
            hotels.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            return hotels
            
        except Exception as e:
            logger.warning(f"호텔 순위 매기기 오류: {e}")
            return hotels
    
    @staticmethod
    def _rank_flights(flights: List[Dict]) -> List[Dict]:
        """품질과 가격으로 항공편 순위 매기기"""
        try:
            for flight in flights:
                quality_score = 0
                
                # 항공사 품질
                airline = flight.get('airline', '')
                major_airlines = ['Korean Air', 'Asiana', 'Delta', 'United', 'American', 
                                'Lufthansa', 'Emirates', 'Singapore Airlines']
                if any(major in airline for major in major_airlines):
                    quality_score += 2
                
                # 가격 가용성
                if flight.get('price'):
                    quality_score += 1
                
                # 소요시간 정보
                if flight.get('duration'):
                    quality_score += 1
                
                # 시간 정보
                if flight.get('departure_time'):
                    quality_score += 1
                
                flight['quality_score'] = quality_score
            
            flights.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            return flights
            
        except Exception as e:
            logger.warning(f"항공편 순위 매기기 오류: {e}")
            return flights
    
    @staticmethod
    def save_travel_report(content: str, filename: str, reports_dir: str = None) -> str:
        """여행 검색 보고서를 파일로 저장"""
        try:
            if reports_dir is None:
                reports_dir = config.get_logging_config().get('reports_dir', 'travel_scout_reports')
            
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
                "user_location": TravelSearchUtils.get_user_location(),
                "destination_options": TravelSearchUtils.load_destination_options(),
                "origin_options": TravelSearchUtils.load_origin_options(),
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
            raise TravelSearchError(f"여행 보고서 저장 실패: {str(e)}")
    
    @staticmethod
    def generate_travel_report_content(results: dict, search_params: dict) -> str:
        """여행 검색 보고서 내용 생성"""
        try:
            # 기본 정보 추출
            hotels = results.get('hotels', [])
            flights = results.get('flights', [])
            recommendations = results.get('recommendations', {})
            analysis = results.get('analysis', {})

            # 보고서 내용 생성
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
            return f"보고서 내용 생성 중 오류 발생: {e}"
