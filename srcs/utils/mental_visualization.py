"""
Mental Care Visualization Utilities
----------------------------------
심리 상태 시각화를 위한 유틸리티 모듈
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import os


# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 설정
EMOTION_COLORS = {
    '기쁨': '#FFD700',      # Gold
    '슬픔': '#4169E1',      # Royal Blue
    '분노': '#DC143C',      # Crimson
    '두려움': '#800080',    # Purple
    '불안': '#FF6347',      # Tomato
    '우울': '#2F4F4F',      # Dark Slate Gray
    '외로움': '#708090',    # Slate Gray
    '스트레스': '#FF4500',  # Orange Red
    '혼란': '#DDA0DD',      # Plum
    '절망': '#000000'       # Black
}


class MentalStateVisualizer:
    """심리 상태 시각화 클래스"""

    def __init__(self, output_dir: str = "mental_care_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def parse_conversation_data(self, conversation_history: List[Dict[str, str]]) -> pd.DataFrame:
        """대화 기록을 분석용 데이터프레임으로 변환"""
        data = []

        for i, entry in enumerate(conversation_history):
            if entry['speaker'] == 'user':
                # 간단한 감정 분석 (실제로는 더 정교한 분석이 필요)
                emotions = self._simple_emotion_analysis(entry['message'])
                timestamp = datetime.fromisoformat(entry['timestamp'])

                for emotion, intensity in emotions.items():
                    data.append({
                        'timestamp': timestamp,
                        'turn': i // 2,  # 대화 턴
                        'emotion': emotion,
                        'intensity': intensity,
                        'message': entry['message']
                    })

        return pd.DataFrame(data)

    def _simple_emotion_analysis(self, message: str) -> Dict[str, float]:
        """간단한 키워드 기반 감정 분석"""
        emotion_keywords = {
            '기쁨': ['기쁘', '행복', '좋', '웃', '즐거', '만족', '흥미'],
            '슬픔': ['슬프', '우울', '눈물', '힘들', '아프', '괴로'],
            '분노': ['화나', '짜증', '분노', '열받', '억울', '미워'],
            '두려움': ['무서', '걱정', '두려', '불안', '겁나'],
            '불안': ['불안', '초조', '걱정', '긴장', '떨려'],
            '우울': ['우울', '침울', '무기력', '공허', '절망'],
            '외로움': ['외로', '혼자', '쓸쓸', '고립'],
            '스트레스': ['스트레스', '압박', '부담', '피곤', '지쳐'],
            '혼란': ['혼란', '모르겠', '어떻게', '복잡', '갈등'],
            '절망': ['절망', '포기', '끝', '안돼', '못하겠']
        }

        emotions = {}
        message_lower = message.lower()

        for emotion, keywords in emotion_keywords.items():
            intensity = sum(1 for keyword in keywords if keyword in message_lower)
            emotions[emotion] = min(intensity * 0.5, 5.0)  # 0-5 스케일

        return emotions

    def create_emotion_timeline(self, df: pd.DataFrame, session_id: str) -> str:
        """감정 변화 시계열 그래프 생성"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty and emotion_data['intensity'].sum() > 0:
                ax.plot(emotion_data['turn'], emotion_data['intensity'],
                       marker='o', label=emotion, linewidth=2, markersize=6,
                       color=EMOTION_COLORS.get(emotion, '#808080'))

        ax.set_xlabel('대화 턴', fontsize=12)
        ax.set_ylabel('감정 강도', fontsize=12)
        ax.set_title('시간에 따른 감정 변화', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)

        plt.tight_layout()
        filename = f"{self.output_dir}/{session_id}_emotion_timeline.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_emotion_distribution(self, df: pd.DataFrame, session_id: str) -> str:
        """감정 분포 파이 차트 생성"""
        emotion_totals = df.groupby('emotion')['intensity'].sum()
        emotion_totals = emotion_totals[emotion_totals > 0]

        if emotion_totals.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in emotion_totals.index]
        wedges, texts, autotexts = ax.pie(emotion_totals.values,
                                         labels=emotion_totals.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)

        # 텍스트 스타일 개선
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('세션 전체 감정 분포', fontsize=16, fontweight='bold')

        plt.tight_layout()
        filename = f"{self.output_dir}/{session_id}_emotion_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_intensity_heatmap(self, df: pd.DataFrame, session_id: str) -> str:
        """감정 강도 히트맵 생성"""
        # 피벗 테이블 생성
        pivot_df = df.pivot_table(values='intensity',
                                 index='emotion',
                                 columns='turn',
                                 fill_value=0)

        if pivot_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd',
                   cbar_kws={'label': '감정 강도'}, ax=ax)

        ax.set_title('대화 턴별 감정 강도 히트맵', fontsize=16, fontweight='bold')
        ax.set_xlabel('대화 턴', fontsize=12)
        ax.set_ylabel('감정', fontsize=12)

        plt.tight_layout()
        filename = f"{self.output_dir}/{session_id}_intensity_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_dominant_emotion_chart(self, df: pd.DataFrame, session_id: str) -> str:
        """각 대화 턴별 주요 감정 막대 차트"""
        turn_emotions = df.groupby(['turn', 'emotion'])['intensity'].sum().unstack(fill_value=0)
        dominant_emotions = turn_emotions.idxmax(axis=1)
        dominant_intensities = turn_emotions.max(axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in dominant_emotions]
        bars = ax.bar(range(len(dominant_emotions)), dominant_intensities, color=colors)

        # 막대 위에 감정 라벨 추가
        for i, (bar, emotion) in enumerate(zip(bars, dominant_emotions)):
            if dominant_intensities.iloc[i] > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       emotion, ha='center', va='bottom', fontsize=10, rotation=45)

        ax.set_xlabel('대화 턴', fontsize=12)
        ax.set_ylabel('감정 강도', fontsize=12)
        ax.set_title('각 대화 턴의 주요 감정', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(dominant_emotions)))
        ax.set_xticklabels([f'턴 {i+1}' for i in range(len(dominant_emotions))], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"{self.output_dir}/{session_id}_dominant_emotions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_emotion_correlation_matrix(self, df: pd.DataFrame, session_id: str) -> str:
        """감정 간 상관관계 매트릭스"""
        correlation_df = df.pivot_table(values='intensity',
                                       index='turn',
                                       columns='emotion',
                                       fill_value=0)

        if correlation_df.shape[1] < 2:
            return None

        correlation_matrix = correlation_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, ax=ax)

        ax.set_title('감정 간 상관관계', fontsize=16, fontweight='bold')

        plt.tight_layout()
        filename = f"{self.output_dir}/{session_id}_emotion_correlation.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_session_summary_dashboard(self, df: pd.DataFrame, session_id: str,
                                       session_info: Dict[str, Any]) -> str:
        """세션 요약 대시보드"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 감정 분포 (파이 차트)
        emotion_totals = df.groupby('emotion')['intensity'].sum()
        emotion_totals = emotion_totals[emotion_totals > 0]

        if not emotion_totals.empty:
            colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in emotion_totals.index]
            ax1.pie(emotion_totals.values, labels=emotion_totals.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('감정 분포', fontweight='bold')

        # 2. 시간별 감정 변화
        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty and emotion_data['intensity'].sum() > 0:
                ax2.plot(emotion_data['turn'], emotion_data['intensity'],
                        marker='o', label=emotion, linewidth=2,
                        color=EMOTION_COLORS.get(emotion, '#808080'))

        ax2.set_xlabel('대화 턴')
        ax2.set_ylabel('감정 강도')
        ax2.set_title('감정 변화 추이', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 주요 감정 막대 차트
        top_emotions = emotion_totals.nlargest(5)
        colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in top_emotions.index]
        ax3.bar(range(len(top_emotions)), top_emotions.values, color=colors)
        ax3.set_xticks(range(len(top_emotions)))
        ax3.set_xticklabels(top_emotions.index, rotation=45)
        ax3.set_ylabel('총 감정 강도')
        ax3.set_title('주요 감정 순위', fontweight='bold')

        # 4. 세션 정보 텍스트
        ax4.axis('off')
        session_text = f"""
세션 정보

세션 ID: {session_id}
시작 시간: {session_info.get('start_time', 'N/A')}
총 대화 턴: {len(df['turn'].unique())}
총 감정 발현: {len(df)}
평균 감정 강도: {df['intensity'].mean():.2f}

주요 감정:
{chr(10).join([f"• {emotion}: {intensity:.1f}" for emotion, intensity in top_emotions.head(3).items()])}
        """
        ax4.text(0.1, 0.9, session_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.suptitle(f'심리 상담 세션 요약 대시보드', fontsize=18, fontweight='bold')
        plt.tight_layout()

        filename = f"{self.output_dir}/{session_id}_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def generate_all_visualizations(self, conversation_history: List[Dict[str, str]],
                                  session_id: str, session_info: Dict[str, Any]) -> List[str]:
        """모든 시각화 생성"""
        df = self.parse_conversation_data(conversation_history)

        if df.empty:
            return []

        generated_files = []

        # 각 시각화 생성
        timeline_file = self.create_emotion_timeline(df, session_id)
        if timeline_file:
            generated_files.append(timeline_file)

        distribution_file = self.create_emotion_distribution(df, session_id)
        if distribution_file:
            generated_files.append(distribution_file)

        heatmap_file = self.create_intensity_heatmap(df, session_id)
        if heatmap_file:
            generated_files.append(heatmap_file)

        dominant_file = self.create_dominant_emotion_chart(df, session_id)
        if dominant_file:
            generated_files.append(dominant_file)

        correlation_file = self.create_emotion_correlation_matrix(df, session_id)
        if correlation_file:
            generated_files.append(correlation_file)

        dashboard_file = self.create_session_summary_dashboard(df, session_id, session_info)
        if dashboard_file:
            generated_files.append(dashboard_file)

        return generated_files


def create_sample_visualization(session_id: str = "sample_session"):
    """샘플 시각화 생성 (테스트용)"""
    # 샘플 대화 데이터
    sample_conversation = [
        {"timestamp": "2024-01-15T10:00:00", "speaker": "user", "message": "오늘 정말 힘든 하루였어요. 일이 너무 많아서 스트레스받고 있어요."},
        {"timestamp": "2024-01-15T10:01:00", "speaker": "counselor", "message": "힘드셨겠어요. 어떤 일들이 가장 스트레스를 주고 있나요?"},
        {"timestamp": "2024-01-15T10:02:00", "speaker": "user", "message": "상사가 계속 무리한 요구를 해서 화가 나고, 동시에 불안하기도 해요."},
        {"timestamp": "2024-01-15T10:03:00", "speaker": "counselor", "message": "분노와 불안이 동시에 느껴지는군요. 그런 감정들을 어떻게 다루고 계신가요?"},
        {"timestamp": "2024-01-15T10:04:00", "speaker": "user", "message": "솔직히 잘 모르겠어요. 그냥 혼란스럽고 때로는 절망적이기도 해요."},
    ]

    session_info = {
        "start_time": "2024-01-15T10:00:00",
        "session_duration": "30분"
    }

    visualizer = MentalStateVisualizer()
    generated_files = visualizer.generate_all_visualizations(sample_conversation, session_id, session_info)

    print(f"샘플 시각화 파일들이 생성되었습니다:")
    for file in generated_files:
        print(f"  - {file}")

    return generated_files


if __name__ == "__main__":
    create_sample_visualization()
