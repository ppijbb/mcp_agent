"""
Common Styles Module

공통으로 사용되는 CSS 스타일들을 모아둔 모듈
"""

# 공통 헤더 스타일
HEADER_STYLES = {
    "main_style_block": """
    <style>
        .page-header {
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
    </style>
    """,
    "main": "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);",
    "product": "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);",
    "business": "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);",
    "seo": "background: linear-gradient(45deg, #ff4757, #ff3838);",
    "finance": "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);",
    "cyber": "background: linear-gradient(90deg, #ff4757, #ff3838);",
    "data": "background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);"
}

# 공통 버튼 스타일
BUTTON_STYLES = """
<style>
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* 홈 버튼 특별 스타일 */
    .home-button > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
    }

    .home-button > button:hover {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%) !important;
    }

    /* 위험 버튼 스타일 */
    .danger-button > button {
        background: linear-gradient(135deg, #ff4757, #ff3838) !important;
    }

    .danger-button > button:hover {
        background: linear-gradient(135deg, #ff3838, #ff2f2f) !important;
    }
</style>
"""

# 공통 카드 스타일
CARD_STYLES = """
<style>
    .agent-card {
        background: var(--background-color);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s;
        border: 1px solid var(--secondary-background-color);
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .agent-card-content {
        flex-grow: 1;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        border: 1px solid var(--secondary-background-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .category-header {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }

    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--secondary-background-color);
    }
</style>
"""

# 다크모드 대응 스타일
DARK_MODE_STYLES = """
<style>
    /* 다크모드에서 텍스트 색상 보정 */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
    }

    .stats-container {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--secondary-background-color);
    }
</style>
"""

# 모바일 최적화 스타일
MOBILE_STYLES = """
<style>
    /* 모바일 우선 설계 */
    .main > div {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* 큰 버튼 스타일 */
    .mobile-button > button {
        height: 3rem;
        width: 100%;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        margin: 0.5rem 0;
    }

    /* 모바일 텍스트 크기 */
    .metric-big {
        font-size: 2rem !important;
        font-weight: bold !important;
    }

    /* 터치 친화적 스페이싱 */
    .touch-friendly {
        min-height: 44px;
        padding: 12px;
        margin: 8px 0;
    }

    /* 진동 애니메이션 */
    @keyframes vibrate {
        0% { transform: translateX(0); }
        25% { transform: translateX(-2px); }
        50% { transform: translateX(2px); }
        75% { transform: translateX(-2px); }
        100% { transform: translateX(0); }
    }

    .vibrate {
        animation: vibrate 0.3s ease-in-out;
    }

    /* 로딩 스피너 */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
</style>
"""


def get_common_styles():
    """
    Combine and return all common styles for the application.

    Returns:
        Combined CSS string containing header, button, card, and dark mode styles.
    """
    return HEADER_STYLES["main_style_block"] + BUTTON_STYLES + CARD_STYLES + DARK_MODE_STYLES


def get_mobile_styles():
    """
    Return mobile-optimized styles.

    Returns:
        CSS string for mobile-responsive design.
    """
    return MOBILE_STYLES


def apply_custom_styles():
    """
    Apply common styles to Streamlit application.

    Uses Streamlit's markdown to inject CSS styles with unsafe_allow_html=True.
    """
    import streamlit as st
    st.markdown(get_common_styles(), unsafe_allow_html=True)


def get_page_header(page_type, title, subtitle):
    """
    Generate page header HTML with custom styling.

    Args:
        page_type: Type of page for styling (e.g., 'main', 'agent', etc.)
        title: Main title to display in the header.
        subtitle: Subtitle text to display below the title.

    Returns:
        HTML string containing the styled page header.
    """
    background = HEADER_STYLES.get(page_type, HEADER_STYLES["main"])

    return f"""
    <div class="page-header" style="{background}">
        <h1>{title}</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            {subtitle}
        </p>
    </div>
    """


def get_home_button():
    """
    Generate home button HTML for navigation.

    Returns:
        HTML string containing the styled home button.
    """
    return """
    <div class="home-button">
        🏠 홈으로 돌아가기
    </div>
    """
