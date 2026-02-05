# Chrome Built-in AI (Gemini Nano) 튜토리얼

Chrome [기본 제공 AI API](https://developer.chrome.com/docs/ai/built-in-apis?hl=ko)(Gemini Nano)를 로컬 브라우저에서 사용해 번역·요약·프롬프트 등 여러 작업을 수행하는 튜토리얼입니다.

## Cron으로 돌릴 수 있나?

**요약: 전통적인 서버 cron(백그라운드만)에서는 직접 불가에 가깝고, “브라우저를 띄워서” 우회하면 제한적으로 가능합니다.**

- **Chrome 내장 AI는 브라우저 전용**  
  Prompt API, Summarizer, Translator, Language Detector 등은 모두 **Chrome 브라우저(또는 Chrome 확장 프로그램) 안에서만** 동작하는 JavaScript API입니다.  
  서버에서 돌아가는 cron 프로세스에는 브라우저가 없으므로, **cron만으로는 이 API들을 직접 호출할 수 없습니다.**

- **가능한 우회 방식**
  1. **cron → Chrome 실행 스크립트**  
     cron이 “Chrome을 띄우는 스크립트”를 실행하고, 그 스크립트가 [Playwright](https://playwright.dev/) 등으로 **실제 Chrome**을 띄운 뒤, Gemini Nano를 쓰는 페이지를 열어 작업을 수행하고 결과를 파일/stdout으로 저장하는 방식입니다.  
     - 이 경우에도 **Chrome 프로필에 Gemini Nano가 다운로드·활성화**되어 있어야 하고,  
     - **사용자 활성화(user gesture)** 요구나 headless 미지원 등으로 **환경에 따라 동작하지 않을 수 있습니다.**  
     - 실험용 스크립트는 `scripts/run_demo_via_browser.py`에 있으며, “가능하면 시도해 보는 수준”으로 사용하세요.
  2. **브라우저 안에서 주기 실행**  
     cron 대신 **Chrome 확장 프로그램**의 [chrome.alarms](https://developer.chrome.com/docs/extensions/reference/alarms/) 또는 웹 페이지의 `setInterval` 등으로 “주기적으로 로컬 브라우저에서 AI 작업”을 실행하는 구성이면, Gemini Nano를 그대로 사용할 수 있습니다.

이 튜토리얼은 **로컬 브라우저에서** 위 API들을 쓰는 방법을 다루며, cron 연동은 위와 같은 제한이 있음을 전제로 합니다.

---

## 사전 요구사항

- **Chrome 138+** (또는 Canary + 플래그)
- **하드웨어**: [공식 요구사항](https://developer.chrome.com/docs/ai/get-started?hl=ko#hardware) 참고  
  - 저장 공간 22GB 이상, RAM 16GB 이상 또는 GPU VRAM 4GB 이상 등
- **localhost 사용 시** 다음 플래그 활성화:
  - `chrome://flags/#optimization-guide-on-device-model` → **사용 설정됨**
  - `chrome://flags/#prompt-api-for-gemini-nano` → **Enabled** (또는 Enabled multilingual)
- **웹에서 사용 시**: [오리진 트라이얼](https://developer.chrome.com/docs/web-platform/origin-trials?hl=ko) 또는 [Chrome 확장 프로그램](https://developer.chrome.com/docs/ai/prompt-api?hl=ko) (Prompt API는 확장에서 Chrome 138부터 사용 가능)

참고: [기본 제공 AI 시작하기](https://developer.chrome.com/docs/ai/get-started?hl=ko), [API 상태 및 개요](https://developer.chrome.com/docs/ai/built-in-apis?hl=ko).

---

## 튜토리얼 구성

| 항목 | 설명 |
|------|------|
| **demo/** | 로컬에서 열어볼 수 있는 웹 데모 (Prompt, 요약, 번역, 언어 감지) |
| **runner.py** | 다른 agent에서 import해 호출하는 러너. Playwright로 데모 페이지를 열고 작업 실행 후 결과 반환. |
| **scripts/run_demo_via_browser.py** | (선택) Playwright로 Chrome을 띄워 데모 페이지를 여는 실험용 스크립트. |

---

## 로컬 브라우저에서 데모 실행

1. Chrome에서 위 플래그를 설정한 뒤 재시작합니다.
2. 데모 디렉터리에서 로컬 HTTP 서버를 띄웁니다 (localhost 필요):

   ```bash
   cd cron_agents/chrome_builtin_ai_tutorial/demo
   python -m http.server 8080
   ```

3. 브라우저에서 `http://localhost:8080` 을 엽니다.
4. 페이지에서 **한 번 클릭 등 사용자 동작** 후, 각 섹션(Prompt, 요약, 번역, 언어 감지)을 사용해 봅니다.  
   (일부 API는 사용자 활성화 후에만 `create()` 등이 동작합니다.)

API가 없는 환경에서는 각 섹션에 “이 브라우저에서는 지원되지 않습니다” 같은 안내가 나오도록 되어 있습니다.

---

## (선택) Playwright로 데모 자동 열기

데모 디렉터리에서 HTTP 서버를 대신 띄우고 Chrome을 자동으로 열려면:

```bash
cd cron_agents/chrome_builtin_ai_tutorial
pip install -r requirements.txt
playwright install chrome   # 또는 chromium
python scripts/run_demo_via_browser.py
```

이미 다른 터미널에서 `python -m http.server 8765` 를 `demo/` 디렉터리에서 실행 중이면 `--no-serve` 옵션을 사용하세요.  
이 스크립트는 실험용이며, Built-in AI가 자동화 환경에서 동작하지 않을 수 있습니다.

## 다른 agent에서 cron으로 호출하기 (권장 접근)

**cron → (다른 agent) → Playwright → Chrome Built-in AI** 흐름으로 쓰는 방식입니다.  
cron이 “다른 agent”를 부르고, 그 agent가 Playwright를 사용해 Built-in AI 데모 페이지를 열고, 입력을 넣고 결과를 가져옵니다.

1. **다른 agent** (예: `cron_agents/my_scheduler/main.py`)에서 이 튜토리얼의 **runner** 모듈을 import합니다.
2. **runner**가 내부적으로 Playwright로 Chrome을 띄우고, 데모 페이지에 접속한 뒤, 지정한 작업(prompt / summarize / translate / detect_language)을 실행하고 결과를 반환합니다.
3. 해당 agent는 반환된 결과를 로그·DB·다음 단계 등에 활용합니다.

### 사용 예 (다른 agent 코드에서)

```python
from pathlib import Path
import sys

# cron_agents 기준으로 경로 추가 (프로젝트 루트가 아닐 수 있음)
_CRON_AGENTS = Path(__file__).resolve().parent.parent  # 예: cron_agents
sys.path.insert(0, str(_CRON_AGENTS))

from chrome_builtin_ai_tutorial.runner import run_builtin_ai_task_sync

# 동기 호출 (asyncio 없이)
result = run_builtin_ai_task_sync(
    "prompt",
    {"text": "오늘 점심 메뉴 한 가지 추천해줘"},
    timeout_seconds=60.0,
)
if result["success"]:
    print(result["result"])
else:
    print("실패:", result["error"])
```

### 지원 작업 (task_type / input_data)

| task_type | input_data | 비고 |
|-----------|------------|------|
| `prompt` | `{"text": "질문"}` | Prompt API |
| `summarize` | `{"text": "요약할 긴 글"}` | Summarizer API |
| `translate` | `{"text": "번역할 문장", "target_language": "en"}` | Translator API |
| `detect_language` | `{"text": "언어 감지할 문장"}` | Language Detector API |

`run_builtin_ai_task_sync()`는 내부에서 HTTP 서버를 띄우고(옵션), Playwright로 Chrome을 실행한 뒤 위 작업을 수행합니다.  
환경(Chrome·Gemini Nano·사용자 활성화)에 따라 실패할 수 있으므로, **호출하는 agent 쪽에서 실패 시 fallback(다른 LLM, 스킵 등)을 두는 것을 권장**합니다.

### asyncio 사용 시

```python
from chrome_builtin_ai_tutorial.runner import run_builtin_ai_task
import asyncio

async def my_agent_step():
    out = await run_builtin_ai_task("summarize", {"text": "긴 글..."})
    return out
```

---

## Cron과 연동하려면 (실험적)

- **방법 1 (권장)**  
  cron이 **다른 agent**를 실행하고, 그 agent가 위 **runner**를 사용해 Playwright로 Built-in AI 작업을 수행합니다.  
  예: `0 9 * * * cd /path/to/cron_agents/my_agent && python main.py`

- **방법 2**  
  cron에서 직접 `scripts/run_demo_via_browser.py` 를 호출해 데모 페이지만 열 수도 있습니다.  
  환경에 따라 API가 동작하지 않을 수 있으므로 “데모 열기” 수준으로만 쓰는 것을 권장합니다.

- **방법 3**  
  주기 실행을 **cron이 아닌** Chrome 확장 프로그램의 `chrome.alarms` 또는 브라우저를 켜 둔 상태의 페이지 타이머로 하는 구성도 가능합니다.

---

## 참고 링크

- [기본 제공 AI API (Built-in APIs)](https://developer.chrome.com/docs/ai/built-in-apis?hl=ko)
- [Prompt API](https://developer.chrome.com/docs/ai/prompt-api?hl=ko)
- [Summarizer API](https://developer.chrome.com/docs/ai/summarizer-api?hl=ko)
- [Translator API](https://developer.chrome.com/docs/ai/translator-api?hl=ko)
- [Language Detector API](https://developer.chrome.com/docs/ai/language-detection?hl=ko)
- [기본 제공 AI 시작하기](https://developer.chrome.com/docs/ai/get-started?hl=ko)
