# AGRS (Adaptive Graph Reasoning System) 개발 계획

## 1. 개념 및 배경

- **AGRS란?**
    - 공유된 Neo4j 그래프 환경을 '공통 작업 공간'으로 삼아, 각 에이전트가 자신만의 규칙(패턴 매칭, 임계값 기준 등)에 따라 그래프 상태를 수정하며 협업하는 '분산 인지 시스템'.
    - 전통적 CoT/ToT가 텍스트 기반 추론에 머무른다면, AGRS는 **그래프 패턴 매칭**과 **네이티브 Cypher 연산**을 활용해 더욱 구조화된 사고 흐름을 지원.

- **핵심 개념 자료**
    1. **그래프 기반 지식 표현 및 추론**
       - 노드와 관계로 복잡한 도메인 지식(인과관계, 객체 속성 등)을 표현. 예: `(:Thought {id:1})-[:CONFIRMS]->(:Thought {id:2})`.
    2. **창발적 행동 및 자기조직화**
       - 단순 규칙을 따라 움직이는 에이전트들이 상호작용하며, 예상치 못한 문제 해결 경로가 자연스럽게 등장.
    3. **에이전트 기반 모델링 (ABM)**
       - 각 에이전트를 자율적 의사결정 주체로 정의, 시뮬레이션하듯 사고 그래프에서 협업.
    4. **분산 인공지능 (DAI)**
       - 중앙 집중 없이 그래프 상태 변화에 따라 비동기·병렬 실행, 시스템 확장성과 복원력을 확보.

- **참고 논문 및 프레임워크**
    - Besta et al. "Graph of Thoughts" ([arXiv:2308.09687](https://arxiv.org/pdf/2308.09687))
    - Chain-of-Thought / Tree-of-Thoughts → 한계점: 선형/트리 구조
    - LangGraph → 메시지 기반 워크플로우

## 2. AGRS 시스템 아키텍처

- **주요 에이전트**
    1. **Initiator Agent**
       - **트리거:** 사용자 질문 수신
       - **동작 예시:**
         ```cypher
         CREATE (t:Thought {id: apoc.create.uuid(), text: $prompt, status: 'new', is_initial: true});
         ```
       - **LLM 프롬프트 템플릿:**
         ```text
         "This is the initial thought for the prompt: {{prompt}}. Provide a concise summary node." 
         ```
    2. **Expander Agent**
       - **트리거:** `MATCH (t:Thought {status:'new'}) RETURN t` 패턴
       - **동작 예시:**
         ```cypher
         MATCH (t:Thought {status:'new'})
         CALL apoc.cypher.run(
           'CREATE (n:Thought {id: apoc.create.uuid(), text: $new_text, status:''unevaluated''})-[:DERIVES_FROM]->(t)',
           {new_text: $generated}
         );
         SET t.status = 'expanding';
         ```
       - **LLM 프롬프트 템플릿:**
         ```text
         "Given the thought: '{{t.text}}', generate two follow-up thoughts: one supporting (CONFIRMS) and one questioning (DOUBTS)."
         ```
    3. **Evaluator Agent**
       - **트리거:** `MATCH (t:Thought {status:'unevaluated'}) RETURN t`
       - **동작 예시:**
         ```cypher
         MATCH (t:Thought {status:'unevaluated'})
         SET t.validity_score = $score, t.status = 'evaluated';
         ```
       - **LLM 프롬프트 템플릿:**
         ```text
         "Evaluate the validity of the thought: '{{t.text}}'. Return a score between 0.0 and 1.0 with a short rationale."
         ```
    4. **Pruner Agent**
       - **트리거:** `MATCH (t:Thought) WHERE t.validity_score < 0.2 OR (t)-[:DERIVES_FROM*]->(t) RETURN t`
       - **동작 예시:**
         ```cypher
         MATCH (t:Thought {status:'evaluated'}) WHERE t.validity_score < 0.2
         SET t.status = 'pruned';
         ```
    5. **Synthesizer Agent**
       - **트리거:** `MATCH (t:Thought) WHERE t.status IN ['evaluated'] RETURN t`가 없을 때
       - **동작 예시:**
         ```cypher
         MATCH path = (root:Thought {is_initial:true})-[:DERIVES_FROM*]->(leaf)
         WITH path, reduce(sumScore = 0.0, n IN nodes(path) | sumScore + n.validity_score) AS total
         ORDER BY total DESC LIMIT 1
         RETURN nodes(path);
         ```
       - **LLM 프롬프트 템플릿:**
         ```text
         "Compile the final decision based on the sequence of thoughts: {{path_texts}}. Provide a coherent conclusion."
         ```

- **그래프 데이터 모델**
    - **Thought 노드**
        - `id: UUID`, `text: String`, `status: 'new'|'expanding'|'unevaluated'|'evaluated'|'pruned'`, `validity_score: Float`, `is_initial: Boolean`, `timestamp: DateTime`
    - **관계 유형**
        - `:DERIVES_FROM`, `:CONFIRMS`, `:DOUBTS`, `:PREDICTS_OUTCOME`, `:DECISION`

## 3. 개발 단계별 계획

### Phase 1: 기반 구축 및 환경 설정 (2일)
- 기술 스택 선정, 프로젝트 구조, Neo4j 데이터 모델 설계, 환경 세팅

### Phase 2: 핵심 에이전트 프로토타입 개발 (5일)
- 5개 전문 에이전트의 기본 기능 구현 (LLM 연동, Cypher 쿼리 생성 등)

### Phase 3: 그래프 상호작용 및 거버넌스 로직 구현 (4일)
- Graph Interface, Pattern Watcher & Dispatcher, 거버넌스 규칙 코드화

### Phase 4: 시스템 통합 및 워크플로우 오케스트레이션 (3일)
- 메인 컨트롤 루프, 로깅, 시각화 연동

### Phase 5: 테스트 및 고도화 (5일)
- 다양한 시나리오 테스트, 성능 분석, 프롬프트 튜닝

### Phase 6: 문서화 및 정리 (2일)
- README, 코드 주석, 예제 스크립트 작성

## 4. 기대 효과 및 활용 방안
- 복잡한 문제에 대한 창발적 해결, 논리적 사고의 시각화, 다양한 분야(전략, 분석, 창의적 문제해결 등) 적용 가능 