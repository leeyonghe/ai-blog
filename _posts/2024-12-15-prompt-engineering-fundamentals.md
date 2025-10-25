---
layout: post
title: "프롬프트 엔지니어링 기초와 인컨텍스트 학습 완벽 가이드"
date: 2024-12-15 14:30:00 +0900
categories: [AI, Prompt Engineering, In-Context Learning]
tags: [prompt-engineering, in-context-learning, zero-shot, few-shot, system-prompt, context-efficiency]
---

## 개요

**프롬프트 엔지니어링**은 대규모 언어 모델(LLM)과 효과적으로 소통하는 핵심 기술입니다. 이번 포스트에서는 프롬프트의 기본 개념부터 인컨텍스트 학습, 시스템/사용자 프롬프트 구분, 그리고 컨텍스트 효율성까지 체계적으로 살펴보겠습니다.

## 1. 프롬프트 소개

### 1.1 프롬프트의 정의와 중요성

<div class="mermaid">
graph TB
    subgraph "Prompt Engineering Ecosystem"
        A[Prompt Engineering] --> B[Core Techniques]
        A --> C[Learning Paradigms]
        A --> D[Optimization Strategies]
        
        B --> B1[Prompt Structure]
        B --> B2[Context Management]
        B --> B3[Output Formatting]
        
        C --> C1[Zero-shot Learning]
        C --> C2[Few-shot Learning]
        C --> C3[Chain of Thought]
        C --> C4[In-Context Learning]
        
        D --> D1[Response Quality]
        D --> D2[Cost Optimization]
        D --> D3[Latency Reduction]
        D --> D4[Token Efficiency]
        
        subgraph "Prompt Components"
            E[System Prompt]
            F[User Prompt]
            G[Examples]
            H[Instructions]
        end
        
        B1 --> E
        B1 --> F
        C2 --> G
        B3 --> H
        
        subgraph "Application Domains"
            I[Text Generation]
            J[Code Generation]
            K[Analysis & Reasoning]
            L[Creative Tasks]
        end
        
        A --> I
        A --> J
        A --> K
        A --> L
    end
    
    style A fill:#ff9999
    style C fill:#66b3ff
    style D fill:#99ff99
    style B fill:#ffcc99
</div>

### 1.2 프롬프트의 기본 구조

```python
class PromptStructure:
    """프롬프트 구조 설계 클래스"""
    
    def __init__(self):
        self.components = {
            "system_context": "",
            "task_instruction": "",
            "input_data": "",
            "examples": [],
            "output_format": "",
            "constraints": []
        }
    
    def build_prompt(self, task_type="general"):
        """작업 유형에 따른 프롬프트 구성"""
        
        prompt_templates = {
            "classification": self._build_classification_prompt,
            "generation": self._build_generation_prompt,
            "analysis": self._build_analysis_prompt,
            "extraction": self._build_extraction_prompt,
            "reasoning": self._build_reasoning_prompt
        }
        
        builder = prompt_templates.get(task_type, self._build_general_prompt)
        return builder()
    
    def _build_classification_prompt(self):
        """분류 작업용 프롬프트"""
        return f"""
{self.components['system_context']}

작업: 주어진 텍스트를 다음 카테고리 중 하나로 분류하세요.

분류 대상: {self.components['input_data']}

가능한 카테고리:
{self._format_categories()}

{self._format_examples()}

응답 형식:
{self.components['output_format']}

제약 조건:
{self._format_constraints()}
"""
    
    def _build_generation_prompt(self):
        """생성 작업용 프롬프트"""
        return f"""
{self.components['system_context']}

작업: {self.components['task_instruction']}

입력 정보:
{self.components['input_data']}

{self._format_examples()}

생성 요구사항:
- 톤: 전문적이고 명확한
- 길이: 200-500 단어
- 구조: 도입부, 본문, 결론

출력 형식:
{self.components['output_format']}
"""
    
    def add_few_shot_examples(self, examples):
        """퓨샷 예제 추가"""
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            formatted_examples.append(f"""
예제 {i}:
입력: {example['input']}
출력: {example['output']}
""")
        
        self.components['examples'] = formatted_examples
    
    def optimize_for_context_length(self, max_tokens=4000):
        """컨텍스트 길이 최적화"""
        current_length = self._estimate_token_count()
        
        if current_length > max_tokens:
            # 예제 수 줄이기
            if len(self.components['examples']) > 3:
                self.components['examples'] = self.components['examples'][:3]
            
            # 긴 설명 요약
            if len(self.components['system_context']) > 500:
                self.components['system_context'] = self._summarize_context()
        
        return self._estimate_token_count()
    
    def _estimate_token_count(self):
        """토큰 수 추정 (대략적)"""
        full_prompt = self.build_prompt()
        # 대략 4 characters = 1 token
        return len(full_prompt) // 4

class PromptValidator:
    """프롬프트 유효성 검증기"""
    
    def __init__(self):
        self.validation_rules = {
            "clarity": self._check_clarity,
            "specificity": self._check_specificity,
            "completeness": self._check_completeness,
            "consistency": self._check_consistency
        }
    
    def validate_prompt(self, prompt):
        """프롬프트 종합 검증"""
        validation_results = {}
        
        for rule_name, rule_func in self.validation_rules.items():
            validation_results[rule_name] = rule_func(prompt)
        
        # 전체 점수 계산
        overall_score = sum(result['score'] for result in validation_results.values()) / len(validation_results)
        
        return {
            "overall_score": overall_score,
            "detailed_results": validation_results,
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def _check_clarity(self, prompt):
        """명확성 검사"""
        clarity_indicators = {
            "has_clear_instruction": "수행해야 할" in prompt or "작업:" in prompt,
            "uses_simple_language": self._check_language_complexity(prompt),
            "avoids_ambiguity": self._check_ambiguous_terms(prompt),
            "has_examples": "예제" in prompt or "예시" in prompt
        }
        
        score = sum(clarity_indicators.values()) / len(clarity_indicators)
        
        return {
            "score": score,
            "indicators": clarity_indicators,
            "suggestions": self._generate_clarity_suggestions(clarity_indicators)
        }
    
    def _check_specificity(self, prompt):
        """구체성 검사"""
        specificity_indicators = {
            "has_output_format": "형식" in prompt or "포맷" in prompt,
            "defines_constraints": "제약" in prompt or "조건" in prompt,
            "specifies_length": any(word in prompt for word in ["글자", "단어", "문장", "단락"]),
            "provides_context": len(prompt.split('\n')) > 3
        }
        
        score = sum(specificity_indicators.values()) / len(specificity_indicators)
        
        return {
            "score": score,
            "indicators": specificity_indicators
        }
```

## 2. 인컨텍스트 학습: 제로샷과 퓨샷

### 2.1 제로샷 학습 (Zero-shot Learning)

```python
class ZeroShotPrompting:
    """제로샷 프롬프팅 클래스"""
    
    def __init__(self, model):
        self.model = model
        self.zero_shot_templates = self._load_templates()
    
    def _load_templates(self):
        """제로샷 템플릿 로드"""
        return {
            "classification": """
다음 텍스트를 {categories} 중 하나로 분류해주세요.

텍스트: "{text}"

분류 결과: """,
            
            "sentiment_analysis": """
다음 텍스트의 감정을 분석해주세요.

텍스트: "{text}"

감정 (긍정/부정/중립): """,
            
            "summarization": """
다음 텍스트를 3문장으로 요약해주세요.

텍스트: "{text}"

요약: """,
            
            "question_answering": """
주어진 문맥을 바탕으로 질문에 답해주세요.

문맥: "{context}"
질문: "{question}"

답변: """,
            
            "translation": """
다음 텍스트를 {target_language}로 번역해주세요.

원문: "{text}"

번역: """
        }
    
    def classify_text(self, text, categories):
        """텍스트 분류 (제로샷)"""
        prompt = self.zero_shot_templates["classification"].format(
            text=text,
            categories=", ".join(categories)
        )
        
        response = self.model.generate(prompt, temperature=0.1)
        return self._parse_classification_response(response, categories)
    
    def analyze_sentiment(self, text):
        """감정 분석 (제로샷)"""
        prompt = self.zero_shot_templates["sentiment_analysis"].format(text=text)
        
        response = self.model.generate(prompt, temperature=0.1)
        return self._parse_sentiment_response(response)
    
    def zero_shot_reasoning(self, problem):
        """제로샷 추론"""
        reasoning_prompt = f"""
다음 문제를 단계별로 해결해주세요.

문제: {problem}

해결 과정:
1. 문제 이해:
2. 접근 방법:
3. 단계별 해결:
4. 최종 답안:
"""
        
        response = self.model.generate(reasoning_prompt, temperature=0.3)
        return self._parse_reasoning_response(response)
    
    def chain_of_thought_zero_shot(self, problem):
        """제로샷 사고 연쇄"""
        cot_prompt = f"""
{problem}

단계별로 생각해보겠습니다:
"""
        
        response = self.model.generate(cot_prompt, temperature=0.3)
        return response
    
    def _parse_classification_response(self, response, categories):
        """분류 응답 파싱"""
        response_lower = response.lower().strip()
        
        for category in categories:
            if category.lower() in response_lower:
                return {
                    "predicted_category": category,
                    "confidence": self._estimate_confidence(response),
                    "raw_response": response
                }
        
        return {
            "predicted_category": "Unknown",
            "confidence": 0.0,
            "raw_response": response
        }

class FewShotPrompting:
    """퓨샷 프롬프팅 클래스"""
    
    def __init__(self, model):
        self.model = model
        self.example_database = {}
    
    def add_examples(self, task_type, examples):
        """예제 데이터베이스에 추가"""
        if task_type not in self.example_database:
            self.example_database[task_type] = []
        
        self.example_database[task_type].extend(examples)
    
    def few_shot_classify(self, text, task_type, num_examples=3):
        """퓨샷 분류"""
        examples = self._select_best_examples(task_type, text, num_examples)
        
        prompt = self._build_few_shot_prompt(
            task_type="classification",
            examples=examples,
            input_text=text
        )
        
        response = self.model.generate(prompt, temperature=0.1)
        return response
    
    def _build_few_shot_prompt(self, task_type, examples, input_text):
        """퓨샷 프롬프트 구성"""
        prompt_parts = [
            "다음은 텍스트 분류 작업의 예제들입니다:\n"
        ]
        
        # 예제 추가
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"""
예제 {i}:
입력: {example['input']}
출력: {example['output']}
""")
        
        # 실제 작업
        prompt_parts.append(f"""
이제 다음 텍스트를 분류해주세요:
입력: {input_text}
출력: """)
        
        return "\n".join(prompt_parts)
    
    def _select_best_examples(self, task_type, input_text, num_examples):
        """가장 적합한 예제 선택"""
        available_examples = self.example_database.get(task_type, [])
        
        if len(available_examples) <= num_examples:
            return available_examples
        
        # 유사도 기반 예제 선택
        similarities = []
        for example in available_examples:
            similarity = self._calculate_similarity(input_text, example['input'])
            similarities.append((similarity, example))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [example for _, example in similarities[:num_examples]]
    
    def _calculate_similarity(self, text1, text2):
        """텍스트 유사도 계산 (간단한 자카드 유사도)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def dynamic_few_shot(self, input_text, task_type, max_examples=5):
        """동적 퓨샷 학습"""
        # 초기에는 적은 예제로 시작
        for num_examples in range(1, max_examples + 1):
            examples = self._select_best_examples(task_type, input_text, num_examples)
            
            prompt = self._build_few_shot_prompt(task_type, examples, input_text)
            response = self.model.generate(prompt, temperature=0.1)
            
            # 응답 품질 평가
            confidence = self._evaluate_response_confidence(response)
            
            if confidence > 0.8:  # 충분히 확신하는 경우
                return {
                    "response": response,
                    "num_examples_used": num_examples,
                    "confidence": confidence
                }
        
        # 최대 예제 수를 사용한 결과 반환
        return {
            "response": response,
            "num_examples_used": max_examples,
            "confidence": confidence
        }

class InContextLearningOptimizer:
    """인컨텍스트 학습 최적화기"""
    
    def __init__(self, model):
        self.model = model
        self.performance_cache = {}
    
    def optimize_example_selection(self, task_data, validation_data):
        """예제 선택 최적화"""
        optimization_results = {}
        
        strategies = {
            "random": self._random_selection,
            "similarity": self._similarity_based_selection,
            "diversity": self._diversity_based_selection,
            "difficulty": self._difficulty_based_selection,
            "performance": self._performance_based_selection
        }
        
        for strategy_name, strategy_func in strategies.items():
            selected_examples = strategy_func(task_data, num_examples=5)
            
            # 검증 데이터로 성능 평가
            performance = self._evaluate_examples(selected_examples, validation_data)
            
            optimization_results[strategy_name] = {
                "examples": selected_examples,
                "performance": performance
            }
        
        # 최적 전략 선택
        best_strategy = max(optimization_results.items(), 
                          key=lambda x: x[1]['performance']['accuracy'])
        
        return {
            "best_strategy": best_strategy[0],
            "best_examples": best_strategy[1]['examples'],
            "all_results": optimization_results
        }
    
    def _diversity_based_selection(self, task_data, num_examples):
        """다양성 기반 예제 선택"""
        selected = []
        remaining = task_data.copy()
        
        # 첫 번째 예제는 랜덤 선택
        first_example = random.choice(remaining)
        selected.append(first_example)
        remaining.remove(first_example)
        
        # 나머지 예제들은 다양성을 고려하여 선택
        for _ in range(num_examples - 1):
            if not remaining:
                break
            
            max_diversity = -1
            best_candidate = None
            
            for candidate in remaining:
                # 이미 선택된 예제들과의 다양성 계산
                diversity_score = self._calculate_diversity(candidate, selected)
                
                if diversity_score > max_diversity:
                    max_diversity = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _calculate_diversity(self, candidate, selected_examples):
        """예제 다양성 계산"""
        if not selected_examples:
            return 1.0
        
        similarities = []
        for selected in selected_examples:
            similarity = self._calculate_similarity(
                candidate['input'], 
                selected['input']
            )
            similarities.append(similarity)
        
        # 평균 유사도가 낮을수록 다양성이 높음
        avg_similarity = sum(similarities) / len(similarities)
        return 1 - avg_similarity
```

## 3. 시스템 프롬프트와 사용자 프롬프트

### 3.1 시스템 프롬프트 설계

```python
class SystemPromptDesigner:
    """시스템 프롬프트 설계기"""
    
    def __init__(self):
        self.system_prompt_templates = {
            "assistant": self._create_assistant_prompt,
            "analyst": self._create_analyst_prompt,
            "teacher": self._create_teacher_prompt,
            "translator": self._create_translator_prompt,
            "coder": self._create_coder_prompt
        }
    
    def create_system_prompt(self, role, domain=None, constraints=None):
        """역할별 시스템 프롬프트 생성"""
        base_prompt = self.system_prompt_templates.get(role, self._create_generic_prompt)()
        
        # 도메인 특화 지식 추가
        if domain:
            domain_expertise = self._add_domain_expertise(domain)
            base_prompt += f"\n\n도메인 전문성:\n{domain_expertise}"
        
        # 제약 조건 추가
        if constraints:
            constraint_text = self._format_constraints(constraints)
            base_prompt += f"\n\n제약 조건:\n{constraint_text}"
        
        return base_prompt
    
    def _create_assistant_prompt(self):
        """일반 어시스턴트 프롬프트"""
        return """
당신은 도움이 되고 정확한 AI 어시스턴트입니다.

핵심 원칙:
- 정확하고 사실에 기반한 정보를 제공합니다
- 불확실한 내용은 명시적으로 표현합니다  
- 사용자의 요청을 주의 깊게 이해하고 맞춤형 응답을 제공합니다
- 윤리적이고 안전한 가이드라인을 준수합니다

응답 스타일:
- 명확하고 구조적인 설명
- 적절한 예시와 구체적인 정보 포함
- 전문적이면서도 접근하기 쉬운 톤 유지
"""
    
    def _create_analyst_prompt(self):
        """분석가 프롬프트"""
        return """
당신은 데이터와 정보를 체계적으로 분석하는 전문 분석가입니다.

분석 접근법:
- 주어진 데이터를 객관적으로 검토합니다
- 패턴, 트렌드, 이상 징후를 식별합니다
- 근거 기반의 결론을 도출합니다
- 불확실성과 한계점을 명시합니다

분석 결과 제시:
- 핵심 발견사항을 우선 제시
- 지지 증거와 데이터 포함
- 대안적 해석 가능성 고려
- 실행 가능한 인사이트 제공
"""
    
    def _create_teacher_prompt(self):
        """교사 프롬프트"""
        return """
당신은 학습자의 이해를 돕는 전문 교육자입니다.

교육 철학:
- 학습자의 현재 수준을 파악하여 맞춤형 설명 제공
- 복잡한 개념을 단계적으로 분해하여 설명
- 실제 예시와 비유를 통한 이해 촉진
- 능동적 학습을 격려하는 질문 제시

교수법:
- 기초 개념부터 점진적으로 발전
- 다양한 학습 스타일 고려
- 즉각적인 피드백과 격려 제공
- 실습과 적용 기회 창출
"""
    
    def _add_domain_expertise(self, domain):
        """도메인 전문성 추가"""
        domain_knowledge = {
            "healthcare": """
의료 분야 전문 지식:
- 의학 용어와 절차에 대한 정확한 이해
- 환자 안전과 프라이버시 최우선 고려
- 의료 가이드라인과 모범 사례 준수
- 의료 조언은 정보 제공 목적으로만 제한
""",
            "finance": """
금융 분야 전문 지식:
- 금융 상품과 시장 메커니즘 이해
- 리스크 관리와 규제 준수 중시
- 투자 조언의 한계와 위험성 명시
- 개인 재정 정보 보호 우선
""",
            "technology": """
기술 분야 전문 지식:
- 최신 기술 트렌드와 발전 동향 파악
- 기술적 구현과 아키텍처 이해
- 보안과 프라이버시 고려사항 포함
- 실무 적용 가능한 솔루션 제시
""",
            "legal": """
법률 분야 전문 지식:
- 법률 원칙과 절차에 대한 이해
- 관할권과 법률 변화 고려
- 법률 조언의 한계 명시
- 전문 법률 상담 권장
"""
        }
        
        return domain_knowledge.get(domain, "일반적인 전문 지식을 바탕으로 답변합니다.")

class UserPromptOptimizer:
    """사용자 프롬프트 최적화기"""
    
    def __init__(self):
        self.optimization_strategies = {
            "clarity": self._improve_clarity,
            "specificity": self._add_specificity,
            "context": self._enrich_context,
            "structure": self._improve_structure
        }
    
    def optimize_user_prompt(self, original_prompt, optimization_goals=None):
        """사용자 프롬프트 최적화"""
        if optimization_goals is None:
            optimization_goals = ["clarity", "specificity", "context"]
        
        optimized_prompt = original_prompt
        optimization_log = []
        
        for goal in optimization_goals:
            if goal in self.optimization_strategies:
                optimizer_func = self.optimization_strategies[goal]
                optimized_prompt, changes = optimizer_func(optimized_prompt)
                optimization_log.append({
                    "goal": goal,
                    "changes": changes
                })
        
        return {
            "original": original_prompt,
            "optimized": optimized_prompt,
            "optimization_log": optimization_log,
            "improvement_score": self._calculate_improvement_score(original_prompt, optimized_prompt)
        }
    
    def _improve_clarity(self, prompt):
        """명확성 개선"""
        improvements = []
        optimized = prompt
        
        # 모호한 표현 식별 및 개선
        ambiguous_patterns = {
            "이것": "구체적인 대상",
            "좀": "",
            "약간": "정확한 정도",
            "대충": "자세히",
            "뭔가": "구체적인 내용"
        }
        
        for ambiguous, replacement in ambiguous_patterns.items():
            if ambiguous in optimized:
                if replacement:
                    optimized = optimized.replace(ambiguous, replacement)
                    improvements.append(f"'{ambiguous}'를 '{replacement}'로 명확화")
                else:
                    optimized = optimized.replace(ambiguous, "")
                    improvements.append(f"불필요한 '{ambiguous}' 제거")
        
        # 질문 형태로 변환
        if not optimized.endswith("?") and not any(word in optimized for word in ["해주세요", "부탁드립니다", "알려주세요"]):
            optimized += "해주세요."
            improvements.append("명확한 요청 형태로 변환")
        
        return optimized, improvements
    
    def _add_specificity(self, prompt):
        """구체성 추가"""
        improvements = []
        optimized = prompt
        
        # 출력 형식 지정 추가
        if "형식" not in optimized and "포맷" not in optimized:
            format_addition = "\n\n출력 형식: 명확하고 구조화된 답변으로 제공해주세요."
            optimized += format_addition
            improvements.append("출력 형식 지정 추가")
        
        # 길이 제한 추가 (필요시)
        if len(optimized.split()) > 10 and "길이" not in optimized:
            length_guideline = " 간결하면서도 포괄적인 답변으로"
            optimized = optimized.replace("해주세요", length_guideline + " 해주세요")
            improvements.append("답변 길이 가이드라인 추가")
        
        return optimized, improvements
    
    def _enrich_context(self, prompt):
        """컨텍스트 강화"""
        improvements = []
        optimized = prompt
        
        # 목적 명시
        if "목적" not in optimized and "이유" not in optimized:
            context_addition = "\n\n이 정보가 필요한 목적: "
            optimized = context_addition + optimized
            improvements.append("목적 명시 섹션 추가")
        
        # 대상 청중 명시
        if "대상" not in optimized and len(optimized.split()) > 15:
            audience_note = " (일반인도 이해할 수 있도록)"
            optimized = optimized.replace("해주세요", audience_note + " 해주세요")
            improvements.append("대상 청중 명시")
        
        return optimized, improvements

class PromptChaining:
    """프롬프트 체이닝 시스템"""
    
    def __init__(self, model):
        self.model = model
        self.chain_history = []
    
    def execute_chain(self, initial_prompt, chain_steps):
        """프롬프트 체인 실행"""
        current_context = initial_prompt
        results = []
        
        for step_num, step_config in enumerate(chain_steps, 1):
            step_prompt = self._build_step_prompt(
                current_context, 
                step_config, 
                step_num
            )
            
            response = self.model.generate(
                step_prompt,
                temperature=step_config.get('temperature', 0.3)
            )
            
            # 결과 저장
            step_result = {
                "step": step_num,
                "prompt": step_prompt,
                "response": response,
                "config": step_config
            }
            results.append(step_result)
            
            # 다음 단계를 위한 컨텍스트 업데이트
            current_context = self._update_context(
                current_context, 
                response, 
                step_config
            )
        
        return {
            "final_result": results[-1]["response"],
            "chain_results": results,
            "execution_summary": self._generate_execution_summary(results)
        }
    
    def _build_step_prompt(self, context, step_config, step_num):
        """단계별 프롬프트 구성"""
        step_instruction = step_config['instruction']
        
        if step_num == 1:
            return f"{context}\n\n{step_instruction}"
        else:
            return f"""
이전 단계의 결과를 바탕으로 다음 작업을 수행하세요:

이전 컨텍스트: {context}

현재 작업: {step_instruction}
"""
    
    def create_analysis_chain(self, data, analysis_type="comprehensive"):
        """분석 체인 생성"""
        chain_templates = {
            "comprehensive": [
                {
                    "instruction": "주어진 데이터의 핵심 특성과 패턴을 식별하세요.",
                    "temperature": 0.2
                },
                {
                    "instruction": "식별된 패턴의 원인과 의미를 분석하세요.",
                    "temperature": 0.3
                },
                {
                    "instruction": "분석 결과를 바탕으로 실행 가능한 인사이트와 권장사항을 제시하세요.",
                    "temperature": 0.4
                }
            ],
            "problem_solving": [
                {
                    "instruction": "문제의 핵심 요소와 제약 조건을 명확히 정의하세요.",
                    "temperature": 0.1
                },
                {
                    "instruction": "가능한 해결 방안들을 브레인스토밍하고 각각의 장단점을 평가하세요.",
                    "temperature": 0.5
                },
                {
                    "instruction": "최적의 해결책을 선택하고 구체적인 실행 계획을 수립하세요.",
                    "temperature": 0.3
                }
            ]
        }
        
        return chain_templates.get(analysis_type, chain_templates["comprehensive"])
```

## 4. 컨텍스트 길이와 컨텍스트 효율성

### 4.1 컨텍스트 길이 관리

```python
class ContextManager:
    """컨텍스트 관리자"""
    
    def __init__(self, model_context_limit=4096):
        self.context_limit = model_context_limit
        self.tokenizer = self._load_tokenizer()
        self.compression_strategies = {
            "summarization": self._summarize_content,
            "extraction": self._extract_key_information,
            "hierarchical": self._hierarchical_compression,
            "selective": self._selective_retention
        }
    
    def manage_context(self, content, strategy="adaptive"):
        """컨텍스트 관리 실행"""
        current_tokens = self._count_tokens(content)
        
        if current_tokens <= self.context_limit:
            return {
                "status": "no_compression_needed",
                "content": content,
                "original_tokens": current_tokens,
                "final_tokens": current_tokens
            }
        
        if strategy == "adaptive":
            strategy = self._select_optimal_strategy(content)
        
        compression_func = self.compression_strategies.get(strategy)
        if not compression_func:
            raise ValueError(f"Unknown compression strategy: {strategy}")
        
        compressed_content = compression_func(content)
        final_tokens = self._count_tokens(compressed_content)
        
        return {
            "status": "compressed",
            "content": compressed_content,
            "strategy_used": strategy,
            "original_tokens": current_tokens,
            "final_tokens": final_tokens,
            "compression_ratio": final_tokens / current_tokens
        }
    
    def _summarize_content(self, content):
        """내용 요약"""
        if len(content) < 1000:
            return content
        
        # 섹션별로 분할
        sections = self._split_into_sections(content)
        summarized_sections = []
        
        for section in sections:
            if len(section) > 200:
                summary = self._generate_section_summary(section)
                summarized_sections.append(summary)
            else:
                summarized_sections.append(section)
        
        return "\n\n".join(summarized_sections)
    
    def _extract_key_information(self, content):
        """핵심 정보 추출"""
        extraction_patterns = {
            "facts": r'(?:사실|팩트|정보)[:：]\s*(.+)',
            "numbers": r'\d+(?:\.\d+)?(?:[%％]|\s*(?:개|명|건|회|번|시간|분|초))',
            "entities": r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            "dates": r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}',
            "keywords": self._extract_keywords
        }
        
        extracted_info = {}
        
        for category, pattern in extraction_patterns.items():
            if category == "keywords":
                extracted_info[category] = pattern(content)
            else:
                matches = re.findall(pattern, content, re.IGNORECASE)
                extracted_info[category] = matches
        
        # 추출된 정보를 구조화된 형태로 재구성
        structured_content = self._structure_extracted_info(extracted_info)
        return structured_content
    
    def _hierarchical_compression(self, content):
        """계층적 압축"""
        hierarchy_levels = [
            ("high_priority", ["결론", "요약", "핵심", "중요"]),
            ("medium_priority", ["분석", "설명", "방법", "과정"]),
            ("low_priority", ["배경", "부가", "참고", "예시"])
        ]
        
        compressed_sections = {}
        
        for priority, keywords in hierarchy_levels:
            sections = self._find_sections_by_keywords(content, keywords)
            
            if priority == "high_priority":
                # 높은 우선순위는 전체 유지
                compressed_sections[priority] = sections
            elif priority == "medium_priority":
                # 중간 우선순위는 요약
                compressed_sections[priority] = [
                    self._summarize_section(section) for section in sections
                ]
            else:
                # 낮은 우선순위는 키포인트만 추출
                compressed_sections[priority] = [
                    self._extract_keypoints(section) for section in sections
                ]
        
        # 우선순위 순으로 재구성
        final_content = []
        for priority in ["high_priority", "medium_priority", "low_priority"]:
            if compressed_sections[priority]:
                final_content.extend(compressed_sections[priority])
        
        return "\n\n".join(final_content)
    
    def optimize_context_efficiency(self, prompts_history):
        """컨텍스트 효율성 최적화"""
        optimization_results = {}
        
        # 중복 정보 제거
        deduplicated = self._remove_duplicate_information(prompts_history)
        optimization_results["deduplication"] = {
            "original_length": len(prompts_history),
            "optimized_length": len(deduplicated),
            "reduction": 1 - len(deduplicated) / len(prompts_history)
        }
        
        # 정보 우선순위 재정렬
        prioritized = self._prioritize_information(deduplicated)
        optimization_results["prioritization"] = prioritized
        
        # 컨텍스트 윈도우 슬라이딩
        windowed = self._apply_sliding_window(prioritized)
        optimization_results["windowing"] = windowed
        
        return optimization_results

class ContextEfficiencyAnalyzer:
    """컨텍스트 효율성 분석기"""
    
    def __init__(self):
        self.efficiency_metrics = [
            "information_density",
            "relevance_score", 
            "redundancy_rate",
            "compression_potential"
        ]
    
    def analyze_efficiency(self, context_data):
        """컨텍스트 효율성 분석"""
        analysis_results = {}
        
        for metric in self.efficiency_metrics:
            analyzer_func = getattr(self, f"_calculate_{metric}")
            analysis_results[metric] = analyzer_func(context_data)
        
        # 종합 효율성 점수
        overall_efficiency = self._calculate_overall_efficiency(analysis_results)
        
        return {
            "efficiency_score": overall_efficiency,
            "detailed_metrics": analysis_results,
            "recommendations": self._generate_efficiency_recommendations(analysis_results)
        }
    
    def _calculate_information_density(self, context_data):
        """정보 밀도 계산"""
        total_tokens = len(context_data.split())
        unique_concepts = len(set(self._extract_concepts(context_data)))
        
        density = unique_concepts / total_tokens if total_tokens > 0 else 0
        
        return {
            "density_score": density,
            "total_tokens": total_tokens,
            "unique_concepts": unique_concepts,
            "interpretation": self._interpret_density_score(density)
        }
    
    def _calculate_relevance_score(self, context_data):
        """관련성 점수 계산"""
        # 키워드 빈도 분석
        keywords = self._extract_keywords(context_data)
        keyword_frequency = {}
        
        for keyword in keywords:
            keyword_frequency[keyword] = context_data.lower().count(keyword.lower())
        
        # 관련성 점수 계산 (주요 키워드의 분포 기반)
        if not keyword_frequency:
            return {"relevance_score": 0, "keywords": []}
        
        top_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        total_mentions = sum(freq for _, freq in top_keywords)
        
        # 상위 키워드가 전체에서 차지하는 비율
        relevance_score = total_mentions / len(context_data.split())
        
        return {
            "relevance_score": relevance_score,
            "top_keywords": top_keywords,
            "total_mentions": total_mentions
        }
    
    def adaptive_context_optimization(self, context_history, current_task):
        """적응적 컨텍스트 최적화"""
        # 현재 작업과의 관련성 분석
        relevance_scores = []
        for context_item in context_history:
            relevance = self._calculate_task_relevance(context_item, current_task)
            relevance_scores.append((context_item, relevance))
        
        # 관련성 순으로 정렬
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 동적 컨텍스트 윈도우 크기 결정
        optimal_window_size = self._determine_optimal_window_size(
            relevance_scores, 
            current_task
        )
        
        # 최적화된 컨텍스트 구성
        optimized_context = []
        current_tokens = 0
        max_tokens = optimal_window_size
        
        for context_item, relevance in relevance_scores:
            item_tokens = len(context_item.split())
            
            if current_tokens + item_tokens <= max_tokens:
                optimized_context.append(context_item)
                current_tokens += item_tokens
            elif relevance > 0.8:  # 매우 관련성이 높은 경우 압축하여 포함
                compressed_item = self._compress_high_relevance_item(context_item)
                compressed_tokens = len(compressed_item.split())
                
                if current_tokens + compressed_tokens <= max_tokens:
                    optimized_context.append(compressed_item)
                    current_tokens += compressed_tokens
        
        return {
            "optimized_context": optimized_context,
            "tokens_used": current_tokens,
            "optimization_ratio": current_tokens / sum(len(item.split()) for item in context_history)
        }
```

## 결론

프롬프트 엔지니어링과 인컨텍스트 학습은 **LLM의 성능을 극대화하는 핵심 기술**입니다.

**핵심 인사이트:**
- **구조화된 접근**: 체계적인 프롬프트 설계가 일관된 고품질 결과를 보장
- **적응적 학습**: 제로샷과 퓨샷 학습의 적절한 조합으로 효율성 극대화
- **컨텍스트 최적화**: 한정된 컨텍스트 윈도우에서 정보 밀도와 관련성 최적화
- **반복적 개선**: 지속적인 프롬프트 평가와 개선을 통한 성능 향상

다음 포스트에서는 **프롬프트 엔지니어링의 실전 모범 사례**를 상세히 살펴보겠습니다.

---

**시리즈 연결:**
- 다음: [프롬프트 엔지니어링 실전 모범 사례 가이드]({% post_url 2024-12-18-prompt-engineering-best-practices %})

**참고 자료:**
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [In-Context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)