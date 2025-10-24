# 검색 엔진 등록 가이드

이 문서는 AI Development Blog를 구글, 네이버, 다음 등의 검색 엔진에 등록하는 방법을 안내합니다.

## 1. Google Search Console 등록

1. [Google Search Console](https://search.google.com/search-console/)에 접속
2. "속성 추가" 클릭
3. "URL 접두어" 선택하고 사이트 URL 입력: `https://leeyonghe.github.io/ai-blog`
4. HTML 태그 방법으로 소유권 확인:
   - 제공된 메타 태그의 content 값을 복사
   - `_config.yml`의 `google_site_verification` 값에 입력
5. 사이트맵 제출: `https://leeyonghe.github.io/ai-blog/sitemap.xml`
6. RSS 피드 제출: `https://leeyonghe.github.io/ai-blog/feed.xml`

## 2. 네이버 웹마스터 도구 등록

1. [네이버 웹마스터 도구](https://searchadvisor.naver.com/)에 접속
2. "사이트 등록" 클릭
3. 사이트 URL 입력: `https://leeyonghe.github.io/ai-blog`
4. HTML 태그 방법으로 소유권 확인:
   - 제공된 메타 태그의 content 값을 복사
   - `_config.yml`의 `naver_site_verification` 값에 입력
5. 사이트맵 제출: `https://leeyonghe.github.io/ai-blog/sitemap.xml`
6. RSS 피드 제출: `https://leeyonghe.github.io/ai-blog/feed.xml`

## 3. 다음(Daum) 검색 등록

1. [다음 검색등록](https://register.search.daum.net/index.daum)에 접속
2. "URL 등록" 선택
3. 사이트 URL과 설명 입력
4. 이메일 주소 입력 후 제출

## 4. 빙(Bing) 웹마스터 도구 등록 (선택사항)

1. [Bing 웹마스터 도구](https://www.bing.com/webmasters/)에 접속
2. Google Search Console 계정으로 가져오기 또는 수동 등록
3. 사이트맵 제출: `https://leeyonghe.github.io/ai-blog/sitemap.xml`

## 설정해야 할 값들

`_config.yml` 파일에 다음 값들을 설정하세요:

```yaml
# 검색 엔진 최적화
google_site_verification: "여기에_구글_인증_코드_입력"
naver_site_verification: "여기에_네이버_인증_코드_입력"
```

## 추가 최적화 팁

1. **정기적인 콘텐츠 업데이트**: 검색 엔진은 정기적으로 업데이트되는 사이트를 선호합니다.
2. **내부 링크 구축**: 관련 포스트 간의 링크를 추가하세요.
3. **키워드 최적화**: 제목과 본문에 관련 키워드를 자연스럽게 포함하세요.
4. **메타 설명 작성**: 각 포스트에 고유한 메타 설명을 추가하세요.
5. **이미지 최적화**: 이미지에 alt 속성을 추가하고 파일 크기를 최적화하세요.

## 모니터링

- Google Search Console에서 검색 성능 모니터링
- 네이버 웹마스터 도구에서 수집 현황 확인
- 정기적으로 사이트맵 업데이트 확인

## 문제 해결

- 인덱싱이 안 될 경우: robots.txt 확인, 사이트맵 재제출
- 검색 노출이 낮을 경우: 콘텐츠 품질 개선, 키워드 최적화
- 크롤링 오류 발생시: 웹마스터 도구에서 오류 내용 확인 후 수정