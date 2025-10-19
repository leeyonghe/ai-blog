# GitHub Pages Troubleshooting Guide

## 설정 완료 사항 ✅

1. **_config.yml 수정 완료**
   - GitHub Pages 호환 설정 추가
   - 플러그인 설정 (jekyll-feed, jekyll-seo-tag, jekyll-sitemap)
   - remote_theme: plainwhite 설정
   - permalink 구조 수정

2. **Gemfile 수정 완료**
   - github-pages gem 추가
   - 필수 Jekyll 플러그인 추가
   - 테마 설정 유지

3. **GitHub Actions 워크플로우 추가**
   - 자동 빌드 및 배포 설정
   - Ruby 환경 설정
   - HTML 프루프 테스트 포함

4. **index.md 개선**
   - 한국어 콘텐츠 추가
   - 블로그 소개 및 주요 콘텐츠 설명

## 다음 단계 📋

### GitHub Repository Settings에서 확인 필요:

1. **Pages 설정 (Repository > Settings > Pages)**
   - Source: Deploy from a branch
   - Branch: gh-pages (GitHub Actions가 자동 생성)
   - 또는 Source: GitHub Actions 선택

2. **Actions 권한 확인 (Repository > Settings > Actions > General)**
   - Actions permissions: Allow all actions and reusable workflows
   - Workflow permissions: Read and write permissions 체크

3. **GitHub Actions 실행 확인**
   - Actions 탭에서 워크플로우 실행 상태 확인
   - 빌드 성공 여부 확인

### 배포 완료 후 확인사항:

- 사이트 URL: `https://leeyonghe.github.io/ai-blog/`
- DNS 전파 시간: 최대 10분 소요
- SSL 인증서 활성화: 자동 (최대 24시간)

## 문제 해결 방법 🔧

### 1. 404 오류가 지속되는 경우:
```bash
# 로컬에서 Jekyll 서버 실행하여 테스트
bundle install
bundle exec jekyll serve
```

### 2. 빌드 실패 시:
- Actions 탭에서 로그 확인
- Gemfile.lock 파일 삭제 후 재시도
- Jekyll 버전 호환성 확인

### 3. 테마 문제 시:
- _config.yml의 remote_theme 설정 확인
- plainwhite 테마 문서 참조

## 추가 개선사항 💡

1. **SEO 최적화**
   - sitemap.xml 자동 생성됨
   - robots.txt 추가 고려

2. **성능 최적화**
   - 이미지 압축
   - CDN 설정 고려

3. **커스텀 도메인 (선택사항)**
   - CNAME 파일 추가
   - DNS 설정 필요