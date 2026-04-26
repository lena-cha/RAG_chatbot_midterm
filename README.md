과제 제출용 (study)

## 포함물
- `rag_app.py`: 제출용 파일 복사 스크립트
- `requirements.txt`: 의존성 (자동 복사됨)
- `docs/midterm_report_template.md`: 5페이지 이상 리포트 템플릿 (자동 복사됨)
- `DEPLOY_LINK.txt`: 배포된 Streamlit URL 기입
- `data/`: 사용한 PDF/TXT 파일을 여기에 넣어 제출

## 실행
```bash
pip install -r requirements.txt
python rag_app.py  # 제출물 복사 후 생성됨: rag_app_impl.py
streamlit run rag_app_impl.py
```

## 배포 링크
- `DEPLOY_LINK.txt`에 URL 기입 후 제출

## 압축 제출
PowerShell:
```bash
Compress-Archive -Path "study" -DestinationPath "이름_중간고사_코드.zip" -Force
```


