# Korean-Lip-Reading

> 국민대학교 2024-1 전자종합공학설계1

담당교수 : 정성희 교수

팀원 : 안태현, 김기현

## 주제
> Korean Lip Reading

## 데이터셋 추출
![자음 분류표 (1)](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/5a65cb3a-c025-4980-8b02-a3301e67a342)

⇒ 입술(양순음)로 구분할 수 있는 자음은 ㅂ,ㅍ,ㅃ,ㅁ 밖에 없음

⇒ 따라서, **‘양순음 + 모음’** 의 음절 단위로 데이터를 수집하기로 결정

⇒ 추출할 음절 : 아 여 마 오 애 임 으 어 워 왠 우 안 (총 12개 음절)


● Augmentation(총 4가지 방식) 진행 → 데이터량 ↑
1. Flip(좌우 반전)

2. Crop(자르기)
  
3. Noise(노이즈)
  
4. Distort(왜곡)
  
5. Brightness(밝기 조절)

→ 현재 데이터 량 : 12음절 * 5(원본 + augmentation) * 10(이미지 개수) * 7셋 = 4200개 (6.1)

→ 현재 데이터 량 : 12음절 * 5(원본 + augmentation) * 10(이미지 개수) * 13셋 = 7800개 (6.2)

→ 현재 데이터 량 : 12음절 * 6(원본 + augmentation) * 10(이미지 개수) * 28셋 = 20160개 (6.3)

→ 현재 데이터 량 : 12음절 * 6(원본 + augmentation) * 10(이미지 개수) * 34셋(Val : 27개 + Test : 7개) = 24480개 (6.4)

→ 현재 데이터 량 : 12음절 * 6(원본 + augmentation) * 10(이미지 개수) * 51셋(Val : 40개 + Test : 10개) = 36720개 (6.5)

● 추출한 데이터 라벨링

![라벨링](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/cc72a1a5-b5e9-479a-b774-8a3a2127c8da)

● Depth 카메라와 68-Landmarks를 활용해 입술 부분만 추출

![입술추출](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/4829fb99-333b-490c-a459-2af444d24b6d)


## 결과
※ 과적합 방지 파라미터 적용 후 학습 결과

![학습완료 - 얼리 스타핑 적용](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/aa0503d3-8aa4-419e-b697-19debf008651)

※ 5-fold cross validation 적용 결과 (폴드 1,3,5)

![폴드1](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/bd8ee45c-c265-43c6-a434-53c5c3f9ed58)

![폴드3](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/826c3209-5355-42db-9401-565151e57871)

![폴드5](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/31019f1a-3d4f-4cd7-854b-b4ebe7972fc7)


※ Accrucay 향상

![초기 테스트 결과](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/a261096b-052d-46b3-beee-99b1b8926564) 
![최종 테스트 결과](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/cceba588-5a92-4af3-a771-eaffbd1522dc)


※ 인터페이스 적용 화면

![인터페이스 화면](https://github.com/ahntae98/Korean-Lip-Reading/assets/133379277/c7925695-f0b8-4339-aa08-157a02dd0f2e)

