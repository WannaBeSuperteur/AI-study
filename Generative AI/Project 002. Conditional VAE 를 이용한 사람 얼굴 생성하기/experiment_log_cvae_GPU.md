## 실험 개요
* 실험 조건
  * 기본 설정
    * **all 16,864 images / 24 epochs**
    * **GPU에서 작동**
* 용어 설명
  * HIDDEN_DIMS : latent vector 차원 개수
  * MSE Loss weight : KL Loss의 weight에 대한 reconstruction loss (MSE Loss) 의 weight의 전체 loss에서의 비중
  * info : conditional VAE의 condition에 해당하는 값
    * 성별 정보 (male prob, female prob)
    * hair color
    * mouth (입을 벌린 정도)
    * eyes (눈을 뜬 정도)

## baseline model (2024.04.11 22시)
TBU