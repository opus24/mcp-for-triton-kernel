# Triton 커널 개발 가이드

## 개요

Triton은 GPU 커널을 Python으로 작성할 수 있게 해주는 언어/컴파일러입니다.
PyTorch 연산을 Triton으로 변환하면 custom fusion, memory optimization 등의 이점을 얻을 수 있습니다.

## 기본 구조

```python
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_name(
    # 포인터 인자들 (입력/출력 텐서)
    input_ptr,
    output_ptr,
    # 스칼라 인자들 (크기, stride 등)
    N,
    # constexpr 인자들 (컴파일 타임 상수)
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Program ID로 현재 블록 위치 파악
    pid = tl.program_id(0)
    
    # 2. 블록 내 오프셋 계산
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 마스크 생성 (경계 체크)
    mask = offsets < N
    
    # 4. 데이터 로드
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 5. 연산 수행
    result = data * 2  # 예시
    
    # 6. 결과 저장
    tl.store(output_ptr + offsets, result, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    
    # 그리드 크기 계산
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    
    # 커널 실행
    kernel_name[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
```

## 개발 프로세스

1. **Torch 연산 이해**: 변환할 연산의 입출력, 동작 방식 파악
2. **메모리 레이아웃 설계**: 텐서 접근 패턴, stride 계산
3. **블록 분할 전략**: BLOCK_SIZE, 그리드 차원 결정
4. **커널 구현**: tl 함수들을 사용해 연산 구현
5. **정확성 검증**: torch 결과와 비교 (allclose)
6. **성능 최적화**: 메모리 coalescing, 캐시 활용

## 주요 고려사항

### 메모리 접근
- Coalesced access: 연속 메모리 접근이 빠름
- Stride 고려: row-major vs column-major
- Shared memory: 반복 접근 데이터는 캐싱

### 병렬화
- 각 program은 독립적으로 실행
- program_id로 작업 분할
- 동기화는 블록 내에서만 가능

### 수치 안정성
- fp16/bf16 사용 시 overflow 주의
- softmax 등은 max 빼기 트릭 필요
- accumulator는 fp32 사용 권장

