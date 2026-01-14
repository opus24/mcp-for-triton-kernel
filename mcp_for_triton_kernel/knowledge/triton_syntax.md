# Triton 문법 레퍼런스

## 데코레이터

### @triton.jit
커널 함수를 JIT 컴파일합니다.
```python
@triton.jit
def my_kernel(...):
    pass
```

### @triton.autotune
여러 설정을 자동으로 테스트해 최적 설정을 선택합니다.
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
    ],
    key=["N"],  # 이 값이 바뀌면 재튜닝
)
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    pass
```

## triton.language (tl) 주요 함수

### Program 정보
```python
tl.program_id(axis)      # 현재 프로그램의 ID (0, 1, 2 축)
tl.num_programs(axis)    # 해당 축의 총 프로그램 수
```

### 인덱스 생성
```python
tl.arange(start, end)    # [start, end) 범위의 정수 벡터
```

### 메모리 연산
```python
tl.load(ptr, mask=None, other=0.0)   # 메모리에서 로드
tl.store(ptr, value, mask=None)       # 메모리에 저장
tl.atomic_add(ptr, value, mask=None)  # 원자적 덧셈
```

### 수학 연산
```python
# 기본 연산
tl.exp(x)
tl.log(x)
tl.sqrt(x)
tl.abs(x)
tl.sin(x), tl.cos(x)

# 특수 연산
tl.sigmoid(x)
tl.softmax(x, axis=0)
tl.ravel(x)  # flatten
```

### Reduction 연산
```python
tl.sum(x, axis=0)
tl.max(x, axis=0)
tl.min(x, axis=0)
tl.argmax(x, axis=0)
tl.argmin(x, axis=0)
```

### 형변환
```python
x.to(tl.float32)
x.to(tl.float16)
x.to(tl.int32)
```

### 조건문
```python
tl.where(condition, x, y)  # condition ? x : y
```

### 디버깅
```python
tl.device_print("message", value)  # GPU에서 출력
tl.debug_barrier()                  # 동기화 배리어
```

## 타입 (tl.constexpr)

컴파일 타임에 알려져야 하는 값들:
- BLOCK_SIZE
- 텐서 shape의 일부
- 조건부 컴파일에 사용되는 값

```python
def kernel(..., BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE는 컴파일 타임에 결정됨
```

## 커널 실행

```python
# 그리드 정의 방법 1: 튜플
grid = (num_blocks_x, num_blocks_y, num_blocks_z)

# 그리드 정의 방법 2: 람다 (autotune과 함께 사용)
grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

# 실행
kernel[grid](arg1, arg2, ..., BLOCK_SIZE=256)
```

## 제약사항

1. **BLOCK_SIZE는 2의 거듭제곱**: 64, 128, 256, 512, 1024 등
2. **동적 shape 제한**: 일부 연산은 컴파일 타임에 shape을 알아야 함
3. **재귀 불가**: 커널 내에서 다른 커널 호출 불가
4. **Python 객체 사용 불가**: 커널 내에서는 tl 연산만 사용
5. **최대 블록 크기**: GPU에 따라 1024 또는 2048 스레드

## 일반적인 패턴

### 1D 벡터 처리
```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < N
```

### 2D 타일 처리
```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

### Row-wise Reduction
```python
row_idx = tl.program_id(0)
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < num_cols
data = tl.load(ptr + row_idx * stride + col_offsets, mask=mask)
result = tl.sum(data, axis=0)
```

