# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np # 수학적 연산을 쉽게 하기 위함.
import os

# 깔금한 그래프 출력을 위해
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import os # csv 파일 호출을 하기 위해, os = operating system
import tarfile # 여러 개의 파일을 tar 형식으로 합치거나 이를 해제할 때 사용하는 모듈
import urllib.request # URL(대부분 HTTP)을 여는 데 도움이 되는 함수와 클래스를 정의

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd # 데이터를 읽어 들이기 위함.

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head() # 처음 5행만 확인함.

housing.info() # 데이터에 대한 간략한 설명, 전체 행, 각 특성의 데이터 타입, 널이 아닌 값의 개수 확인하는데 유용

housing["ocean_proximity"].value_counts()

housing.describe() # 숫자형 특성의 요약 정보를 확인, 널 값은 제외, std = 값이 퍼져있는 정도, 표준편차

# 숫자형 특성을 히스토그램으로 그려봄.
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

# 노트북의 실행 결과가 동일하도록
# 컴퓨터 프로그램에서 발생한 랜덤값은 무작위 수가 아니라, 특정 시작 숫자값을 정해주면 정해진 알고리즘에 따라 마치 난수처럼
# 보이는 수열을 생성함. 이때 설정해주는 특정 시작 숫자가 시드(seed)
# 시드값은 보통 현재 시각 등을 이용해 자동으로 정하기도 하지만, 직접 사람이 수동으로 설정할 수 있음.
# 42 의미 : Life, the universe, everything 을 컴퓨터가 계산한 값
np.random.seed(42)

import numpy as np

def train_test_split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = train_test_split(housing, 0.2)
len(train_set)
len(test_set)

# 데이터의 CRC(순환중복검사)-32 값(32비트 정수)을 구하는 방법
# Python2 이하 버전 시 동일한 숫자 값을 생성하기 위해서는 .crc32(data) & 0xffffffff
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# identifier = 식별자
import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

# `index` 열이 추가된 데이터프레임을 반환합니다
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

test_set.head()