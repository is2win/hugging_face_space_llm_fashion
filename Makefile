check_nvcc_version:
	@nvcc --version

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


all: install lint test format