# Env

运行环境与本地模型权重。实验生成走 Ollama；这里只放 conda/pip 说明和 HuggingFace 下载副本。实验日志在 `Testing/logs/`，不在这里。

```text
Env/
  environment-sociosim.yml
  requirements-sociosim.txt
  models/Meta-Llama-3.1-8B-Instruct/
  models/modelscope_cache/
  scripts/                 # Llama 下载
  logs/                    # 下载日志
```

创建 conda 环境：

```bash
conda env create -f Env/environment-sociosim.yml
conda activate sociosim
```

下载 Llama-3.1-8B-Instruct：

```bash
bash Env/scripts/start_download_llama31.sh
# 或
bash Env/scripts/curl_download_llama31.sh
```
