# ローカルLLM（vLLM）でOASIS SNSシミュレーション実行ガイド

## 📋 概要

このガイドでは、Ubuntu PC上のvLLMサーバーを使用してOASISのTwitterシミュレーションを実行する方法を説明します。

## 🔧 前提条件

### Windows/ローカルマシン
- Python 3.9以上
- OASIS がインストールされている

### Ubuntu PC
- Python 3.9以上  
- PyTorch
- vLLM

## 📦 セットアップ手順

### 1. Ubuntu PCでのvLLM サーバー起動

```bash
# PyTorchとvLLMをインストール（初回のみ）
pip install torch vllm

# vLLMサーバーを起動（モデルは任意）
vllm serve mistralai/Mistral-7B-v0.1 --host 0.0.0.0 --port 8000
```

**注意**: 
- `--host 0.0.0.0` でネットワークからアクセス可能にします
- ポート `8000` はカスタマイズ可能です
- モデルのダウンロードは初回のみ時間がかかります

### 2. Windows/ローカルマシンでの環境準備

```bash
# 作業ディレクトリに移動
cd c:\Users\81901\Desktop\oasis\examples

# 必要なライブラリをインストール
pip install openai

# llm_adapter.py と twitter_vllm_remote.py があることを確認
ls -la *.py | grep -E "(llm_adapter|twitter_vllm)"
```

## 🚀 実行方法

### 基本的な実行（ローカルホスト）

```bash
cd examples
python twitter_vllm_remote.py
```

### Ubuntu PCのリモートvLLMを使用

```bash
# Ubuntu PCのIPアドレスを指定（例: 192.168.1.100）
python twitter_vllm_remote.py --vllm-host 192.168.1.100
```

### カスタムペルソナを使用

```bash
# custom_users.csv を使用
python twitter_vllm_remote.py \
  --vllm-host 192.168.1.100 \
  --profile-path ../data/custom_users.csv
```

### シミュレーション期間を長くする

```bash
# 20ステップ実行
python twitter_vllm_remote.py \
  --vllm-host 192.168.1.100 \
  --steps 20 \
  --num-active 6
```

### すべてのオプション

```bash
python twitter_vllm_remote.py --help
```

## 📊 ペルソナカスタマイズ

### 現在のペルソナ構成（custom_users.csv）

| 名前 | ID | 役割 | 説明 |
|-----|-----|-----|-----|
| Manager_A | 1001 | マネージャー | プロジェクト進行責任者。チームに仕事を依頼 |
| Expert_B | 1002 | 専門家 | 経験豊富。専門的なアドバイスを提供 |
| Leader_C | 1003 | リーダー | 現場指揮官。実行を促す |
| Junior_D | 1004 | 新人A | 上位者の指示に応答 |
| Junior_E | 1005 | 新人B | 地道な作業を得意 |
| Junior_F | 1006 | 新人C | チーム運営を支える |

### ペルソナの作成・修正

`custom_users.csv` をエディタで編集して、カスタムペルソナを作成できます：

```csv
user_id,name,username,following_agentid_list,previous_tweets,user_char,description
1001,CustomName,custom_username,"[1, 2, 3]","[]","独自の行動ルール。プロンプトに従う。","説明文"
```

**各フィールドの説明**:
- `user_id`: ユーザーID（ユニーク）
- `name`: 表示名
- `username`: Twitterハンドル（@の後の部分）
- `following_agentid_list`: フォロー中のエージェントID
- `previous_tweets`: 過去のツイート（JSON配列）
- `user_char`: LLMへのプロンプト（重要！エージェント性格の定義）
- `description`: プロフィール説明

## 📝 トラブルシューティング

### vLLMサーバーに接続できない

```
❌ エラー: vLLMサーバーに接続できません。
   URL: http://192.168.1.100:8000/v1
```

**確認事項**:
1. Ubuntu PCでvLLMが起動している：`curl http://192.168.1.100:8000/v1/models`
2. ファイアウォールが許可している
3. ホストアドレスが正しい

### メモリ不足エラー

```
RuntimeError: CUDA out of memory
```

**対策**:
- より小さいモデルを使用：`--model mistralai/Mistral-7B`
- バッチサイズを削減：`--num-active 2`

### モデルのダウンロード遅い

初回起動時はモデルのダウンロードに時間がかかります。以下で事前準備可能：

```bash
# Ubuntu PCで実行
python -c "from vllm import LLM; LLM(model='mistralai/Mistral-7B-v0.1')"
```

## 📚 使用するコマンドライン引数

| オプション | デフォルト | 説明 |
|-----------|----------|-----|
| `--vllm-host` | localhost | vLLMサーバーホスト |
| `--vllm-port` | 8000 | vLLMサーバーポート |
| `--model` | mistralai/Mistral-7B-v0.1 | vLLMモデル |
| `--profile-path` | None | ペルソナCSVパス |
| `--steps` | 10 | シミュレーションステップ数 |
| `--num-active` | 6 | 同時アクティブエージェント数 |
| `--output` | ./results_twitter | 出力ディレクトリ |

## 🔍 実行結果の確認

シミュレーション完了後、結果はデータベースに保存されます：

```bash
# 結果ディレクトリを確認
ls -la results_twitter/
```

**生成されるファイル**:
- `twitter_simulation.db` - シミュレーション結果（SQLite）
- ログファイル（オプション）

## 📖 関連ファイル

- `llm_adapter.py` - ローカルLLMアダプター（Ollama、vLLM対応）
- `twitter_vllm_remote.py` - Twitter実行スクリプト（リモートvLLM対応）
- `oasis_local_llm_run.py` - 汎用実行スクリプト（Reddit/Twitter対応）
- `data/custom_users.csv` - ペルソナ定義ファイル

## 💡 応用例

### 複数のシミュレーションを連続実行

```bash
#!/bin/bash
for i in {1..5}; do
    echo "実行 $i"
    python twitter_vllm_remote.py \
      --vllm-host 192.168.1.100 \
      --steps 15 \
      --output ./results_twitter_run_$i
done
```

### 異なるペルソナで複数回実行

```bash
# ペルソナAを作成
cp custom_users.csv custom_users_A.csv
# ... 編集 ...

# ペルソナBを作成
cp custom_users.csv custom_users_B.csv
# ... 編集 ...

# 実行
python twitter_vllm_remote.py --profile-path ../data/custom_users_A.csv
python twitter_vllm_remote.py --profile-path ../data/custom_users_B.csv
```
