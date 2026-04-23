#!/usr/bin/env python3
"""
Twitter SNS シミュレーション - vLLM版（Ubuntu リモート実行対応）

使用方法:
  python twitter_vllm_remote.py --vllm-host 192.168.1.100
  python twitter_vllm_remote.py --vllm-host 192.168.1.100 --model mistralai/Mistral-7B-v0.1
  python twitter_vllm_remote.py --vllm-host 192.168.1.100 --profile-path ../data/custom_users.csv

前提条件:
  - vLLMサーバーがUbuntu PCで起動している
  - vLLMのポート: 8000（カスタマイズ可能）
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path

# ローカルアダプターをインポート
try:
    from llm_adapter import LocalLLMFactory
except ImportError:
    print("❌ エラー: llm_adapter.py が見つかりません")
    print("   同じディレクトリに配置してください")
    sys.exit(1)

# OASISライブラリのインポート
try:
    import oasis
    from oasis import (ActionType, LLMAction, ManualAction,
                       generate_twitter_agent_graph)
except ImportError:
    print("❌ エラー: oasis がインストールされていません")
    print("   実行: pip install -e ../")
    sys.exit(1)


class Logger:
    """ログ出力ユーティリティ"""
    
    @staticmethod
    def info(msg: str):
        print(f"ℹ️  {msg}")
    
    @staticmethod
    def success(msg: str):
        print(f"✓ {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"⚠️  {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"❌ {msg}")
    
    @staticmethod
    def section(title: str):
        print(f"\n{'='*70}\n{title}\n{'='*70}")


async def main():
    """vLLMを使ったTwitterシミュレーション"""
    parser = argparse.ArgumentParser(
        description="Twitter LLMシミュレーション（vLLM版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # ローカルホストのvLLM
  python twitter_vllm_remote.py
  
  # リモートホスト指定
  python twitter_vllm_remote.py --vllm-host 192.168.1.100
  
  # モデル指定
  python twitter_vllm_remote.py --vllm-host 192.168.1.100 --model meta-llama/Llama-2-7b
  
  # カスタムペルソナ使用
  python twitter_vllm_remote.py --profile-path ../data/custom_users.csv
  
  # 長時間シミュレーション
  python twitter_vllm_remote.py --vllm-host 192.168.1.100 --steps 20
        """
    )
    
    parser.add_argument("--vllm-host", type=str, default="localhost", help="vLLMサーバーホスト")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLMサーバーポート")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="vLLMモデル")
    parser.add_argument("--profile-path", type=str, default=None, help="ペルソナCSVパス")
    parser.add_argument("--steps", type=int, default=10, help="シミュレーション実行ステップ数")
    parser.add_argument("--num-active", type=int, default=6, help="同時アクティブエージェント数")
    parser.add_argument("--output", type=str, default="./results_twitter", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    Logger.section("🐦 Twitter LLMシミュレーション開始")
    Logger.info(f"vLLMサーバー: {args.vllm_host}:{args.vllm_port}")
    Logger.info(f"モデル: {args.model}")
    Logger.info(f"シミュレーション ステップ数: {args.steps}")
    Logger.info(f"出力ディレクトリ: {output_dir}")
    
    # vLLM初期化
    Logger.section("vLLM初期化")
    base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
    Logger.info(f"vLLMサーバー: {base_url}")
    
    try:
        llm = LocalLLMFactory.create(
            backend="vllm",
            model=args.model,
            base_url=base_url
        )
        Logger.success("vLLM接続成功")
    except ConnectionError as e:
        Logger.error(str(e))
        return False
    
    # エージェント作成
    Logger.section("Twitterエージェント作成")
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.QUOTE_POST,
        ActionType.DO_NOTHING,
    ]
    
    if args.profile_path:
        Logger.info(f"ペルソナCSV: {args.profile_path}")
    else:
        Logger.info("ペルソナ: ランダム生成")
    
    try:
        agent_graph = await generate_twitter_agent_graph(
            profile_path=args.profile_path,
            model=llm,
            available_actions=available_actions,
        )
        Logger.success(f"{agent_graph.num_agents} 個のエージェントを作成")
    except Exception as e:
        Logger.error(f"エージェント作成エラー: {e}")
        return False
    
    # 環境初期化
    Logger.section("環境初期化")
    db_path = output_dir / "twitter_simulation.db"
    
    if db_path.exists():
        os.remove(db_path)
        Logger.info(f"既存DB削除: {db_path}")
    
    try:
        env = oasis.make(
            agent_graph=agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=str(db_path),
        )
        await env.reset()
        Logger.success("環境初期化完了")
    except Exception as e:
        Logger.error(f"環境初期化エラー: {e}")
        return False
    
    # シミュレーション実行
    Logger.section("シミュレーション実行")
    
    try:
        # ステップ1: 初期投稿
        Logger.info("ステップ1: 初期投稿を作成...")
        agent_0 = env.agent_graph.get_agent(0)
        initial_action = {
            agent_0: ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": "新しいプロジェクト管理システムが稼働しました。皆さん、まずは自分の担当タスクや現在の状況を共有してください。"}
            )
        }
        await env.step(initial_action)
        Logger.success("ステップ1完了")
        
        # メインシミュレーション
        Logger.info(f"ステップ2-{args.steps}: LLM推論を {args.steps-1} ステップ実行")
        Logger.info("⏳ 処理中（時間がかかります）...")
        
        for step in range(2, args.steps + 1):
            actions = {
                agent: LLMAction()
                for idx, agent in env.agent_graph.get_agents(
                    list(range(min(args.num_active, env.agent_graph.num_agents)))
                )
            }
            await env.step(actions)
            Logger.success(f"ステップ{step}完了 ({len(actions)} エージェント)")
        
        # 終了
        await env.close()
        Logger.success("環境をクローズ")
        Logger.section("✅ シミュレーション完了!")
        Logger.info(f"結果は {db_path} に保存されました")
        return True
    
    except Exception as e:
        Logger.error(f"シミュレーション実行エラー: {e}")
        await env.close()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
