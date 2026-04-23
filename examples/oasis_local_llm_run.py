#!/usr/bin/env python3
"""
OASIS Simulation - ローカルLLM実行スクリプト

使用方法:
  python oasis_local_llm_run.py [--backend ollama|vllm] [--model MODEL_NAME]

例:
  python oasis_local_llm_run.py
  python oasis_local_llm_run.py --backend ollama --model neural-chat
  python oasis_local_llm_run.py --backend vllm --model mistralai/Mistral-7B-v0.1
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path

# ローカルアダプターをインポート
try:
    from local_llm_adapter import LocalLLMFactory, CAMELLocalLLMAdapter
except ImportError:
    print("❌ エラー: local_llm_adapter.py が見つかりません")
    print("   同じディレクトリに配置してください")
    sys.exit(1)

# CAMELライブラリのインポート
try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
except ImportError:
    print("❌ エラー: camel-ai がインストールされていません")
    print("   実行: pip install camel-ai==0.2.78")
    sys.exit(1)

# OASISライブラリのインポート
try:
    import oasis
    from oasis import (ActionType, LLMAction, ManualAction,
                       generate_reddit_agent_graph,
                       generate_twitter_agent_graph)
except ImportError:
    print("❌ エラー: oasis がインストールされていません")
    print("   実行: cd ~/OASIS_Simulation && pip install -e .")
    sys.exit(1)


# ============================================================================
# 設定とログ出力
# ============================================================================

class Logger:
    """簡単なログ出力"""
    
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
        print(f"\n{'='*60}\n{title}\n{'='*60}")


# ============================================================================
# ローカルLLMを使用したOASIS実行
# ============================================================================

class OASISLocalLLMRunner:
    """ローカルLLMを使用したOASIS実行管理"""
    
    def __init__(
        self,
        backend: str = "ollama",
        model: str = None,
        platform: str = "reddit",
        num_agents: int = 5,
        num_active_agents: int = 2,
        output_dir: str = "./results_local",
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        profile_path: str = None,
        num_steps: int = 10
    ):
        self.backend = backend.lower()
        self.model = model
        self.platform = platform.lower()
        self.num_agents = num_agents
        self.num_active_agents = min(num_active_agents, num_agents)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.profile_path = profile_path
        self.num_steps = num_steps
        
        self.llm = None
        self.env = None
    
    def _init_llm(self):
        """LLMの初期化"""
        Logger.section("ローカルLLM初期化")
        
        try:
            Logger.info(f"バックエンド: {self.backend.upper()}")
            if self.model:
                Logger.info(f"モデル: {self.model}")
            
            if self.backend == "vllm":
                base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
                Logger.info(f"vLLMサーバー: {base_url}")
                self.llm = LocalLLMFactory.create(
                    backend=self.backend,
                    model=self.model,
                    base_url=base_url
                )
            else:
                self.llm = LocalLLMFactory.create(
                    backend=self.backend,
                    model=self.model
                )
            Logger.success("LLM初期化完了")
            return True
        
        except ConnectionError as e:
            Logger.error(str(e))
            return False
        except Exception as e:
            Logger.error(f"予期しないエラー: {e}")
            return False
    
    async def _create_agent_graph(self):
        """エージェントグラフの作成"""
        Logger.section("エージェントグラフ作成")
        
        available_actions = [
            ActionType.LIKE_POST,
            ActionType.DISLIKE_POST,
            ActionType.CREATE_POST,
            ActionType.CREATE_COMMENT,
            ActionType.LIKE_COMMENT,
            ActionType.DISLIKE_COMMENT,
            ActionType.SEARCH_POSTS,
            ActionType.SEARCH_USER,
            ActionType.TREND,
            ActionType.REFRESH,
            ActionType.DO_NOTHING,
            ActionType.FOLLOW,
            ActionType.MUTE,
        ]
        
        # Twitter用の限定的なアクション
        if self.platform == "twitter":
            available_actions = [
                ActionType.CREATE_POST,
                ActionType.QUOTE_POST,
                ActionType.DO_NOTHING,
            ]
        
        Logger.info(f"プラットフォーム: {self.platform.upper()}")
        Logger.info(f"エージェント数: {self.num_agents}")
        if self.profile_path:
            Logger.info(f"ペルソナCSV: {self.profile_path}")
        
        try:
            if self.platform == "reddit":
                agent_graph = await generate_reddit_agent_graph(
                    profile_path=self.profile_path,
                    model=self.llm,
                    available_actions=available_actions,
                    num_agents=self.num_agents
                )
            elif self.platform == "twitter":
                agent_graph = await generate_twitter_agent_graph(
                    profile_path=self.profile_path,
                    model=self.llm,
                    available_actions=available_actions,
                    num_agents=self.num_agents
                )
            else:
                raise ValueError(f"未知のプラットフォーム: {self.platform}")
            
            Logger.success(f"{agent_graph.num_agents} 個のエージェントを作成")
            return agent_graph
        
        except Exception as e:
            Logger.error(f"エージェント作成エラー: {e}")
            raise
    
    async def _create_environment(self, agent_graph):
        """環境の作成"""
        Logger.section("環境初期化")
        
        db_path = self.output_dir / f"{self.platform}_simulation.db"
        
        # 既存DBを削除
        if db_path.exists():
            os.remove(db_path)
            Logger.info(f"既存DB削除: {db_path}")
        
        Logger.info(f"DB保存先: {db_path}")
        
        try:
            platform_map = {
                "reddit": oasis.DefaultPlatformType.REDDIT,
                "twitter": oasis.DefaultPlatformType.TWITTER,
            }
            
            self.env = oasis.make(
                agent_graph=agent_graph,
                platform=platform_map[self.platform],
                database_path=str(db_path),
            )
            
            await self.env.reset()
            Logger.success("環境初期化完了")
            return True
        
        except Exception as e:
            Logger.error(f"環境初期化エラー: {e}")
            raise
    
    async def _run_simulation(self):
        """シミュレーション実行"""
        Logger.section("シミュレーション実行")
        
        try:
            # ステップ1: 手動アクション
            Logger.info("ステップ1: 初期投稿を作成")
            
            actions_1 = {}
            agent_0 = self.env.agent_graph.get_agent(0)
            
            if self.platform == "twitter":
                initial_content = "新しいプロジェクト管理システムが稼働しました。皆さん、まずは自分の担当タスクや現在の状況を共有してください。"
            else:
                initial_content = f"Hello from local {self.backend} LLM! #{self.backend}"
            
            actions_1[agent_0] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": initial_content}
            )
            
            await self.env.step(actions_1)
            Logger.success("ステップ1完了")
            
            # メインシミュレーションループ
            Logger.info(f"ステップ2-{self.num_steps}: LLMエージェント推論を {self.num_steps-1} ステップ実行")
            Logger.info("⏳ この処理には時間がかかる可能性があります...")
            
            for step in range(2, self.num_steps + 1):
                actions = {
                    agent: LLMAction()
                    for idx, agent in self.env.agent_graph.get_agents(
                        list(range(min(self.num_active_agents, self.env.agent_graph.num_agents)))
                    )
                }
                
                await self.env.step(actions)
                Logger.success(f"ステップ{step}完了 ({len(actions)} エージェント)")
            
            return True
        
        except Exception as e:
            Logger.error(f"シミュレーション実行エラー: {e}")
            raise
    
    async def run(self):
        """全体実行"""
        Logger.section(f"OASIS Local LLM シミュレーション開始")
        Logger.info(f"バックエンド: {self.backend}")
        Logger.info(f"プラットフォーム: {self.platform}")
        Logger.info(f"出力ディレクトリ: {self.output_dir}")
        
        try:
            # ステップ1: LLM初期化
            if not self._init_llm():
                return False
            
            # ステップ2: エージェント作成
            agent_graph = await self._create_agent_graph()
            
            # ステップ3: 環境作成
            await self._create_environment(agent_graph)
            
            # ステップ4: シミュレーション実行
            await self._run_simulation()
            
            # ステップ5: クローズ
            if self.env:
                await self.env.close()
                Logger.success("環境をクローズ")
            
            Logger.section("✅ シミュレーション完了!")
            return True
        
        except KeyboardInterrupt:
            Logger.warning("ユーザーによって中断されました")
            if self.env:
                await self.env.close()
            return False
        
        except Exception as e:
            Logger.error(f"予期しないエラー: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# メイン処理
# ============================================================================

def parse_arguments():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="OASIS Simulation - ローカルLLM実行スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python oasis_local_llm_run.py --backend vllm --platform twitter
  python oasis_local_llm_run.py --backend vllm --platform twitter --vllm-host 192.168.1.100
  python oasis_local_llm_run.py --backend vllm --platform twitter --profile-path ../data/custom_users.csv
  python oasis_local_llm_run.py --backend ollama --platform reddit --num-agents 10 --steps 20
        """
    )
    
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help="LLMバックエンド (デフォルト: ollama)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="モデル名 (例: mistral, neural-chat)"
    )
    
    parser.add_argument(
        "--platform",
        choices=["reddit", "twitter"],
        default="reddit",
        help="シミュレーションプラットフォーム (デフォルト: reddit)"
    )
    
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="エージェント総数 (デフォルト: 5)"
    )
    
    parser.add_argument(
        "--num-active",
        type=int,
        default=2,
        help="アクティブエージェント数 (デフォルト: 2)"
    )
    
    parser.add_argument(
        "--vllm-host",
        type=str,
        default="localhost",
        help="vLLMサーバーホスト (デフォルト: localhost)"
    )
    
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="vLLMサーバーポート (デフォルト: 8000)"
    )
    
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help="ペルソナCSVパス (例: ../data/custom_users.csv)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="シミュレーション実行ステップ数 (デフォルト: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./results_local",
        help="出力ディレクトリ (デフォルト: ./results_local)"
    )
    
    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()
    
    # ローカルLLMランナーの作成と実行
    runner = OASISLocalLLMRunner(
        backend=args.backend,
        model=args.model,
        platform=args.platform,
        num_agents=args.num_agents,
        num_active_agents=args.num_active,
        output_dir=args.output,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        profile_path=args.profile_path,
        num_steps=args.steps
    )
    
    # 非同期実行
    success = asyncio.run(runner.run())
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  中断されました")
        sys.exit(130)
    except Exception as e:
        Logger.error(f"致命的エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
