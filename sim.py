import asyncio
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import oasis
from oasis import ActionType, LLMAction, generate_twitter_agent_graph

async def main():
    # OpenRouter経由で大きなモデルと小さなモデルを定義
    large_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type="anthropic/claude-opus-4.6", # 任意の大きなモデル
    )
    small_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type="anthropic/claude-haiku-4.5", # 任意の小さなモデル
    )
    
    # 6人のエージェント（Manager_A〜Leader_Cがlarge、Junior_D〜Fがsmall）にモデルを割り当てる
    models = [large_model, large_model, large_model, small_model, small_model, small_model]

    # Twitter環境で利用可能な行動を定義
    available_actions = [
        ActionType.CREATE_POST, # 新規投稿
        ActionType.QUOTE_POST,  # 引用投稿（返信として機能）
        ActionType.DO_NOTHING,  # 何もしない
    ]

    # ！！ここで先ほど作成した custom_users.csv を読み込みます ！！
    agent_graph = await generate_twitter_agent_graph(
        profile_path="./data/custom_users.csv",
        model=models,
        available_actions=available_actions,
    )

    # データベースの保存先指定
    db_path = "./data/twitter_simulation.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # シミュレーション環境の構築
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    await env.reset()

    # まずは動作確認のため、短め（3ステップ）でLLMエージェントに自律行動させます
    print("シミュレーションを開始します...")
    for step in range(3): 
        print(f"ステップ {step+1} 実行中...")
        actions = {agent: LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)

    await env.close()
    print("シミュレーションが完了しました。")

if __name__ == "__main__":
    asyncio.run(main())