import sqlite3
import pandas as pd

def main():
    # データベースの保存場所
    db_path = "./data/twitter_simulation.db"
    conn = sqlite3.connect(db_path)

    try:
        # 投稿(post)と返信(comment)のデータを読み込む
        df_posts = pd.read_sql_query("SELECT * FROM post", conn)
        df_comments = pd.read_sql_query("SELECT * FROM comment", conn)

        print("=== 【LLMエージェントたちの投稿一覧】 ===")
        if not df_posts.empty:
            for _, row in df_posts.iterrows():
                post_id = row.get('post_id', '?')
                user_id = row.get('user_id', '?')
                content = row.get('content', '')
                print(f"[投稿ID: {post_id}] ユーザーID({user_id})の投稿:")
                print(f"  {content}\n")
        else:
            print("投稿はまだありません。\n")

        print("=== 【LLMエージェントたちの返信一覧】 ===")
        if not df_comments.empty:
            for _, row in df_comments.iterrows():
                comment_id = row.get('comment_id', '?')
                user_id = row.get('user_id', '?')
                post_id = row.get('post_id', '?') # どの投稿に対する返信か
                content = row.get('content', '')
                print(f"[投稿ID: {post_id} への返信] ユーザーID({user_id})の返信:")
                print(f"  {content}\n")
        else:
            print("返信はまだありません。\n")

    except Exception as e:
        print(f"データの読み込み中に問題が発生しました: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()