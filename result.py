import sqlite3
import pandas as pd
import os
import json
from datetime import datetime

def main():
    # データベースの保存場所
    db_path = "./data/twitter_simulation.db"
    
    # 保存先フォルダの設定
    results_dir = "./results"
    
    # フォルダが存在しない場合は作成する
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # ファイル名の生成（現在の日時を使用：例 timeline_20260331_153000.txt）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"timeline_{current_time}.txt")

    # 出力する文章を溜め込むための一覧
    output_lines = []

    try:
        conn = sqlite3.connect(db_path)
        
        # 投稿、返信、そして全行動履歴のデータを読み込む
        df_posts = pd.read_sql_query("SELECT * FROM post", conn)
        df_comments = pd.read_sql_query("SELECT * FROM comment", conn)
        df_traces = pd.read_sql_query("SELECT * FROM trace", conn)

        # ==========================================
        # 1. タイムライン表示（省略なしの全文表示）
        # ==========================================
        output_lines.append("\n" + "━"*50)
        output_lines.append(" 🌟 タイムライン (シミュレーション結果)")
        output_lines.append("━"*50 + "\n")

        if df_posts.empty:
            output_lines.append("📭 投稿はまだありません。\n")
        else:
            for _, post_row in df_posts.iterrows():
                post_id = post_row.get('post_id', '?')
                user_id = post_row.get('user_id', '?')
                
                # 省略せずに全文を取得し、改行を維持して字下げする
                content = str(post_row.get('content', ''))
                formatted_content = "\n".join([f"      {line}" for line in content.split('\n')])

                output_lines.append(f"👤 ユーザーID({user_id}) [投稿ID: {post_id}]")
                output_lines.append(f"💬 投稿内容:\n{formatted_content}")

                if not df_comments.empty:
                    replies = df_comments[df_comments['post_id'] == post_id]
                    num_replies = len(replies)

                    if num_replies > 0:
                        output_lines.append(" ┃")
                        for i, (_, reply_row) in enumerate(replies.iterrows()):
                            r_user_id = reply_row.get('user_id', '?')
                            
                            r_content = str(reply_row.get('content', ''))
                            formatted_r_content = "\n".join([f"          {line}" for line in r_content.split('\n')])

                            is_last = (i == num_replies - 1)
                            prefix = " ┗━" if is_last else " ┣━"
                            
                            output_lines.append(f"{prefix} 👤 ユーザーID({r_user_id}) の返信:")
                            output_lines.append(f"{formatted_r_content}")
                            
                            if not is_last:
                                output_lines.append(" ┃")
                output_lines.append("")

        # ==========================================
        # 2. エージェントの行動履歴表示
        # ==========================================
        output_lines.append("\n" + "━"*50)
        output_lines.append(" 🕵️‍♂️ エージェントの行動履歴")
        output_lines.append("━"*50 + "\n")

        if df_traces.empty:
            output_lines.append("📭 行動履歴はありません。\n")
        else:
            for _, trace_row in df_traces.iterrows():
                user_id = trace_row.get('user_id', '?')
                action = trace_row.get('action', '?')
                info_str = trace_row.get('info', '{}')
                created_at = trace_row.get('created_at', '?')
                
                output_lines.append(f"🕒 [{created_at}] 👤 Agent {user_id} | 行動: {action}")
                
                # 【修正箇所】日本語の文字化け（Unicodeエスケープ）を直し、綺麗に改行して表示する
                try:
                    # 文字列を一度データとして読み込み、日本語を維持したまま見やすい形に変換
                    info_dict = json.loads(info_str)
                    formatted_info = json.dumps(info_dict, ensure_ascii=False, indent=2)
                    
                    output_lines.append("   └ 詳細:")
                    for line in formatted_info.split('\n'):
                        # なかにある改行コード（\n）も実際の改行として表示する処理を追加
                        sub_lines = line.replace('\\n', '\n').split('\n')
                        for sub_line in sub_lines:
                            output_lines.append(f"       {sub_line}")
                except Exception:
                    # 変換に失敗した場合はそのまま表示
                    output_lines.append(f"   └ 詳細: {info_str}")
            
            output_lines.append("")

    except Exception as e:
        output_lines.append(f"データの読み込み中に問題が発生しました: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    # 一覧に溜め込んだ文章を一つに結合
    final_output = "\n".join(output_lines)
    
    # 画面に表示
    print(final_output)

    # 保存先ファイルに書き込み
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"✅ 結果をファイルに保存しました: {output_file}")
    except Exception as e:
        print(f"ファイルの保存中に問題が発生しました: {e}")

if __name__ == "__main__":
    main()