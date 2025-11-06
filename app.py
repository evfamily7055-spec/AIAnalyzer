import streamlit as st
import pandas as pd
import io
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- 定数 (Constants) ---
# AIにユニーク値を渡すカテゴリ列の上限
MAX_UNIQUE_VALUES_FOR_SCHEMA = 20

# --- ページ設定 (Page Config) ---
st.set_page_config(layout="wide")
st.title("AIコード生成アナリスト 👨‍💻 (プライバシー保護 & 高推論)")
st.info("実データをAIに送信せず、拡張スキーマ（列の構造やカテゴリ情報）と日本語の指示だけでPandasコードを自動生成し、実行します。")

# --- セッションステートの初期化 (Initialize Session State) ---
# 実データをここに保持（AIには送らない）
if 'df' not in st.session_state:
    st.session_state.df = None 
# (NEW) 拡張スキーマ（ユニーク値などを含む辞書）
if 'schema_dict' not in st.session_state:
    st.session_state.schema_dict = None 
# AIが生成したコード
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = "" 
# コードの実行結果
if 'exec_result' not in st.session_state:
    st.session_state.exec_result = None 

# --- (NEW) Gemini API 呼び出し関数 (Python) ---
@st.cache_data(ttl=600) # 10分間は同じ指示ならキャッシュを使う
def generate_code_from_ai(schema_json: str, user_prompt: str, api_key: str):
    """
    拡張スキーマとユーザーの指示をGemini APIに送信し、Pandasコードを生成する。
    """
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"APIキーの設定に失敗しました: {e}")
        return None

    # (NEW) AIへの指示（システムプロンプト）を強化
    system_prompt = (
        "あなたは、Pandasを専門とする世界クラスのPythonデータアナリストです。"
        "あなたの仕事は、渡された「データスキーマ」と「ユーザーの曖昧な指示」から、実行可能なPandasコードを推論して生成することです。"
        
        "## ルール:"
        "1. 入力データフレームは *常に* `df` という名前です。"
        "2. ユーザーの指示（例: '男女別'）がスキーマの列名（例: 'Gender'）と完全一致しなくても、スキーマの `unique_values` や `min`/`max` 情報を参照し、ユーザーがどの列について話しているかを *積極的に推論* してください。"
        "3. 生成するコードは、Pythonコードの *スニペットのみ* とし、説明、インポート、マークダウン（```python など）を *絶対に* 含めないでください。"
        "4. コードの *最終行* は、集計結果を保持する変数 `result` で *必ず* 終わるようにしてください。（例: `result = df.groupby('Gender')['Age'].mean()`）"
        "5. `print()` 文は使わないでください。"
    )

    # (NEW) AIへの入力を構成
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025", # 高速なモデルを使用
        system_instruction=system_prompt
    )
    
    # スキーマとユーザー指示を結合してプロンプトを作成
    full_prompt = (
        f"## データスキーマ (JSON形式):\n{schema_json}\n\n"
        f"## ユーザーの指示:\n{user_prompt}"
    )

    try:
        response = model.generate_content(full_prompt)
        # (NEW) テキスト部分のみを抽出し、マークダウンの```を削除
        code = response.text.strip().replace("```python", "").replace("```", "").strip()
        return code
    
    except google_exceptions.InvalidArgument as e:
        st.error(f"APIキーが無効、または設定が正しくありません: {e}")
        return None
    except Exception as e:
        st.error(f"Gemini API 呼び出し中に予期せぬエラーが発生しました: {e}")
        return None


# --- (NEW) サイドバーでAPIキーを入力 ---
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini API Key", type="password", help="Gemini APIキーをここに入力してください。")
    st.markdown("---")
    st.info("このアプリは実データをAIに送信しません。AIには列名とカテゴリのユニーク値（20種類以下）のみが送信されます。")

# --- 1. ファイルアップローダー ---
uploaded_file = st.file_uploader("Excelファイル (.xlsx) をアップロードしてください", type=["xlsx"])

if uploaded_file:
    try:
        bytes_data = uploaded_file.getvalue()
        df = pd.read_excel(io.BytesIO(bytes_data))
        
        # (重要) 実データをセッションステートに保存
        st.session_state.df = df 
        
        # --- (NEW) 拡張スキーマの生成 ---
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema[col] = {"dtype": dtype}
            
            # (重要) object型でユニーク値が少ない場合、ユニーク値を取得
            if dtype == 'object' and df[col].nunique() <= MAX_UNIQUE_VALUES_FOR_SCHEMA:
                unique_vals = df[col].dropna().unique().tolist()
                schema[col]["unique_values"] = unique_vals
            # (おまけ) 数値列の統計情報も追加
            elif pd.api.types.is_numeric_dtype(df[col]):
                 try:
                     schema[col]["mean"] = df[col].mean()
                     schema[col]["min"] = df[col].min()
                     schema[col]["max"] = df[col].max()
                 except Exception:
                     pass # 統計が計算できない場合はスキップ
                 
        st.session_state.schema_dict = schema
        # --- 拡張スキーマ生成ここまで ---
        
        st.success("ファイルの読み込みが完了しました。")
        
        # 以前の実行結果をリセット
        st.session_state.generated_code = ""
        st.session_state.exec_result = None

    except Exception as e:
        st.error(f"Excelファイルの読み込みに失敗しました: {e}")
        st.session_state.df = None


# --- 2. スキーマ確認 & AIへの指示 ---
if st.session_state.df is not None:
    st.markdown("---")
    st.header("Step 1: データスキーマの確認")
    st.write("AIには以下の拡張スキーマ（列名、型、カテゴリのユニーク値）のみが送信されます。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("データの先頭5行 (プレビュー用)")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
    with col2:
        st.subheader("拡張スキーマ (AIに送信する情報)")
        # (NEW) 辞書をJSON文字列にして見やすく表示
        st.json(st.session_state.schema_dict, expanded=True)

    st.markdown("---")
    st.header("Step 2: AIへの集計指示")
    
    user_prompt = st.text_area(
        "実行したい集計内容を日本語で指示してください。",
        placeholder="例: 「男女別の年齢の平均値」\n例: 「満足度（高・中・低）の人数をカウント」\n例: 「'所属' 列に '営業部' が含まれる行だけを抽出」",
        height=100
    )

    if st.button("🤖 AIコード生成", type="primary"):
        if not api_key:
            st.error("サイドバーからGemini APIキーを入力してください。")
        elif not user_prompt:
            st.warning("指示を入力してください。")
        else:
            with st.spinner("AIがPandasコードを生成中です..."):
                # (NEW) 拡張スキーマの辞書をJSON文字列に変換
                schema_json = json.dumps(st.session_state.schema_dict, indent=2, ensure_ascii=False)
                
                # (NEW) Python関数でAPI呼び出し
                generated_code = generate_code_from_ai(schema_json, user_prompt, api_key)
                
                if generated_code:
                    st.session_state.generated_code = generated_code
                    st.session_state.exec_result = None # 実行結果をリセット
                    st.success("コードが生成されました。Step 3で確認・実行してください。")

    # --- 3. 生成されたコードの確認 & 実行 ---
    if st.session_state.generated_code:
        st.markdown("---")
        st.header("Step 3: AIが生成したコードの確認と実行")
        
        st.subheader("生成されたPandasコード")
        st.code(st.session_state.generated_code, language="python")
        
        st.warning("AIが生成したコードが意図通りか（例: データの削除などを含まないか）確認してから実行してください。")
        
        if st.button("▶️ このコードを実行する"):
            with st.spinner("サーバー上でコードを実行中..."):
                try:
                    # (重要)
                    # サーバー上で、実データ(df)に対してコード(generated_code)を実行
                    
                    # 実行環境を準備
                    global_vars = {"pd": pd}
                    # 実データを 'df' という名前で渡す
                    local_vars = {"df": st.session_state.df.copy()} 
                    
                    # execでコードを実行
                    exec(st.session_state.generated_code, global_vars, local_vars)
                    
                    # 結果を取得 (AIプロンプトで 'result' に代入するよう指示済み)
                    result = local_vars.get("result", None)
                    
                    if result is not None:
                        st.session_state.exec_result = result
                    else:
                        st.error("コードは実行されましたが、'result' 変数に結果が見つかりませんでした。AIの生成コードが `result = ...` で終わっているか確認してください。")
                        
                except Exception as e:
                    st.error(f"コードの実行に失敗しました: {e}")

    # --- 4. 実行結果の表示 ---
    if st.session_state.exec_result is not None:
        st.markdown("---")
        st.header("Step 4: 集計・分析結果")
        
        result = st.session_state.exec_result
        
        # 結果がDataFrameまたはSeriesの場合、データフレームとして表示
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            st.dataframe(result, use_container_width=True)
            
            # CSVダウンロード機能
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                # DataFrameでない場合(Seriesなど)はDataFrameに変換
                if not isinstance(df_to_convert, pd.DataFrame):
                    df_to_convert = df_to_convert.to_frame()
                return df_to_convert.to_csv(index=True).encode('utf-8-sig')
            
            try:
                csv_data = convert_df_to_csv(result)
                st.download_button(label="結果をCSVダウンロード", data=csv_data, file_name="analysis_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"CSV変換に失敗しました: {e}")
            
        # それ以外（数値、文字列、リストなど）の場合
        else:
            st.write("実行結果:")
            st.write(result)
