import streamlit as st
import pandas as pd
import io
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# グラフ描画ライブラリ
import plotly.express as px
import plotly.graph_objects as go

# (NEW) 日本語テキストマイニング（形態素解析）ライブラリ
try:
    from janome.tokenizer import Tokenizer
    from janome.tokenfilter import POSKeepFilter, TokenCountFilter
    from janome.analyzer import Analyzer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

# --- 定数 (Constants) ---
MAX_UNIQUE_VALUES_FOR_SCHEMA = 20

# --- ページ設定 (Page Config) ---
st.set_page_config(layout="wide")
st.title("AIデータアナリスト (NLP・解説対応) 🔬")
st.info("集計・可視化・テキストマイニングを実行し、論文用の「分析内容の解説」もAIが自動生成します。")

# --- セッションステートの初期化 (Initialize Session State) ---
if 'df' not in st.session_state:
    st.session_state.df = None 
if 'schema_dict' not in st.session_state:
    st.session_state.schema_dict = None 
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = "" 
if 'exec_output' not in st.session_state:
    st.session_state.exec_output = None 
# (NEW) 分析内容の説明文を保存
if 'analysis_explanation' not in st.session_state:
    st.session_state.analysis_explanation = "" 

# --- Gemini API 呼び出し関数 ---
@st.cache_data(ttl=600) 
def generate_code_and_explanation(schema_json: str, user_prompt: str, api_key: str):
    """
    (NEW) 拡張スキーマと指示をGeminiに送信し、
    「コード」と「分析内容の日本語説明」を含むJSONを生成する。
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"APIキーの設定に失敗しました: {e}")
        return None

    # (NEW) AIへの指示（システムプロンプト）をJSON出力・NLP対応に超強化
    system_prompt = (
        "あなたは、Pandas, Plotly (px), Janome (日本語形態素解析) を専門とする世界クラスのPythonデータアナリストです。"
        "あなたの仕事は、渡された「データスキーマ」と「ユーザーの曖昧な指示」から、「実行コード」と「そのコードの分析内容の日本語説明」の2つを *JSON形式* で生成することです。"
        
        "## ルール:"
        "1. 出力は *必ず* 以下のJSON形式の *文字列のみ* としてください。説明やマークダウン（```json など）は絶対に含めないでください。"
        "   {\n"
        "     \"code_to_execute\": \"... (ここにPythonコードスニペットを記述) ...\",\n"
        "     \"analysis_explanation\": \"... (ここに分析内容の日本語説明を記述) ...\"\n"
        "   }\n"
        
        "2. `code_to_execute` のルール:"
        "   - 入力データフレームは *常に* `df` という名前です。"
        "   - ユーザーの指示から、データ集計 (Pandas), グラフ描画 (Plotly Express as px), テキストマイニング (Janome) のどれが最適か *推論* してください。"
        "   - コードの *最終行* は、集計結果（DataFrame, Series, Plotly Figure）を `output` という単一の変数に *必ず* 代入してください。"
        "   - （例: `output = df.groupby('Gender')['Age'].mean()`）"
        "   - （例: `output = px.scatter(df, x='Age', y='Income')`）"
        "   - `print()` や `fig.show()` 文は *絶対に* 使わないでください。"
        
        "3. `analysis_explanation` のルール:"
        "   - `code_to_execute` で実行する分析が *何をしているか* を、学術論文の「方法」セクションで使えるような、客観的かつ簡潔な日本語で説明してください。"
        "   - （例: 「'Gender' 列をグループ化キーとし、'Age' 列の平均値を算出した。」）"
        "   - （例: 「'Age' 列をX軸、'Income' 列をY軸とする散布図を作成し、両変数の関係性を可視化した。」）"

        "4. (NEW) 日本語テキストマイニングの指示（例: 「自由回答を分析」「単語頻度」）の場合:"
        "   - スキーマから `object` 型でユニーク値が多いテキスト列を推論してください。"
        "   - `janome.tokenizer.Tokenizer` を使って形態素解析を行ってください。"
        "   - 分析対象は *名詞*, *動詞*, *形容詞* の *原形* としてください（`token.base_form` と `token.part_of_speech.startswith` を使用）。"
        "   - `stop_words` (例: 'する', 'ある', 'ない', 'こと', 'もの') を定義し、除外してください。"
        "   - 単語頻度をカウントし、上位50件を `pd.DataFrame(..., columns=['word', 'count'])` に格納してください。"
        "   - 最後に `px.treemap` を使用し、`path=[px.Constant('all'), 'word']`, `values='count'` で結果をツリーマップとして可視化し、それを `output` に代入してください。"
        
        "5. 曖昧な指示（例: '男女別'）は、スキーマの `unique_values` などを参照し、*積極的に推論* してください。"
    )


    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        system_instruction=system_prompt,
        # (NEW) 出力をJSON形式に強制
        generation_config={"response_mime_type": "application/json"}
    )
    
    full_prompt = (
        f"## データスキーマ (JSON形式):\n{schema_json}\n\n"
        f"## ユーザーの指示:\n{user_prompt}"
    )

    try:
        response = model.generate_content(full_prompt)
        # (NEW) JSON文字列をパース
        response_json = response.text.strip()
        response_data = json.loads(response_json)
        return response_data
    except json.JSONDecodeError as e:
        st.error(f"AIがJSON形式でない応答を返しました。AIの応答: {response.text}\nエラー: {e}")
        return None
    except google_exceptions.InvalidArgument as e:
        st.error(f"APIキーが無効、または設定が正しくありません: {e}")
        return None
    except Exception as e:
        st.error(f"Gemini API 呼び出し中に予期せぬエラーが発生しました: {e}")
        return None

# --- サイドバー (APIキー入力) ---
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini API Key", type="password", help="Gemini APIキーをここに入力してください。")
    st.markdown("---")
    st.info("このアプリは実データをAIに送信しません。AIには列名とカテゴリのユニーク値（20種類以下）のみが送信されます。")
    if not JANOME_AVAILABLE:
        st.error("Janomeライブラリが見つかりません。テキストマイニング機能は無効です。\n`pip install janome` を実行してください。")


# --- 1. ファイルアップローダー ---
uploaded_file = st.file_uploader("Excelファイル (.xlsx) をアップロードしてください", type=["xlsx"])

if uploaded_file:
    try:
        bytes_data = uploaded_file.getvalue()
        df = pd.read_excel(io.BytesIO(bytes_data))
        
        st.session_state.df = df 
        
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema[col] = {"dtype": dtype}
            
            if dtype == 'object' and df[col].nunique() <= MAX_UNIQUE_VALUES_FOR_SCHEMA:
                unique_vals = df[col].dropna().unique().tolist()
                schema[col]["unique_values"] = unique_vals
            elif pd.api.types.is_numeric_dtype(df[col]):
                 try:
                     schema[col]["mean"] = df[col].mean()
                     schema[col]["min"] = df[col].min()
                     schema[col]["max"] = df[col].max()
                 except Exception:
                     pass 
                 
        st.session_state.schema_dict = schema
        
        st.success("ファイルの読み込みが完了しました。")
        
        st.session_state.generated_code = ""
        st.session_state.exec_output = None 
        st.session_state.analysis_explanation = ""

    except Exception as e:
        st.error(f"Excelファイルの読み込みに失敗しました: {e}")
        st.session_state.df = None


# --- 2. メインの作業領域 (左右分割) ---
if st.session_state.df is not None:
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    # --- 左カラム (col1): AIへの指示と実行（作業領域） ---
    with col1:
        st.header("Step 1: AIへの指示")
        st.write("右側でデータ（列名）を確認しながら、実行したい内容を日本語で指示してください。")
        
        user_prompt = st.text_area(
            "指示入力欄:",
            placeholder=(
                "（集計例）: 「男女別の年齢の平均値」\n"
                "（可視化例）: 「'年齢' と '給与' の散布図を表示」\n"
                "（NLP例）: 「'自由回答' 列の単語頻度を可視化」"
            ),
            height=150
        )

        if st.button("🤖 AIコード生成", type="primary"):
            if not api_key:
                st.error("サイドバーからGemini APIキーを入力してください。")
            elif not user_prompt:
                st.warning("指示を入力してください。")
            elif "テキストマイニング" in user_prompt or "単語" in user_prompt:
                 if not JANOME_AVAILABLE:
                     st.error("テキストマイニングにはJanomeライブラリが必要です。サイドバーのエラーメッセージを確認してください。")
                     st.stop()
            
            with st.spinner("AIがコードと解説を生成中です..."):
                schema_json = json.dumps(st.session_state.schema_dict, indent=2, ensure_ascii=False)
                # (NEW) JSON応答を受け取る
                response_data = generate_code_and_explanation(schema_json, user_prompt, api_key)
                
                if response_data:
                    # (NEW) コードと説明文を別々に保存
                    st.session_state.generated_code = response_data.get("code_to_execute", "")
                    st.session_state.analysis_explanation = response_data.get("analysis_explanation", "(説明が生成されませんでした)")
                    st.session_state.exec_output = None 
                    
                    if st.session_state.generated_code:
                        st.success("コードと解説が生成されました。Step 2で確認・実行してください。")
                    else:
                        st.error("AIは応答しましたが、実行可能なコードが含まれていませんでした。")

        st.markdown("---")
        st.header("Step 2: コードの確認と実行")
        if st.session_state.generated_code:
            st.subheader("生成されたPythonコード")
            st.code(st.session_state.generated_code, language="python")
            
            st.warning("AIが生成したコードが意図通りか確認してから実行してください。")
            
            if st.button("▶️ このコードを実行する"):
                with st.spinner("サーバー上でコードを実行中..."):
                    try:
                        # (NEW) 実行環境にJanomeも渡す
                        global_vars = {"pd": pd, "px": px, "go": go}
                        if JANOME_AVAILABLE:
                            global_vars["Tokenizer"] = Tokenizer
                            global_vars["Analyzer"] = Analyzer
                            global_vars["POSKeepFilter"] = POSKeepFilter
                            global_vars["TokenCountFilter"] = TokenCountFilter
                            
                        local_vars = {"df": st.session_state.df.copy()} 
                        
                        exec(st.session_state.generated_code, global_vars, local_vars)
                        
                        output = local_vars.get("output", None)
                        
                        if output is not None:
                            st.session_state.exec_output = output
                            st.success("コードが実行されました。Step 3で結果を確認してください。")
                        else:
                            st.error("コードは実行されましたが、'output' 変数に結果が見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"コードの実行に失敗しました: {e}")
        else:
            st.info("Step 1でコードを生成してください。")

        st.markdown("---")
        st.header("Step 3: 実行結果と分析の解説")
        
        # (NEW) まず分析内容の解説を表示
        if st.session_state.analysis_explanation:
            st.subheader("分析内容の解説（論文用）")
            st.success(f"📄 {st.session_state.analysis_explanation}")
        
        if st.session_state.exec_output is not None:
            
            output = st.session_state.exec_output
            
            # 1. 結果がデータテーブル (DataFrame or Series) の場合
            if isinstance(output, (pd.DataFrame, pd.Series)):
                st.subheader("集計・分析結果 (テーブル)")
                st.dataframe(output, use_container_width=True)
                
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    if not isinstance(df_to_convert, pd.DataFrame):
                        df_to_convert = df_to_convert.to_frame()
                    return df_to_convert.to_csv(index=True).encode('utf-8-sig')
                
                try:
                    csv_data = convert_df_to_csv(output)
                    st.download_button(label="結果をCSVダウンロード", data=csv_data, file_name="analysis_result.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"CSV変換に失敗しました: {e}")

            # 2. 結果がグラフ (Plotly Figure) の場合
            elif isinstance(output, go.Figure):
                st.subheader("生成されたグラフ")
                st.plotly_chart(output, use_container_width=True)
                
                try:
                    img_bytes = output.to_image(format="png", scale=2)
                    st.download_button(
                        label="グラフを画像(PNG)で保存",
                        data=img_bytes,
                        file_name="chart.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"画像のエクスポートに失敗しました (ローカル環境では動作します): {e}")

            # 3. その他の結果
            else:
                st.subheader("実行結果 (その他)")
                st.write(output)
        else:
            st.info("Step 2でコードを実行してください。")


    # --- 右カラム (col2): データ参照（プレビューとスキーマ） ---
    with col2:
        st.header("データ参照")
        
        st.subheader("データの先頭100行 (プレビュー)")
        st.dataframe(st.session_state.df.head(100), use_container_width=True, height=400)
        
        st.markdown("---")
        
        with st.expander("拡張スキーマ (AIに送信する情報) を表示"):
            st.write("AIはこのスキーマ情報（列名、型、カテゴリのユニーク値など）のみを参照してコードを生成します。")
            st.json(st.session_state.schema_dict, expanded=False)
