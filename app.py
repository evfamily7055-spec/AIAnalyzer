import streamlit as st
import pandas as pd
import io
import json
import traceback
import datetime
import streamlit.runtime

# --- ライブラリのインポート (Import Libraries) ---

# Gemini AI
try:
    import google.generativai as genai
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# グラフ描画 (Visualization)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 日本語テキストマイニング (NLP)
try:
    from janome.tokenizer import Tokenizer
    from janome.tokenfilter import POSKeepFilter, TokenCountFilter
    from janome.analyzer import Analyzer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

# 統計的仮説検定 (Statistics)
try:
    from scipy import stats
    import statsmodels.api as sm
    STATS_LIBS_AVAILABLE = True
except ImportError:
    STATS_LIBS_AVAILABLE = False

# --- 定数 (Constants) ---
MAX_UNIQUE_VALUES_FOR_SCHEMA = 20

# --- ページ設定 (Page Config) ---
st.set_page_config(layout="wide")
st.title("AIデータアナリスト (履歴機能付き) 🚀")
st.info("集計・可視化・テキストマイニング・統計検定・論文用解説まで、AIがワンクリックで実行。分析履歴も保存します。")

# --- セッションステートの初期化 (Initialize Session State) ---
def init_session_state():
    defaults = {
        'df': None,
        'schema_dict': None,
        'current_prompt': "",
        'current_code': "",
        'current_output': None,
        'current_explanation': "",
        'current_interpretation': "",
        'last_uploaded_filename': None,
        'analysis_history': []  # (NEW) 履歴保存用リスト
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Gemini API 呼び出し関数 ---
@st.cache_data(ttl=600)
def generate_code_and_explanation(schema_json: str, user_prompt: str, api_key: str):
    """
    拡張スキーマと指示をGeminiに送信し、
    「コード」「分析説明」「統計的解釈」を含むJSONを生成する。
    """
    if not GEMINI_AVAILABLE:
        st.error("google.generativeai ライブラリが見つかりません。`pip install google-generativeai` を実行してください。")
        return None

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"APIキーの設定に失敗しました: {e}")
        return None

    # (v3 Stable) 安定性のため、スタイル(template)オプションを除外し、統計検定の指示を強化
    system_prompt = (
        "あなたは、Pandas, Plotly (px), Janome (NLP), Scipy (stats), Statsmodels (sm) を専門とする世界クラスのPythonデータアナリストです。"
        "あなたの仕事は、渡された「データスキーマ」と「ユーザーの曖昧な指示」から、「実行コード」「分析内容の日本語説明」「統計的解釈」の3つを *JSON形式* で生成することです。"
        
        "## ルール:"
        "1. 出力は *必ず* 以下のJSON形式の *文字列のみ* としてください。説明やマークダウン（```json など）は絶対に含めないでください。"
        "   {\n"
        "     \"code_to_execute\": \"... (ここにPythonコードスニペットを記述) ...\",\n"
        "     \"analysis_explanation\": \"... (ここに分析内容の日本語説明を記述) ...\",\n"
        "     \"statistical_interpretation\": \"... (ここに統計検定の結果の解釈を記述) ...\"\n"
        "   }\n"
        
        "2. `code_to_execute` のルール:"
        "   - 入力データフレームは *常に* `df` という名前です。"
        "   - ユーザーの指示から、集計(Pandas), グラフ(Plotly as px), NLP(Janome), 統計検定(scipy.stats as stats, statsmodels.api as sm) のどれが最適か *推論* してください。"
        "   - (STABILITY) グラフ描画時 (px.pie, px.bar など) 、`template` のような外観に関するオプションは *絶対* に指定しないでください。デフォルトのスタイルを使ってください。"
        "   - コードの *最終行* は、結果（DataFrame, Series, Plotly Figure, または検定結果の文字列/DataFrame）を `output` という単一の変数に *必ず* 代入してください。"
        "   - `print()` や `fig.show()` 文は *絶対に* 使わないでください。"
        
        "3. `analysis_explanation` のルール:"
        "   - `code_to_execute` で実行する分析が *何をしているか* を、学術論文の「方法」セクションで使える、客観的かつ簡潔な日本語で説明してください。"
        "   - （例: 「'Gender' 列をグループ化キーとし、'Age' 列の平均値を算出した。」）"

        "4. `statistical_interpretation` のルール:"
        "   - *統計検定を実行した場合のみ*、その結果（p値、統計量など）を論文の「結果」セクションで使えるように日本語で解釈してください。"
        "   - （例: 「t検定の結果、p値は0.03であり、5%水準で有意な差が認められた。」）"
        "   - *統計検定でない場合（単純集計やグラフ描画）は、このフィールドは空文字列 \"\" としてください。*"

        "5. 統計検定の指示（例: 「差があるか検定」「関連を分析」「相関を調べて」）の場合:"
        "   - スキーマ（データ型、ユニーク値の数）に基づき、最適な検定手法を *自動で選択* してください。"
        "   - (A) 2つの数値変数の関係性 -> 相関分析 (`stats.pearsonr`)。結果は `r, p = stats.pearsonr(...)` とし、`output = f'相関係数(r): {r:.4f}, p値: {p:.4g}'` のように文字列で返してください。"
        "   - (B) カテゴリ変数(2群) vs 数値変数 -> 独立2群のt検定 (`stats.ttest_ind`)。`group1 = df[df['Group'] == 'A']['Value']`, `group2 = ...` のようにデータを準備し、`stats.ttest_ind(group1, group2, nan_policy='omit')` を実行。結果は `stat, p = ...` とし、`output = f't値: {stat:.4f}, p値: {p:.4g}'` で返してください。"
        "   - (C) 2つのカテゴリ変数の関係性 -> カイ二乗検定 (`stats.chi2_contingency`)。`crosstab = pd.crosstab(df['Var1'], df['Var2'])` でクロス表を作成し、`chi2, p, dof, ex = stats.chi2_contingency(crosstab)` を実行。`output = f'カイ二乗値: {chi2:.4f}, p値: {p:.4g}, 自由度: {dof}'` で返してください。"
        "   - (D) 1つの数値変数 (X) から 1つの数値変数 (Y) を予測 -> 単回帰分析 (`sm.OLS`)。`X = sm.add_constant(df['X'].dropna())`, `Y = df['Y'].dropna()`, `X, Y = X.align(Y, join='inner')` で欠損値を除去・整列。`model = sm.OLS(Y, X).fit()`, `output = model.summary().as_text()` で *サマリー全体を文字列として* 返してください。"
        
        "6. (NLP) 日本語テキストマイニングの指示（例: 「自由回答を分析」「単語頻度」）の場合:"
        "   - `janome.tokenizer.Tokenizer` を使用。分析対象は *名詞*, *動詞*, *形容詞* の *原形* としてください。"
        "   - `stop_words` (例: 'する', 'ある', 'ない', 'こと', 'もの') を定義し、除外。"
        "   - 単語頻度をカウントし、上位50件を `pd.DataFrame(..., columns=['word', 'count'])` に格納。"
        "   - 最後に `px.treemap` を使用し、`path=[px.Constant('all'), 'word']`, `values='count'` で結果を可視化し、それを `output` に代入。"
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        system_instruction=system_prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    full_prompt = (
        f"## データスキーマ (JSON形式):\n{schema_json}\n\n"
        f"## ユーザーの指示:\n{user_prompt}"
    )

    try:
        response = model.generate_content(full_prompt)
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
        st.error(f"Gemini API 呼び出し中に予期せぬエラーが発生しました: {e}\n{traceback.format_exc()}")
        return None

# --- サイドバー (Sidebar) ---
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini API Key", type="password", help="Gemini APIキーをここに入力してください。")
    st.markdown("---")
    st.info("このアプリは実データをAIに送信しません。AIには列名とカテゴリのユニーク値（20種類以下）のみが送信されます。")
    
    # ライブラリチェック
    lib_errors = []
    if not GEMINI_AVAILABLE: lib_errors.append("google.generativeai")
    if not PLOTLY_AVAILABLE: lib_errors.append("plotly")
    if not JANOME_AVAILABLE: lib_errors.append("janome (NLP機能が無効)")
    if not STATS_LIBS_AVAILABLE: lib_errors.append("scipy, statsmodels (統計検定が無効)")
    
    if lib_errors:
        st.error(f"以下のライブラリが見つかりません: {', '.join(lib_errors)}\n`pip install -r requirements.txt` を実行してください。")

# --- 1. ファイルアップローダー (File Uploader) ---
uploaded_file = st.file_uploader("Excelファイル (.xlsx) をアップロードしてください", type=["xlsx"])

if uploaded_file is not None:
    # (BUG FIX) ファイル名が前回と異なる「新しい」アップロードの場合のみDFを読み込み、状態をリセット
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        try:
            st.info(f"'{uploaded_file.name}' を読み込んでいます...")
            bytes_data = uploaded_file.getvalue()
            df = pd.read_excel(io.BytesIO(bytes_data))
            
            st.session_state.df = df 
            st.session_state.last_uploaded_filename = uploaded_file.name
            
            # 拡張スキーマの生成
            schema = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                schema[col] = {"dtype": dtype}
                
                # (TypeError FIX) Numpy型をPython標準型にキャスト
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        schema[col]["mean"] = float(df[col].mean())
                        schema[col]["min"] = float(df[col].min())
                        schema[col]["max"] = float(df[col].max())
                    except (TypeError, ValueError):
                        pass # 変換不能な場合はスキップ
                elif dtype == 'object' and df[col].nunique() <= MAX_UNIQUE_VALUES_FOR_SCHEMA:
                    try:
                        unique_vals = df[col].dropna().unique().tolist()
                        schema[col]["unique_values"] = unique_vals
                    except Exception:
                        pass # 比較不能な型（リストなど）の場合はスキップ
                     
            st.session_state.schema_dict = schema
            st.success(f"ファイルの読み込みが完了しました。 (行: {len(df)}, 列: {len(df.columns)})")
            
            # (FIX) 新しいファイルが読み込まれたら、現在の表示と履歴をリセット
            init_session_state()
            st.session_state.df = df # dfとschemaとファイル名だけは保持する
            st.session_state.schema_dict = schema
            st.session_state.last_uploaded_filename = uploaded_file.name
            
        except Exception as e:
            st.error(f"Excelファイルの読み込みに失敗しました: {e}\n{traceback.format_exc()}")
            init_session_state() # エラー時も完全にリセット

# --- (NEW) 分析結果を表示する関数 ---
def display_analysis_results(prompt, code, output, explanation, interpretation):
    """
    左カラム（作業領域）に現在の分析結果を表示する
    """
    st.header("Step 2: 実行結果と分析の解説")

    if explanation:
        st.subheader("分析内容の解説（論文の「方法」用）")
        st.success(f"📄 {explanation}")
    
    if interpretation:
        st.subheader("統計的解釈（論文の「結果」用）")
        st.info(f"📈 {interpretation}")
    
    if output is not None:
        
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
        elif PLOTLY_AVAILABLE and isinstance(output, go.Figure):
            st.subheader("生成されたグラフ")
            st.plotly_chart(output, use_container_width=True)
            
            # (FIX) Streamlit Cloud (is_streamlit_cloud) かどうかを判定
            is_cloud = False
            if streamlit.runtime.exists():
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                try:
                    # Streamlit Cloud (Share) は is_owner が False になる
                    is_cloud = not get_script_run_ctx().client.is_owner
                except Exception:
                    # ローカルや判定不能な場合は Cloud ではないとみなす
                    pass
            
            if is_cloud:
                st.warning("グラフを保存するには、グラフ右上の「カメラ📷」アイコンをクリックし、「Download plot as a png」を選択してください。")
            else:
                # ローカル環境の場合のみダウンロードボタンを表示
                try:
                    img_bytes = output.to_image(format="png", scale=2)
                    st.download_button(
                        label="グラフを画像(PNG)で保存 (ローカル環境)",
                        data=img_bytes,
                        file_name="chart.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"画像のエクスポートに失敗しました。Chromeがインストールされているか確認してください: {e}")

        # 3. 結果が統計サマリー (文字列) の場合
        elif isinstance(output, str):
            st.subheader("分析・検定結果 (サマリー)")
            st.text(output) 
        
        # 4. その他の結果
        else:
            st.subheader("実行結果 (その他)")
            st.write(output)
    else:
        st.info("Step 1で分析指示を出し、「分析を実行」ボタンを押してください。")

    # 実行されたコードは、結果の下に折りたたんで表示
    if code:
        with st.expander(f"指示:「{prompt}」のために実行されたPythonコード"):
            st.code(code, language="python")

# --- 2. メインの作業領域 (左右分割) ---
if st.session_state.df is not None:
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1]) # 左カラムを広くする (2:1)

    # --- 左カラム (col1): AIへの指示と実行結果（作業領域） ---
    with col1:
        st.header("Step 1: AIへの分析指示")
        st.write("右側でデータ（列名）を確認しながら、実行したい内容を日本語で指示してください。")
        
        user_prompt = st.text_area(
            "指示入力欄:",
            placeholder=(
                "（集計例）: 「男女別の年齢の平均値」\n"
                "（可視化例）: 「'年齢' と '給与' の散布図を表示」\n"
                "（NLP例）: 「'自由回答' 列の単語頻度を可視化」\n"
                "（検定例）: 「'介入群' と '対照群' で 'スコア' に差があるか検定」\n"
                "（検定例）: 「'年齢' と '給与' の相関を分析して」"
            ),
            height=150,
            key="current_prompt" # (NEW) 履歴呼び出しのためにキーを設定
        )

        if st.button("🤖 分析を実行", type="primary"):
            if not api_key:
                st.error("サイドバーからGemini APIキーを入力してください。")
            elif not user_prompt:
                st.warning("指示を入力してください。")
            # (省略) ライブラリチェック ...
            
            else:
                with st.spinner("AIがコードを生成し、サーバー上で実行中です..."):
                    # 1. AIコード生成
                    schema_json = json.dumps(st.session_state.schema_dict, indent=2, ensure_ascii=False)
                    response_data = generate_code_and_explanation(schema_json, user_prompt, api_key)
                    
                    if not response_data or "code_to_execute" not in response_data:
                        st.error("AIによるコード生成に失敗しました。")
                        st.stop()

                    code = response_data.get("code_to_execute", "")
                    explanation = response_data.get("analysis_explanation", "(説明なし)")
                    interpretation = response_data.get("statistical_interpretation", "")
                    
                    if not code:
                        st.error("AIは応答しましたが、実行可能なコードが含まれていませんでした。")
                        st.stop()

                    # 2. コード実行
                    output = None
                    try:
                        global_vars = {"pd": pd}
                        if PLOTLY_AVAILABLE:
                            global_vars["px"] = px
                            global_vars["go"] = go
                        if JANOME_AVAILABLE:
                            global_vars["Tokenizer"] = Tokenizer
                            global_vars["Analyzer"] = Analyzer
                            global_vars["POSKeepFilter"] = POSKeepFilter
                            global_vars["TokenCountFilter"] = TokenCountFilter
                        if STATS_LIBS_AVAILABLE:
                            global_vars["stats"] = stats
                            global_vars["sm"] = sm
                            
                        local_vars = {"df": st.session_state.df.copy()} 
                        
                        exec(code, global_vars, local_vars)
                        
                        output = local_vars.get("output", None)
                        
                        if output is not None:
                            # (NEW) 履歴と現在の状態に保存
                            st.session_state.current_prompt = user_prompt
                            st.session_state.current_code = code
                            st.session_state.current_output = output
                            st.session_state.current_explanation = explanation
                            st.session_state.current_interpretation = interpretation
                            
                            # (NEW) 履歴リストの先頭に追加
                            history_entry = {
                                "id": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "prompt": user_prompt,
                                "code": code,
                                "output": output,
                                "explanation": explanation,
                                "interpretation": interpretation
                            }
                            st.session_state.analysis_history.insert(0, history_entry)
                            
                            st.success("分析が実行されました。")
                        else:
                            st.error("コードは実行されましたが、'output' 変数に結果が見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"コードの実行に失敗しました: {e}\n{traceback.format_exc()}")
                        st.session_state.current_output = None # 失敗したら結果をクリア

        st.markdown("---")
        
        # (NEW) 現在の状態（current）を表示する
        display_analysis_results(
            st.session_state.current_prompt,
            st.session_state.current_code,
            st.session_state.current_output,
            st.session_state.current_explanation,
            st.session_state.current_interpretation
        )


    # --- 右カラム (col2): データ参照（プレビュー、スキーマ、履歴） ---
    with col2:
        st.header("データ参照")
        
        st.subheader("データの先頭100行 (プレビュー)")
        st.dataframe(st.session_state.df.head(100), use_container_width=True, height=300)
        
        with st.expander("拡張スキーマ (AIに送信する情報) を表示"):
            st.write("AIはこのスキーマ情報のみを参照します。")
            st.json(st.session_state.schema_dict, expanded=False)

        st.markdown("---")
        
        # (NEW) 分析履歴
        st.header("分析履歴")
        if not st.session_state.analysis_history:
            st.caption("分析を実行すると、ここに履歴が保存されます。")
        
        for i, entry in enumerate(st.session_state.analysis_history):
            # 履歴項目をボタンとして表示
            if st.button(f"🕒 {entry['id']}\n{entry['prompt'][:50]}..."):
                # (NEW) 履歴呼び出し: ボタンが押されたら、その履歴を「現在」の状態にコピーする
                st.session_state.current_prompt = entry["prompt"]
                st.session_state.current_code = entry["code"]
                st.session_state.current_output = entry["output"]
                st.session_state.current_explanation = entry["explanation"]
                st.session_state.current_interpretation = entry["interpretation"]
                # Streamlitは自動で再実行され、左カラムの表示が更新される
