import streamlit as st
import pandas as pd
import io
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import traceback

# グラフ描画ライブラリ
import plotly.express as px
import plotly.graph_objects as go

# 日本語テキストマイニング（形態素解析）ライブラリ
try:
    from janome.tokenizer import Tokenizer
    from janome.tokenfilter import POSKeepFilter, TokenCountFilter
    from janome.analyzer import Analyzer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

# 統計的仮説検定ライブラリ
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
st.title("AIデータアナリスト (ワンクリック実行) 🚀")
st.info("集計・可視化・テキストマイニング・統計検定・論文用解説の生成まで、AIがワンクリックで実行します。")

# --- セッションステートの初期化 (Initialize Session State) ---
if 'df' not in st.session_state:
    st.session_state.df = None 
if 'schema_dict' not in st.session_state:
    st.session_state.schema_dict = None 
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = "" 
if 'exec_output' not in st.session_state:
    st.session_state.exec_output = None 
if 'analysis_explanation' not in st.session_state:
    st.session_state.analysis_explanation = "" 
if 'statistical_interpretation' not in st.session_state:
    st.session_state.statistical_interpretation = ""
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# --- Gemini API 呼び出し関数 ---
@st.cache_data(ttl=600) 
def generate_code_and_explanation(schema_json: str, user_prompt: str, api_key: str):
    """
    拡張スキーマと指示をGeminiに送信し、
    「コード」「分析説明」「統計的解釈」を含むJSONを生成する。
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"APIキーの設定に失敗しました: {e}")
        return None

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
        "   - (B) カテゴリ変数(2群) vs 数値変数 -> 独立2群のt検定 (`stats.ttest_ind`)。結果は `stat, p = stats.ttest_ind(...)` とし、`output = f't値: {stat:.4f}, p値: {p:.4g}'` で返してください。"
        "   - (C) 2つのカテゴリ変数の関係性 -> カイ二乗検定 (`stats.chi2_contingency`)。`pd.crosstab` でクロス表を作成し、`chi2, p, dof, ex = stats.chi2_contingency(crosstab)` を実行。`output = f'カイ二乗値: {chi2:.4f}, p値: {p:.4g}, 自由度: {dof}'` で返してください。"
        "   - (D) 1つの数値変数 (X) から 1つの数値変数 (Y) を予測 -> 単回帰分析 (`sm.OLS`)。`X = sm.add_constant(df['X'])`, `model = sm.OLS(df['Y'], X).fit()`, `output = model.summary().as_text()` で *サマリー全体を文字列として* 返してください。"
        
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

# --- サイドバー (APIキー入力) ---
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("Gemini API Key", type="password", help="Gemini APIキーをここに入力してください。")
    st.markdown("---")
    st.info("このアプリは実データをAIに送信しません。AIには列名とカテゴリのユニーク値（20種類以下）のみが送信されます。")
    if not JANOME_AVAILABLE:
        st.error("Janomeライブラリが見つかりません。テキストマイニング機能は無効です。\n`pip install janome` を実行してください。")
    if not STATS_LIBS_AVAILABLE:
        st.error("ScipyまたはStatsmodelsが見つかりません。統計検定機能は無効です。\n`pip install scipy statsmodels` を実行してください。")

# --- 1. ファイルアップローダー ---
uploaded_file = st.file_uploader("Excelファイル (.xlsx) をアップロードしてください", type=["xlsx"])

# (BUG FIX) 
# uploaded_file オブジェクトがNoneでなく、
# かつ「ファイル名が前回と異なる」場合にのみ、DFの読み込みと状態のリセットを実行する
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        try:
            st.info(f"'{uploaded_file.name}' を読み込んでいます...")
            bytes_data = uploaded_file.getvalue()
            df = pd.read_excel(io.BytesIO(bytes_data))
            
            st.session_state.df = df 
            st.session_state.last_uploaded_filename = uploaded_file.name # (FIX) ファイル名を記憶
            
            # 拡張スキーマの生成
            schema = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                schema[col] = {"dtype": dtype}
                
                if dtype == 'object' and df[col].nunique() <= MAX_UNIQUE_VALUES_FOR_SCHEMA:
                    unique_vals = df[col].dropna().unique().tolist()
                    schema[col]["unique_values"] = unique_vals
                elif pd.api.types.is_numeric_dtype(df[col]):
                     try:
                         schema[col]["mean"] = float(df[col].mean())
                         schema[col]["min"] = float(df[col].min())
                         schema[col]["max"] = float(df[col].max())
                     except Exception:
                         pass 
                     
            st.session_state.schema_dict = schema
            st.success(f"ファイルの読み込みが完了しました。 (行: {len(df)}, 列: {len(df.columns)})")
            
            # (FIX) 状態をリセット
            st.session_state.generated_code = ""
            st.session_state.exec_output = None 
            st.session_state.analysis_explanation = ""
            st.session_state.statistical_interpretation = ""

        except Exception as e:
            st.error(f"Excelファイルの読み込みに失敗しました: {e}")
            st.session_state.df = None
            st.session_state.last_uploaded_filename = None

# --- 2. メインの作業領域 (左右分割) ---
if st.session_state.df is not None:
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    # --- 左カラム (col1): AIへの指示と実行（作業領域） ---
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
            height=150
        )

        # (UX CHANGE) ボタンを「分析を実行」に変更
        if st.button("🤖 分析を実行", type="primary"):
            if not api_key:
                st.error("サイドバーからGemini APIキーを入力してください。")
            elif not user_prompt:
                st.warning("指示を入力してください。")
            
            # ライブラリチェック
            elif ("テキスト" in user_prompt or "単語" in user_prompt or "NLP" in user_prompt) and not JANOME_AVAILABLE:
                 st.error("テキストマイニングにはJanomeライブラリが必要です。サイドバーのエラーメッセージを確認してください。")
                 st.stop()
            elif ("検定" in user_prompt or "分析" in user_prompt or "差" in user_prompt or "関連" in user_prompt or "相関" in user_prompt) and not STATS_LIBS_AVAILABLE:
                 st.error("統計検定にはScipyとStatsmodelsが必要です。サイドバーのエラーメッセージを確認してください。")
                 st.stop()
            
            # --- (NEW) ワンクリックで「生成」と「実行」を両方行う ---
            else:
                with st.spinner("AIがコードを生成し、サーバー上で実行中です..."):
                    # 1. AIコード生成
                    schema_json = json.dumps(st.session_state.schema_dict, indent=2, ensure_ascii=False)
                    response_data = generate_code_and_explanation(schema_json, user_prompt, api_key)
                    
                    if not response_data or "code_to_execute" not in response_data:
                        st.error("AIによるコード生成に失敗しました。")
                        st.stop()

                    st.session_state.generated_code = response_data.get("code_to_execute", "")
                    st.session_state.analysis_explanation = response_data.get("analysis_explanation", "(説明が生成されませんでした)")
                    st.session_state.statistical_interpretation = response_data.get("statistical_interpretation", "")
                    
                    if not st.session_state.generated_code:
                        st.error("AIは応答しましたが、実行可能なコードが含まれていませんでした。")
                        st.stop()

                    # 2. コード実行
                    try:
                        global_vars = {"pd": pd, "px": px, "go": go}
                        if JANOME_AVAILABLE:
                            global_vars["Tokenizer"] = Tokenizer
                            global_vars["Analyzer"] = Analyzer
                            global_vars["POSKeepFilter"] = POSKeepFilter
                            global_vars["TokenCountFilter"] = TokenCountFilter
                        if STATS_LIBS_AVAILABLE:
                            global_vars["stats"] = stats
                            global_vars["sm"] = sm
                            
                        local_vars = {"df": st.session_state.df.copy()} 
                        
                        exec(st.session_state.generated_code, global_vars, local_vars)
                        
                        output = local_vars.get("output", None)
                        
                        if output is not None:
                            st.session_state.exec_output = output
                            st.success("分析が実行されました。Step 2で結果を確認してください。")
                        else:
                            st.session_state.exec_output = None
                            st.error("コードは実行されましたが、'output' 変数に結果が見つかりませんでした。")
                            
                    except Exception as e:
                        st.session_state.exec_output = None
                        st.error(f"コードの実行に失敗しました: {e}\n{traceback.format_exc()}")

        st.markdown("---")
        
        # (UX CHANGE) Step 2 を「実行結果」に変更
        st.header("Step 2: 実行結果と分析の解説")
        
        if st.session_state.analysis_explanation:
            st.subheader("分析内容の解説（論文の「方法」用）")
            st.success(f"📄 {st.session_state.analysis_explanation}")
        
        if st.session_state.statistical_interpretation:
            st.subheader("統計的解釈（論文の「結果」用）")
            st.info(f"📈 {st.session_state.statistical_interpretation}")
        
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

        # (UX CHANGE) 実行されたコードは、結果の下に折りたたんで表示
        if st.session_state.generated_code:
            with st.expander("今回実行されたPythonコードを表示"):
                st.code(st.session_state.generated_code, language="python")

    # --- 右カラム (col2): データ参照（プレビューとスキーマ） ---
    with col2:
        st.header("データ参照")
        
        st.subheader("データの先頭100行 (プレビュー)")
        st.dataframe(st.session_state.df.head(100), use_container_width=True, height=400)
        
        st.markdown("---")
        
        with st.expander("拡張スキーマ (AIに送信する情報) を表示"):
            st.write("AIはこのスキーマ情報（列名、型、カテゴリのユニーク値など）のみを参照してコードを生成します。")
            st.json(st.session_state.schema_dict, expanded=False)
