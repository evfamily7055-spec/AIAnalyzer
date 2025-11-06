import streamlit as st
import pandas as pd
import io
import json
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- 定数 (Constants) ---
MAX_UNIQUE_VALUES_FOR_SCHEMA = 20

# --- ページ設定 (Page Config) ---
st.set_page_config(layout="wide")
st.title("AIコード生成アナリスト 👨‍💻 (左右分割・高推論)")
st.info("右側でデータ（列名）を確認しながら、左側でAIに指示を出し、集計コードを生成・実行します。")

# --- セッションステートの初期化 (Initialize Session State) ---
if 'df' not in st.session_state:
    st.session_state.df = None 
if 'schema_dict' not in st.session_state:
    st.session_state.schema_dict = None 
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = "" 
if 'exec_result' not in st.session_state:
    st.session_state.exec_result = None 

# --- (NEW) Gemini API 呼び出し関数 (Python) ---
@st.cache_data(ttl=600) 
def generate_code_from_ai(schema_json: str, user_prompt: str, api_key: str):
    """
    拡張スキーマとユーザーの指示をGemini APIに送信し、Pandasコードを生成する。
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"APIキーの設定に失敗しました: {e}")
        return None

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

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        system_instruction=system_prompt
    )
    
    full_prompt = (
        f"## データスキーマ (JSON形式):\n{schema_json}\n\n"
        f"## ユーザーの指示:\n{user_prompt}"
    )

    try:
        response = model.generate_content(full_prompt)
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
        st.session_state.exec_result = None

    except Exception as e:
        st.error(f"Excelファイルの読み込みに失敗しました: {e}")
        st.session_state.df = None


# --- (NEW) 2. メインの作業領域 (左右分割) ---
if st.session_state.df is not None:
    st.markdown("---")
    
    # (NEW) 画面を5:5の2カラムに分割
    col1, col2 = st.columns(2)

    # --- 左カラム (col1): AIへの指示と実行（作業領域） ---
    with col1:
        st.header("Step 1: AIへの集計指示")
        st.write("右側でデータ（列名）を確認しながら、実行したい集計を日本語で指示してください。")
        
        user_prompt = st.text_area(
            "指示入力欄:",
            placeholder="例: 「男女別の年齢の平均値」\n例: 「満足度（高・中・低）の人数をカウント」\n例: 「'所属' 列に '営業部' が含まれる行だけを抽出」",
            height=150
        )

        if st.button("🤖 AIコード生成", type="primary"):
            if not api_key:
                st.error("サイドバーからGemini APIキーを入力してください。")
            elif not user_prompt:
                st.warning("指示を入力してください。")
            else:
                with st.spinner("AIがPandasコードを生成中です..."):
                    schema_json = json.dumps(st.session_state.schema_dict, indent=2, ensure_ascii=False)
                    generated_code = generate_code_from_ai(schema_json, user_prompt, api_key)
                    
                    if generated_code:
                        st.session_state.generated_code = generated_code
                        st.session_state.exec_result = None # 実行結果をリセット
                        st.success("コードが生成されました。Step 2で確認・実行してください。")

        st.markdown("---")
        st.header("Step 2: コードの確認と実行")
        if st.session_state.generated_code:
            st.subheader("生成されたPandasコード")
            st.code(st.session_state.generated_code, language="python")
            
            st.warning("AIが生成したコードが意図通りか確認してから実行してください。")
            
            if st.button("▶️ このコードを実行する"):
                with st.spinner("サーバー上でコードを実行中..."):
                    try:
                        global_vars = {"pd": pd}
                        local_vars = {"df": st.session_state.df.copy()} 
                        
                        exec(st.session_state.generated_code, global_vars, local_vars)
                        
                        result = local_vars.get("result", None)
                        
                        if result is not None:
                            st.session_state.exec_result = result
                            st.success("コードが実行されました。Step 3で確認してください。")
                        else:
                            st.error("コードは実行されましたが、'result' 変数に結果が見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"コードの実行に失敗しました: {e}")
        else:
            st.info("Step 1でコードを生成してください。")

        st.markdown("---")
        st.header("Step 3: 集計・分析結果")
        if st.session_state.exec_result is not None:
            result = st.session_state.exec_result
            
            if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                st.dataframe(result, use_container_width=True)
                
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    if not isinstance(df_to_convert, pd.DataFrame):
                        df_to_convert = df_to_convert.to_frame()
                    return df_to_convert.to_csv(index=True).encode('utf-8-sig')
                
                try:
                    csv_data = convert_df_to_csv(result)
                    st.download_button(label="結果をCSVダウンロード", data=csv_data, file_name="analysis_result.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"CSV変換に失敗しました: {e}")
                
            else:
                st.write("実行結果:")
                st.write(result)
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
