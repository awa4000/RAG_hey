import os
import warnings

# 警告をターミナルに表示させない
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# データベースフォルダの定義
CHROMA_DIR = "./chroma_db"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. 共通の準備（Embeddingsだけ先に作っておく）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# 🔍 【超シンプル存在チェック】
# もし、すでに「chroma_db」というフォルダが存在するなら...
if os.path.exists(CHROMA_DIR):
    
    print(f"✨ 既存のデータベース『{CHROMA_DIR}』を発見しました！")
    print("🚀 WebスクレイピングとAI整形をすべてスキップして、既存のDBをそのまま読み込みます。")
    
    # 既存のChromaデータベースを読み込んで変数 `db` に入れる（一瞬で終わります）
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# もし、まだデータベースのフォルダがないなら（初回起動時）
else:
    
    print(f"📂 データベース『{CHROMA_DIR}』が見つかりません。新規作成を開始します！")
    
    # 2. 脳みそ（LLM）を準備
    

    # 3. URLを自動生成してWebサイトから生データをロード
    urls = [f"https://hey-smith.com/live?y={year}&m={month}" for year in [2026, 2027] for month in range(1, 13) if not (year == 2026 and month < 6)]
    print("🌐 HEY-SMITH公式サイトから最新のライブ情報をロード中...")
    loader = WebBaseLoader(urls)
    documents = loader.load()

    # 4. 最強の命令書（プロンプト）をデザイン
    clean_prompt = ChatPromptTemplate.from_template(
        "あなたは優秀なデータ整形職人です。以下のHEY-SMITHの公式Webサイトから読み込んだ生のテキストから、"
        "【日時】【会場】【出演者】【フェスツアー名】の4つの情報だけを綺麗に抜き出し、"
        "マークダウン形式（- を使った箇条書き）で整理してください。それ以外の挨拶やメニュー項目などのゴミデータは完全に無視（削除）してください。\n\n"
        "--- 生のテキスト ---\n"
        "{raw_text}"
    )

    # 5. 抽出用のChain（パイプライン）を組み立てる
    clean_chain = clean_prompt | llm | StrOutputParser()

    print("✂️ [OpenAIパワー] 全自動でゴミデータを削ぎ落とし、4要素の極上データに整形中...")

    chroma_documents = []

    # 各ページの生テキストをAIに読ませて整形するループ
    for doc in documents:
        page_clean_text = clean_chain.invoke({"raw_text": doc.page_content})
        
        # 中身が空っぽでなければ、Chroma用のデータ形式（Document）に包む
        if page_clean_text.strip():
            chroma_doc = Document(
                page_content=page_clean_text,
                metadata={"source": doc.metadata.get("source")}
            )
            chroma_documents.append(chroma_doc)

    # 6. データをベクトル化してChromaに保存！
    print("💾 綺麗にしたライブデータをChroma（ベクトルDB）に構築中...")
    db = Chroma.from_documents(
        documents=chroma_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"🏆 Chroma DBが『{CHROMA_DIR}』にできました！")


from langchain_core.runnables import RunnablePassthrough

# =====================================================================
# 🌟 ここから先：RAGパイプラインの組み立て
# =====================================================================

# 1. データベース（db）の検索機（Retriever）
retriever = db.as_retriever(search_kwargs={"k": 3})

# 2. 検索した文脈（カンペ）とユーザーの質問を合わせる「RAG専用プロンプト」をデザイン
rag_prompt = ChatPromptTemplate.from_template(
    "あなたはHEY-SMITHの公式スケジュールを案内する優秀なAIアシスタントです。\n"
    "以下の【提供されたライブスケジュール情報】だけを絶対に守って、ユーザーからの質問に正確に答えてください。\n"
    "情報に含まれていない事実を勝手に妄想して答えてはいけません。\n\n"
    "【提供されたライブスケジュール情報】\n"
    "{context}\n\n"
    "ユーザーからの質問: {question}"
)

# 検索したDocument（箱）からテキストだけを抽出するお助け関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. DB、プロンプト、検索機、パーサーを1本のChainに繋ぎ合わせる！
rag_chain = (
    {
        "context": retriever | format_docs,  #  質問を元にChromaから検索し、テキストに変換して {context} に入れる
        "question": RunnablePassthrough()    #  ユーザーが入力した質問を、そのまま {question} に流し込む
    }
    | rag_prompt  # プロンプトに2つのデータが合体して流れる
    | llm         # カンペを持った状態でOpenAIが考える
    | StrOutputParser()  # 文字列だけにする！
)

#実際に対話から情報を得る
while True:
    # コマンドプロンプトからユーザーの入力を受け取る
    user_question = input("\n👤 あなた ➔ ")
    
    # 空文字なら何もしない
    if not user_question.strip():
        continue
        
    # 終了ワードが入力されたら、ループを抜けてプログラムを終わらせる
    if user_question.strip() in ["バイバイ", "ばいばい", "exit", "quit"]:
        print("\n🤖 AI ➔ またいつでも聞いてくれよな！🤘")
        break

    print("🤖 AI ➔ スケジュールを確認中...")
    
    try:
        # 4. ユーザーの質問をChainに流し込んで回答を得る！
        answer = rag_chain.invoke(user_question)
        
        print("\n--- 🤖 AIからの回答 ---")
        print(answer)
        print("-----------------------")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しちゃったみたいだ：{e}")