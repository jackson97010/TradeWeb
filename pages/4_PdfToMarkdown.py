import streamlit as st
import pymupdf4llm
import pathlib

def pdf_to_markdown_page():
    st.title("PDF → Markdown Converter")

    st.write("上傳 PDF 檔案，並將內容轉換成 Markdown。")

    # Option to select chunking pages or not
    page_chunking = st.sidebar.checkbox("Page Chunks?", value=False, 
                                        help="If checked, will chunk pages into separate markdown sections.")

    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if pdf_file is not None:
        # We save the uploaded PDF to a temporary location
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Convert PDF to Markdown
        # If you want images extracted, set write_images=True.
        # If you want page chunking, set page_chunks=True.
        md_text = pymupdf4llm.to_markdown(
            temp_pdf_path,
            write_images=True,
            page_chunks=page_chunking
        )

        st.subheader("Markdown Output")
        # Display the converted Markdown text in Streamlit
        st.markdown(md_text, unsafe_allow_html=True)

        # Optionally, you can also save the MD text to a file if you like
        # out_path = pathlib.Path("output.md")
        # out_path.write_text(md_text, encoding="utf-8")
        # st.write(f"已將 Markdown 儲存至 {out_path}")

    else:
        st.info("請上傳 PDF 檔案後，Markdown 內容將顯示於此。")

def main():
    pdf_to_markdown_page()

if __name__ == "__main__":
    main()
