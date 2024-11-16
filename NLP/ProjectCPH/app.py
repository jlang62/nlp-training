import streamlit as st
from test import process_file

def main():
    st.title("Document Analysis App")
    st.write("Upload a PDF or Image file to analyze.")

    file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg"])
    file_type = st.selectbox("File Type", ["pdf", "image"])

    if st.button("Process File") and file:
        # Save file temporarily
        file_path = f"./temp.{file_type}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        # Process file
        output_data = process_file(file_path, file_type)

        # Display JSON output
        st.json(output_data)

# Run Streamlit app
if __name__ == "__main__":
    main()
