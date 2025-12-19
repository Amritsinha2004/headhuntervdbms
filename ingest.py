import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import zipfile
import io
import os
import hashlib

# --- CONFIGURATION ---
# POINT THIS TO YOUR ZIP FILE LOCATION
ZIP_FILE_PATH = r"C:\Users\asus\Downloads\archive (2).zip"

# Database Path (Must match your Backend.py)
CHROMA_PATH = "local_pdf_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def setup_db():
    print("⏳ Connecting to Database...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="my_documents",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ Database Connected.")
    return collection

def extract_text_from_bytes(pdf_bytes):
    try:
        # Wrap bytes in a memory stream so pypdf thinks it's a file
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception:
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def main():
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"❌ Error: Could not find file at {ZIP_FILE_PATH}")
        return

    collection = setup_db()
    
    print(f"📂 Opening {ZIP_FILE_PATH}...")
    
    success_count = 0
    fail_count = 0

    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
        # Get list of all files inside the zip
        all_files = z.namelist()
        
        # --- THE CHANGE IS HERE ---
        # 1. Find all PDFs
        # 2. Add [:100] to take only the first 100
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')][:500]
        
        total_files = len(pdf_files)
        print(f"found {len(all_files)} total files, filtering for the first {total_files} PDFs...")

        for i, filename in enumerate(pdf_files):
            try:
                # Read file directly from zip into memory
                pdf_bytes = z.read(filename)
                
                # Extract Text
                text = extract_text_from_bytes(pdf_bytes)
                if not text.strip():
                    continue # Skip empty files

                # Chunking
                chunks = chunk_text(text)
                if not chunks:
                    continue

                # Create IDs and Metadata
                clean_filename = os.path.basename(filename)
                
                ids = [hashlib.md5(f"{clean_filename}_{idx}".encode()).hexdigest() for idx in range(len(chunks))]
                metadatas = [{"source": clean_filename} for _ in chunks]

                # Add to DB
                collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                
                success_count += 1
                
                # Print progress every 10 files (since we are only doing 100)
                if success_count % 10 == 0:
                    print(f"   Processed {success_count}/{total_files} files...")

            except Exception as e:
                fail_count += 1
                print(f"   ⚠️ Failed to process {filename}: {e}")

    print("------------------------------------------------")
    print(f"🎉 COMPLETED!")
    print(f"✅ Successfully added: {success_count} resumes")
    print(f"❌ Failed/Empty: {fail_count}")
    print("------------------------------------------------")
    print("You can now run 'uvicorn Backend:app' and search these resumes!")

if __name__ == "__main__":
    main()