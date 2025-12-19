import os
import shutil
import hashlib
import zipfile
import logging
import re
import sqlite3
import sys # Added for dependency check
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from fpdf import FPDF
from passlib.context import CryptContext
from jose import jwt, JWTError

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ResumeEngine")

# --- 2. CONFIGURATION ---
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent
    CHROMA_PATH: Path = BASE_DIR / "local_pdf_db"
    STORAGE_PATH: Path = BASE_DIR / "stored_pdfs"
    TEMP_PATH: Path = BASE_DIR / "temp_uploads"
    USER_DB_PATH: Path = BASE_DIR / "users.db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MIN_CONFIDENCE_SCORE: float = 20.0
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # SECURITY CONFIG
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PROD"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

settings = Settings()

# Ensure directories exist
for path in [settings.STORAGE_PATH, settings.TEMP_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# --- 3. SECURITY & DATABASE SETUP ---
try:
    # Switch to Argon2 (Requires: pip install argon2-cffi)
    pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
except Exception as e:
    logger.critical("Failed to load Argon2. Did you run 'pip install argon2-cffi'?")
    sys.exit(1)

def init_user_db():
    try:
        conn = sqlite3.connect(str(settings.USER_DB_PATH))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT, full_name TEXT)''')
        
        # MIGRATION: Attempt to add full_name if it doesn't exist
        try:
            c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
            logger.info("Migrated Database: Added 'full_name' column.")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        conn.commit()
        conn.close()
        logger.info(f"User Database initialized at: {settings.USER_DB_PATH}")
    except Exception as e:
        logger.critical(f"Failed to initialize User Database: {e}")

init_user_db()

# --- 4. DATA MODELS ---
class ResumeFormData(BaseModel):
    full_name: str
    email: str
    job_title: str
    summary: str
    skills: str
    experience: str
    education: str

class UserSignup(BaseModel):
    username: str
    password: str
    full_name: str
    role: str

class UserLogin(BaseModel):
    username: str
    password: str

# --- 5. BUSINESS LOGIC ---

class AuthHandler:
    @staticmethod
    def validate_password_strength(password: str):
        """
        Enforces Strong Password Policy:
        1. At least 8 characters
        2. At least one Uppercase letter (A-Z)
        3. At least one Lowercase letter (a-z)
        4. At least one Digit (0-9)
        5. At least one Special Character (!@#$...)
        """
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter (A-Z)")
        if not re.search(r"[a-z]", password):
            raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter (a-z)")
        if not re.search(r"\d", password):
            raise HTTPException(status_code=400, detail="Password must contain at least one number (0-9)")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            raise HTTPException(status_code=400, detail="Password must contain at least one special character (!@#$...)")
        return True

    @staticmethod
    def get_password_hash(password):
        # Argon2 handles long passwords natively
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

class DocumentProcessor:
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = os.path.basename(filename)
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

    @staticmethod
    def extract_text(file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path.name}: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        if not text: return []
        words = text.split()
        chunks = []
        for i in range(0, len(words), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + settings.CHUNK_SIZE])
            if len(chunk) > 50: chunks.append(chunk)
        return chunks

class PDFGenerator:
    @staticmethod
    def create_resume_pdf(data: ResumeFormData, output_path: Path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 10, data.full_name, ln=True)
        pdf.set_font("Arial", "I", 14)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, data.job_title, ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, data.email, ln=True)
        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        def add_section(title, content):
            if not content: return
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 0, 128)
            pdf.cell(0, 8, title.upper(), ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
            clean_content = content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, clean_content)
            pdf.ln(5)

        add_section("Summary", data.summary)
        add_section("Skills", data.skills)
        add_section("Experience", data.experience)
        add_section("Education", data.education)
        
        pdf.output(str(output_path))

class VectorDB:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
            self.collection = self.client.get_or_create_collection(
                name="corporate_resumes",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.critical(f"DB Error: {e}")

    def add_document(self, filename: str, chunks: List[str]) -> int:
        if not chunks: return 0
        ids = [hashlib.md5(f"{filename}_{i}".encode()).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query], n_results=limit, include=['documents', 'metadatas', 'distances']
        )
        structured = []
        if results['documents']:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                score = round((1 - dist) * 100, 1)
                if score >= settings.MIN_CONFIDENCE_SCORE:
                    structured.append({"text": doc, "source": meta['source'], "score": score})
        return structured

# --- 6. APP INIT ---
db = VectorDB()
processor = DocumentProcessor()
app = FastAPI(title="TalentScout API")

app.mount("/pdfs", StaticFiles(directory=str(settings.STORAGE_PATH)), name="pdfs")

# Updated CORS to handle Credentials properly
app.add_middleware(
    CORSMiddleware,
    # Allow any HTTP/HTTPS origin (safer than "*" for credentials)
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 7. PAGE ROUTES (HTML SERVING) ---

@app.get("/")
async def root():
    file_path = settings.BASE_DIR / "index.html"
    if not file_path.exists():
        return {"error": "index.html not found. Ensure it exists in the directory."}
    return FileResponse(file_path)

@app.get("/{page_name}.html")
async def serve_pages(page_name: str):
    allowed_pages = ["index", "admin", "candidate", "auth", "auth_candidate", "recruiter_auth"]
    if page_name in allowed_pages:
        file_path = settings.BASE_DIR / f"{page_name}.html"
        if file_path.exists():
            return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Page not found")

# --- 8. API ENDPOINTS ---

@app.post("/auth/signup")
async def signup(user: UserSignup):
    logger.info(f"Signup attempt for username: {user.username} (Role: {user.role})")
    
    if user.role not in ["admin", "candidate"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    AuthHandler.validate_password_strength(user.password)

    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    try:
        hashed_pw = AuthHandler.get_password_hash(user.password)
        c.execute("INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                  (user.username, hashed_pw, user.role, user.full_name))
        conn.commit()
        logger.info(f"User {user.username} created successfully.")
    except sqlite3.IntegrityError:
        conn.close()
        logger.warning(f"Signup failed: Username {user.username} already exists.")
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        conn.close()
        logger.error(f"Database error during signup: {e}")
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
        
    conn.close()
    return {"message": "Account created successfully"}

@app.post("/auth/login")
async def login(creds: UserLogin):
    logger.info(f"Login attempt for username: {creds.username}")
    
    conn = sqlite3.connect(str(settings.USER_DB_PATH))
    c = conn.cursor()
    c.execute("SELECT password_hash, role, full_name FROM users WHERE username=?", (creds.username,))
    result = c.fetchone()
    conn.close()

    if not result:
        logger.warning(f"Login failed: User {creds.username} not found.")
        raise HTTPException(status_code=401, detail="User not found")
    
    if not AuthHandler.verify_password(creds.password, result[0]):
        logger.warning(f"Login failed: Incorrect password for {creds.username}.")
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    logger.info(f"Login success: {creds.username} logged in as {result[1]}")
    token = AuthHandler.create_access_token({"sub": creds.username, "role": result[1]})
    return {"access_token": token, "token_type": "bearer", "role": result[1], "name": result[2]}

@app.post("/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    stats = {"processed": 0, "failed": 0}
    if settings.TEMP_PATH.exists(): shutil.rmtree(settings.TEMP_PATH)
    settings.TEMP_PATH.mkdir()

    for file in files:
        safe_name = processor.sanitize_filename(file.filename)
        temp_path = settings.TEMP_PATH / safe_name
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if safe_name.endswith(".zip"):
                with zipfile.ZipFile(temp_path, 'r') as z:
                    z.extractall(settings.TEMP_PATH / "extracted")
                for root, _, fnames in os.walk(settings.TEMP_PATH / "extracted"):
                    for fname in fnames:
                        if fname.endswith(".pdf"):
                            src = Path(root) / fname
                            dest = settings.STORAGE_PATH / processor.sanitize_filename(fname)
                            shutil.move(str(src), str(dest))
                            if db.add_document(dest.name, processor.chunk_text(processor.extract_text(dest))):
                                stats["processed"] += 1
            elif safe_name.endswith(".pdf"):
                dest = settings.STORAGE_PATH / safe_name
                shutil.copy(temp_path, dest)
                if db.add_document(safe_name, processor.chunk_text(processor.extract_text(dest))):
                    stats["processed"] += 1
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Upload error: {e}")

    shutil.rmtree(settings.TEMP_PATH)
    return {"message": "Complete", "stats": stats}

@app.post("/create-resume")
async def create_resume(data: ResumeFormData):
    base_name = f"{data.full_name}_{data.job_title}".replace(" ", "_")
    safe_name = processor.sanitize_filename(f"{base_name}.pdf")
    dest_path = settings.STORAGE_PATH / safe_name
    try:
        PDFGenerator.create_resume_pdf(data, dest_path)
        full_text = f"Name: {data.full_name}\nRole: {data.job_title}\nSkills: {data.skills}\nSummary: {data.summary}\nExp: {data.experience}\nEdu: {data.education}"
        if db.add_document(safe_name, processor.chunk_text(full_text)):
            return {"message": "Resume created", "filename": safe_name}
        raise HTTPException(500, "Indexing failed")
    except Exception as e:
        logger.error(f"Creation failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/search")
def search_resumes(query: str):
    return {"results": db.search(query), "count": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend:app", host="0.0.0.0", port=8080, reload=True)