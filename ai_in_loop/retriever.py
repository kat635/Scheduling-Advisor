"""Document retrieval with BM25 keyword search."""

from pathlib import Path
from typing import Optional
import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from .config import Config

_retriever: Optional[BM25Retriever] = None
_is_initialized: bool = False


def get_retriever(cfg: Config) -> Optional[BM25Retriever]:
    """Get or create BM25 retriever (singleton). Returns None if no docs."""
    global _retriever, _is_initialized

    if _is_initialized:
        return _retriever

    _is_initialized = True
    resources_dir = Path(cfg.resources_dir)

    if not resources_dir.exists():
        print(f"Info: '{resources_dir}' not found. Document search disabled.", file=sys.stderr)
        return None

    documents = []

    # Load .txt files
    txt_loader = DirectoryLoader(str(resources_dir), glob="**/*.txt", loader_cls=TextLoader)
    try:
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Warning: Error loading .txt files: {e}", file=sys.stderr)

    # Load .pdf files (if pypdf available)
    if HAS_PYPDF:
        pdf_loader = DirectoryLoader(str(resources_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
        try:
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Warning: Error loading .pdf files: {e}", file=sys.stderr)

    if not documents:
        print(f"Info: No documents in '{resources_dir}'. Search disabled.", file=sys.stderr)
        return None

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # Create BM25 retriever
    _retriever = BM25Retriever.from_documents(chunks, k=3)
    print(f"Loaded {len(chunks)} chunks from {len(documents)} documents", file=sys.stderr)

    return _retriever


def reset_retriever() -> None:
    """Reset singleton (for testing)."""
    global _retriever, _is_initialized
    _retriever = None
    _is_initialized = False
