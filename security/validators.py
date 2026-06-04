import re
from pathlib import Path
from config import MAX_FILE_SIZE_MB, MAX_QUERY_SIZE

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(ignore|disregard)\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"role[:=]\s*", re.IGNORECASE)
]

class UnsafeInputError(Exception):
    pass

class FileValidationError(Exception):
    pass

def validate_query(query):
    if not isinstance(query, str):
        raise UnsafeInputError("The query should be a string.")

    query = query.strip()

    if not query:
        raise UnsafeInputError("The query is empty.")

    if len(query) > MAX_QUERY_SIZE:
        raise UnsafeInputError(f"Query size limit exceeded. Max Size is {MAX_QUERY_SIZE} characters.")
        
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            raise UnsafeInputError("Prompt injection attack is detected in the query.")
        
    return query

def validate_file(file: str | Path):
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError
    
    if not file.is_file():
        raise FileValidationError("The path is not a file.")
    
    if file.suffix.lower() != ".pdf":
        raise FileValidationError("Only .pdf files are allowed.")
    
    with file.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise FileValidationError("Invalid PDF file.")
    
    file_size_mb = (file.stat().st_size) / 1000 / 1000

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise FileValidationError(f"File size limit exceeded. Max size is {MAX_FILE_SIZE_MB} mb.")
    
    return file