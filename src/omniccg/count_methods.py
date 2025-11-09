import os, re

METHOD_REGEX = re.compile(
    r"""
    ^\s*                                        # ⬅ anchor to line start
    (?:(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*
    (?:<[^>;{()]*>\s*)?
    (?!class\b|interface\b|enum\b|record\b)
    (?:
        [A-Za-z_][$\w]* (?:\s*\.\s*[A-Za-z_][$\w]*)*
        (?:\s*<[^>;{}()]*>)?
        (?:\s*\[\s*\])*
        | void
    )
    \s+
    ([A-Za-z_][$\w]*)                            # method name
    \s*
    \(
        [^()]* (?:\([^()]*\)[^()]*)*
    \)
    \s*
    (?:throws\s+[A-Za-z_][$\w]*(?:\s*\.\s*[A-Za-z_][$\w]*)*(?:\s*,\s*[A-Za-z_][$\w]*(?:\s*\.\s*[A-Za-z_][$\w]*)*)*)?
    \s*
    (?:\{|\;)                                    # body or abstract/interface
    """,
    re.VERBOSE | re.MULTILINE
)

LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
STRING_LIT  = re.compile(r"\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^\'\\])*\'")

def _strip_comments_and_strings(code: str) -> str:
    code = STRING_LIT.sub('""', code)
    code = BLOCK_COMMENT.sub("", code)
    code = LINE_COMMENT.sub("", code)
    return code

def count_java_methods_in_file(java_file_path: str) -> int:
    if not (java_file_path.lower().endswith(".java") and os.path.isfile(java_file_path)):
        raise ValueError("Please provide a valid path to a .java file.")
    with open(java_file_path, "r", encoding="utf-8", errors="ignore") as fh:
        code = fh.read()
    cleaned = _strip_comments_and_strings(code)
    return sum(1 for _ in METHOD_REGEX.finditer(cleaned))
