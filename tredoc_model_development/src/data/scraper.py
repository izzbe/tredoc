from dotenv import load_dotenv
import os
import requests
import subprocess
import time
import shutil
import psycopg2
import ast

import sys
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"}

REPO_DIR = "/tmp/tredoc_repo_clone"

def get_repo_urls(min_stars=100, pages=5, per_page=100):
    urls = []
    for page in range(1, pages + 1):
        res = requests.get(
            "https://api.github.com/search/repositories",
            headers=HEADERS,
            params={
                "q": f"language:python stars:>{min_stars}",
                "sort": "stars",
                "per_page": per_page,
                "page": page
            }
        )
        items = res.json().get("items", [])
        urls.extend([(r["clone_url"], r["stargazers_count"]) for r in items])
        time.sleep(1)
    return urls

def clone_repo(url):
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, REPO_DIR],
        capture_output=True
    )
    if result.returncode != 0:
        raise Exception(f"Failed to clone {url}: {result.stderr.decode()}")

def delete_repo():
    shutil.rmtree(REPO_DIR)
    
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST")
    )

def insert_repo(conn, clone_url, stars):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO repos (clone_url, stars)
            VALUES (%s, %s)
            ON CONFLICT (clone_url) DO NOTHING
            RETURNING id
        """, [clone_url, stars])
        result = cur.fetchone()
    conn.commit()
    return result[0] if result else None

def find_repo(conn, clone_url):
    with conn.cursor() as cur:
        cur.execute("SELECT id, clone_url FROM repos WHERE clone_url=%s", [clone_url])
        return cur.fetchone()

def insert_pair(conn, repo_id, file_path, func_name, signature, body, docstring, style):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pairs (repo_id, file_path, func_name, signature, body, docstring, style)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_id, file_path, func_name) DO NOTHING
        """, [repo_id, file_path, func_name, signature, body, docstring, style])
    conn.commit()
    
def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            
            pairs.append({
                "file_path": file_path,
                "func_name": node.name,
                "signature": f"def {node.name}({ast.unparse(node.args)})",
                "body": ast.unparse(node),
                "docstring": docstring,
            })
    
    return pairs

def parse_repo(repo_dir):
    all_pairs = []
    for root, dirs, files in os.walk(repo_dir):
        # skip common junk directories
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
        
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)
            pairs = parse_file(file_path)
            all_pairs.extend(pairs)
    
    return all_pairs

def detect_style(docstring):
    if "Args:" in docstring or "Returns:" in docstring or "Raises:" in docstring:
        return "google"
    if "Parameters\n----------" in docstring or "Returns\n-------" in docstring:
        return "numpy"
    if ":param" in docstring or ":type" in docstring or ":rtype" in docstring:
        return "sphinx"
    return "plain"

def process_repo(conn, clone_url, stars):
    py_files = [f for root, dirs, files in os.walk(REPO_DIR) for f in files if f.endswith(".py")]
    if len(py_files) < 10:
        print(f"Skipping {clone_url}: only {len(py_files)} Python files")
        return
    
    print(f"Processing repo from {clone_url}")
    if find_repo(conn, clone_url):
        print(f"ERROR: Repo already exists in table repos, url={clone_url}")
        return
    
    repo_id = insert_repo(conn, clone_url, stars)
    print(f"Added repo into table repos")
    
    pairs = parse_repo(REPO_DIR)
    print(f"Parsing repo finished")

    for pair in pairs:
        insert_pair(conn, repo_id, pair["file_path"], pair["func_name"], pair["signature"],
                    pair["body"], pair["docstring"], detect_style(pair["docstring"]))
    print(f"Inserted all pairs for repo")
    

def main():
    print("Starting scraper...")
    urls = get_repo_urls(100, 10, 1000)
    print(f"Found {len(urls)} repos")
    
    conn = get_conn()
    
    for url, stars in urls:
        try:
            clone_repo(url)
            process_repo(conn, url, stars)
        except Exception as e:
            print(f"Failed to process {url}: {e}")
        finally:
            if os.path.exists(REPO_DIR):
                delete_repo()
    
    conn.close()
        
    
if __name__ == "__main__":
    main()
