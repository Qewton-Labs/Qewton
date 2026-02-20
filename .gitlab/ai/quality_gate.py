import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import git
import time
import requests
import openai

COVERAGE_THRESHOLD = float(os.environ.get("COVERAGE_THRESHOLD", 80))

TEST_GENERATION_PROMPT = """
You are a senior Python QA engineer.
Write pytest tests for the following code.
Focus on public functions and edge cases.
Return only valid Python code. Here is the code:
"""

DOCSTRING_CHECK_PROMPT = """
Review the given Python module.
Check all public functions for proper docstring.
Suggest improvements if missing or incomplete.
Return only updated docstring if needed. Here is the code:
"""


print("AI Quality Gate starting...")

# --- Parse coverage.xml ---
cov_file = Path("coverage.xml")
if not cov_file.exists():
    print("coverage.xml not found, skipping AI checks")
    sys.exit(0)

tree = ET.parse(cov_file)
root = tree.getroot()

low_coverage_files = []
for class_el in root.findall(".//class"):
    filename = class_el.get("filename")
    line_rate = float(class_el.get("line-rate", 0)) * 100

    if line_rate < COVERAGE_THRESHOLD:
        low_coverage_files.append((filename, line_rate))

if low_coverage_files:
    print("Files below coverage threshold:", low_coverage_files)
else:
    print("All files meet coverage threshold")

# --- Detect changed files ---
repo = git.Repo(".")
changed_files = [item.a_path for item in repo.index.diff(None)]
print("Changed files:", changed_files)

# --- Combine files to check ---
# Extract only filenames from low_coverage_files tuples
low_coverage_paths = [filename for filename, _ in low_coverage_files]

all_files_to_process = set(low_coverage_paths + changed_files)

files_to_process = set(
    file for file in all_files_to_process if not file.endswith("__init__.py")
)

if not files_to_process:
    print("No files to process, exiting")
    sys.exit(0)

# --- Check API key ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not set — AI checks skipped")
    sys.exit(0)

# --- Prepare branch for AI changes ---
ai_branch = f"ai/auto-tests-docs-{int(time.time())}"
repo.git.checkout("HEAD", b=ai_branch)

# --- AI Processing ---
repo = git.Repo(".")
repo_root = Path(repo.working_tree_dir)  # absolute path to repo root
print("Current working directory:", Path.cwd())
print("Repo root:", repo_root)
print("Files in repo_root/pioneer/pipelines:")
print(os.listdir(repo_root / "pioneer" / "pipelines"))
for f in files_to_process:
    f_path = repo_root / f
    print(f_path, Path(f_path).exists())
    if not Path(f_path).exists() or not f.endswith(".py"):
        continue
    with open(f_path) as src_file:
        code = src_file.read()
    print("here")
    # --- Test generation for low-coverage files ---
    if f in low_coverage_paths:
        prompt_tests = f"{TEST_GENERATION_PROMPT}\n{code}"
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_tests}],
            api_key=api_key,
        )
        test_code = resp.choices[0].message.content
        test_file = Path("tests") / f"test_{Path(f).stem}.py"
        test_file.write_text(test_code)
        repo.index.add([str(test_file)])
        print(f"Generated tests for {f} -> {test_file}")

    # --- Docstring improvement for changed files ---
    if f in changed_files:
        prompt_docs = f"{DOCSTRING_CHECK_PROMPT}\n{code}"
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_docs}],
            api_key=api_key,
        )
        improved_code = resp.choices[0].message.content
        Path(f).write_text(improved_code)
        repo.index.add([f])
        print(f"Improved docstring for {f}")

# ----------------- Commit and push branch -----------------
commit_message = "AI: auto-generated tests and improved docstrings"
repo.index.commit(commit_message)
origin = repo.remote(name="origin")

try:
    push_info = origin.push(refspec=f"{ai_branch}:{ai_branch}")
    if any(pi.flags & pi.ERROR for pi in push_info):
        print("Error pushing branch:", push_info)
        sys.exit(1)
except Exception as e:
    print("Failed to push branch:", e)
    sys.exit(1)

# ----------------- Create merge request -----------------
project_id = os.environ.get("CI_PROJECT_ID")
gl_token = os.environ.get("GITLAB_TOKEN")
if not project_id or not gl_token:
    print("CI_PROJECT_ID or GITLAB_TOKEN not set — cannot create MR")
    sys.exit(0)

mr_title = "AI: Auto-generated tests and docstring improvements"
mr_desc = (
    "This MR contains AI-generated tests for low-coverage files "
    "and improved docstrings for changed files."
)

headers = {"PRIVATE-TOKEN": gl_token}
mr_url = f"https://gitlab.com/api/v4/projects/{project_id}/merge_requests"

# Check if MR already exists
existing_mrs = requests.get(
    mr_url,
    headers=headers,
    params={"source_branch": ai_branch, "target_branch": "main", "state": "opened"},
    timeout=200,
).json()

if existing_mrs:
    print("Merge request already exists:", existing_mrs[0]["web_url"])
else:
    resp = requests.post(
        mr_url,
        headers=headers,
        data={
            "source_branch": ai_branch,
            "target_branch": "main",
            "title": mr_title,
            "description": mr_desc,
            "remove_source_branch": True,
        },
        timeout=200,
    )
    if resp.status_code == 201:
        print("Merge request created:", resp.json().get("web_url"))
    else:
        print("Failed to create merge request:", resp.text)
        sys.exit(1)
