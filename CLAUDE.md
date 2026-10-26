# Instructions for Claude Code

## Core Operating Principles

1. **Think Before Coding:**
   - State explicit assumptions about tensor shapes (e.g., `(B, N, C)` vs. `(B, C, H, W)`), coordinate conventions (OpenCV vs. OpenGL/COLMAP), and data types before writing code.
   - For trivial, unambiguous tasks, act directly. For architectural changes or ambiguous interfaces, ask and clarify first.

2. **Simplicity First:**
   - Implement the minimal viable solution that solves the task directly.
   - Avoid speculative abstractions, unnecessary wrapper classes, helper modules, or extra dependencies.

3. **Surgical Changes:**
   - Modify strictly only the lines and functions relevant to the task.
   - Do not reformat untouched code, remove existing comments, rename variables, or alter imports outside the target scope.

4. **Goal-Driven Execution:**
   - Translate tasks into verifiable criteria (e.g., shape assertions, minimal test scripts, reproducible outputs).
   - A task is complete only when verified by successful execution or passing assertions, not merely by generating syntactically valid code.

