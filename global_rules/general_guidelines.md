# General AI Interaction Guidelines

These are general principles for interacting with the AI assistant across projects.

## Communication

1.  **Clarity and Specificity:** Be precise in requests. Instead of "fix this," specify *what* needs fixing and *why* (e.g., "Refactor this function to use list comprehensions for better readability").
2.  **Provide Context:** Leverage Cursor's context awareness. Have relevant files open, place the cursor appropriately, and use `@` symbols for files (`@path/to/file.py`) and symbols (`@my_function`).
3.  **Iterative Refinement:** Expect to iterate. If the first response isn't perfect, provide feedback and ask for adjustments.
4.  **Ask Questions:** If unsure about AI suggestions, concepts, or how to proceed, ask for clarification.
5.  **State Assumptions:** If you have specific assumptions about the technology, desired outcome, or constraints, state them clearly.

## Code and Development

1.  **Adhere to Best Practices:** Request code that follows general software engineering principles (readability, maintainability, efficiency) and language/framework conventions.
2.  **Modularity:** Favor modular code design where appropriate.
3.  **Testing:** Encourage the generation of relevant tests (unit, integration) alongside features or fixes.
4.  **Documentation:** Request code comments for complex logic and docstrings for functions/classes. Assist with generating README content and other documentation.
5.  **Version Control:** Assume standard Git practices. Commit messages should be clear and concise.
6.  **Error Handling:** Implement robust error handling unless explicitly requested otherwise.

## Problem Solving

1.  **Break Down Problems:** Tackle complex tasks by breaking them into smaller, manageable steps.
2.  **Explore Alternatives:** Discuss different approaches or algorithms before settling on one, considering trade-offs.
3.  **Use Tools:** Leverage available tools (search, file reading, terminal) to gather information and perform actions, explaining the reasoning.
4.  **Verify Information:** If unsure about information or need up-to-date details, use web search or ask for verification.

## How to Use This File

This file serves as a reminder of effective interaction patterns. While the AI cannot directly read Cursor settings, you can refer to these principles in your prompts or periodically remind the AI of key preferences, especially at the start of a session. Project-specific guidelines should be placed in `behavior_rules/`. 