---
name: create-skill
description: "Workspace skill for creating VS Code Copilot SKILL.md files. Use when you want to define a reusable skill workflow, choose scope, and generate a valid skill file for the repository."
---

# Create Skill

This skill helps you create a new `SKILL.md` file for VS Code Copilot customization. It is designed as a workspace-scoped helper and follows the agent-customization workflow for skills.

## When to use

- You want to add a new reusable workflow skill to this repository.
- You need a valid `SKILL.md` file with `name` and `description` frontmatter.
- You want to choose whether the customization is workspace-scoped or user-scoped.

## Workflow

1. Identify the outcome.
   - What workflow should the new skill automate or guide?
   - What inputs and outputs should it produce?

2. Confirm scope.
   - Workspace-scoped: place under `.github/skills/<skill-name>/SKILL.md`
   - User-scoped: place under `{{VSCODE_USER_PROMPTS_FOLDER}}/` and use personal customization

3. Define the skill metadata.
   - `name`: the skill command name
   - `description`: concise, keyword-rich, user-facing discovery text
   - Optional: additional frontmatter fields if needed by the agent platform

4. Write the skill body.
   - Short introduction describing purpose and usage
   - Step-by-step guidance for the workflow
   - Decision points and quality checks
   - Examples for common prompts or invocations

5. Validate the skill.
   - Ensure YAML frontmatter is syntactically valid
   - Verify the `description` clearly includes trigger keywords
   - Confirm file path matches the chosen scope

## Quality checks

- `name` is unique and matches the folder or workflow.
- `description` is meaningful and discoverable.
- File path is correct for workspace skills.
- The skill solves a single, clearly defined workflow.

## Example prompts

- "Create a new repository skill for adding a code review checklist."
- "Help me write a SKILL.md for managing Python test file creation."
- "Generate a workspace skill that guides users through updating `.github` automation files."

## Notes

If you want to make this a user-level skill instead of workspace-level, move the file to the VS Code user prompts folder and update the description accordingly.
