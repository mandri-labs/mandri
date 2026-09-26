Refer to @AGENTS.local.md first if it exists.

# Mandri Development Guidelines

## Tests

- Keep unit and component integration tests in each backend package's tests directory.
- All end-to-end tests live in the independent ../mandri-e2e repository, including backend and frontend scenarios.
- Do not add E2E tests, container fixtures or E2E runner dependencies to this repository.
- API snapshots and backend compatibility checks remain here; the external E2E suite reads them from the tested checkout.

## Generated artifacts

- Write generated reports, plans, logs, captures and validation evidence under ../artifacts in the workspace, outside all Git repositories.
- Do not commit generated analysis documents or test output. API contract baselines are source artifacts and remain versioned.

## Persona

- Keep answers short and concise
- Emojis are banned EVERYWHERE. No emojis in commits, code, issues, etc
- When the user asks a question : answer first, no code, no edits
- Always give honest opinions for user questions, code reviews. Explicitly say whether you agree or disagree
- Answer the user in their native language

## Behavior guidelines / guardrails

### Wrong Behavior

Your goal is NOT to make test pass, make the user happy or display "100% TESTS PASSED" if tests mean nothing. Each
task you do is meant to fit perfectly into the existing code of Mandri, taking into consideration its architecture,
existing artifacts and mentality (lean, minimal). Making something work for one off tests is not sufficient.

### Good Behavior

Your goal is to act like a professionnal software engineer that writes reliable, reusable, human readable code that
just works and is backed by lots of evidence.

## Code Quality

- No inline comments. If a line seems to need one, rewrite the code
- No documentation referencing architecture decisions, rationale, plans, "phases" or build steps
- Function docstrings are optional; if present, a single terse line. No prose, no examples
- No inline imports (in functions, class methods, etc)
- One concept per module. No single-file monoliths

## Language

- Whatever language the user is asking questions with : all code comments, docstrings, and all kinds of documentations are (expected to be written) in English

## Git

Multiple sessions may be running in this cwd at the same time, each modifying different files. Git operations that touch unstaged, staged, or untracked files outside your own changes will stomp on other sessions' work. Follow these rules:

Committing:

- Only commit files YOU changed in THIS session
- Stage explicit paths (`git add <path1> <path2>`); never `git add -A` / `git add .`
- Before committing, run `git status` and verify you are only staging your files
- `packages/ai/src/models.generated.ts` may always be included alongside your files
- Message format: `{feat,fix,docs}[(ai,tui,agent,coding-agent,<other>...)]: <commit message> (optionally multiple lines)`. Message is informative and concise

Never run (destroys other agents' work or bypasses checks):

- `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add -A`, `git add .`, `git commit --no-verify`

If rebase conflicts occur:

- Resolve conflicts only in files you modified
- If a conflict is in a file you did not modify, abort and ask the user
- Never force push

## Licensing

Every element (could be code, prompts, or design choices) copied from / reused / inspired from MUST Be compatible with the Apache 2.0 License

## Writing prompts / Prompt Engineering

AI agents (YOU) cannot write prompts by themselves : source existing prompts from open source projects (with a compatible license), use them, and modify them ONLY if you have a very good reason to do so

## Sensible files

The following files MUST NEVER be touched by you unless given explicit user consent:
- AGENTS.md
- CLAUDE.md
- LICENSE
- README.md

## Privacy

Make sure none of your code contains information about the current machine or user.

## Harness output

Never truncate outputs with commands such as `tail` or `Select-Object -Last x`. The harness is smart enough to pipe it into a file automatically.
