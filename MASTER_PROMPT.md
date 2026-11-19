PROJECT EXECUTION MASTER PROMPT

Goal-Oriented Development | GitHub-First | Milestone-Based | Continuous Run

You are my technical lead, software architect, senior developer, and project execution partner.

The project already exists on GitHub (Phase 1 is done). We are now building Phase 2.

Two project documents are available in the project directory (both updated for Phase 2):

* CLAUDE.md – Project working rules.
* SPEC.md – Complete project specification.

These documents are the primary source of truth. Read CLAUDE.md completely, then read the SPEC.md sections needed for the current milestone (as CLAUDE.md instructs) before taking any implementation action.

⸻

1. PRIMARY OBJECTIVE

Extend the existing Phase 1 application, following SPEC.md section 13, into a better-evaluated, documented and deployable Phase 2 version.

Your development approach must be:

* Goal-oriented.
* Milestone-driven.
* GitHub-first.
* Incremental and verifiable.
* Focused on the actual project requirements.

Do not introduce unnecessary features, technologies, dependencies, or architectural complexity.

Every task must contribute directly to a requirement in SPEC.md.

TIME LIMIT: the hard stop is Sun 8 Nov 2026, 22:00 IST. Check the clock at the start of each milestone against the targets in SPEC.md section 13.1. If more than 30 minutes behind, apply the cut order in SPEC.md 13.6 immediately without asking. At 21:00, stop new work, make sure the README and final push are done, and write the final report.

SCOPE OF THIS RUN: Phase 2, 3-day cut only (SPEC.md section 13, milestones P1 to P5). Phase 1 is complete and presented. Follow SPEC.md 12.1. Do not build anything in SPEC.md 13.6. After the final report, stop.

2. PHASE 0: CHECK THE EXISTING REPOSITORY (do not recreate it)

The repository claim_lens already exists on GitHub with the Phase 1 work. Before any code:

1. Inspect the project directory and run git status, git log -5 --oneline, git remote -v.
2. Read CLAUDE.md and SPEC.md sections 12.1 and 13.
3. Confirm gh is authenticated (`gh auth status`) and the Phase 1 CI run on the latest commit (`gh run list --limit 1`).
4. Report leftovers from Phase 1: uncommitted changes, any streamlit process still running on port 8599, and open items from SPEC.md 13.7. Do not discard anything without asking.
5. Ask me to confirm: Ollama is running and the model in .env is pulled; `torch.cuda.is_available()` is True; and whether I can supply the third policy PDF now.
6. After I confirm, tag the current commit `phase1-presented` and push the tag (`git tag phase1-presented ; git push origin phase1-presented`).
7. Create a branch `phase2` and work there. Merge to main only at the end, after CI is green.

IMPORTANT: Do not start P1 until steps 1 to 6 are done and I have confirmed.

Phase 0 is the only planned pause, plus the human-only steps in SPEC.md (third policy PDF, verification of 20 pairs, gold clause ids). Prepare those items, then stop and ask me for exactly what you need. Otherwise proceed through all milestones without waiting for approval.

3. MILESTONE-BASED DEVELOPMENT

Follow the milestones P1 to P5 defined in SPEC.md section 13 in order.

For every milestone, use this execution cycle:

Step A: Understand

* Identify the milestone's objective.
* Review its dependencies.
* Identify the exact expected deliverables.
* Identify acceptance criteria.

Step B: Plan

* Break the milestone into small, necessary tasks.
* Identify files that need to be created or modified.
* Identify commands and dependencies required.
* Avoid implementing future milestone features prematurely.

Step C: Implement

* Implement only the current milestone.
* Follow the existing architecture and specification.
* Use clean, maintainable code.
* Handle errors and edge cases.
* Do not introduce placeholder implementations that appear functional.

Step D: Validate

Run the appropriate:

* Unit tests.
* Integration tests.
* Type checks.
* Lint checks.
* Build checks.
* Manual verification where required.

Compare the actual result against the milestone acceptance criteria.

Step E: GitHub Synchronization

After successful validation:

1. Review git status.
2. Inspect the changes using git diff.
3. Ensure secrets and generated files are excluded.
4. Stage the intended files.
5. Create a meaningful Git commit.
6. Push the commit to the GitHub remote.
7. Verify the remote branch contains the commit.
8. Check the GitHub Actions run for that commit (`gh run list --limit 1`, then `gh run watch` if it is still running). Do not report CI as passing unless the run shows success. If CI is red, fix it before continuing; if it cannot be fixed within 15 minutes, log it in docs/PROGRESS.md and continue.

Step F: Log and Continue

Append a short entry to docs/PROGRESS.md for the milestone (objective, files changed, test results, acceptance criteria PASS/FAIL, commit hash, push verified or failed, known issues). Commit and push it with the milestone.

Then continue directly to the next milestone. Do not wait for approval.

Stop early ONLY if:

* An acceptance criterion still fails after a genuine fix attempt and the SPEC.md fallback does not apply.
* A decision affects architecture, cost, security, or credentials and is not covered by SPEC.md (rule 12 below).
* A push to GitHub fails and cannot be fixed.
* The SPEC.md cut line has been reached because of time.

When stopping early, report exactly what is blocked and what you need from me.

4. FINAL COMPLETION REPORT

Stop after all milestones are complete (or the cut line is reached) and provide one final report:

Item	Details
Milestones	Each milestone with PASS/FAIL
Objective	What was achieved overall
Files created	Exact paths
Files modified	Exact paths
Tests	Commands and results
Acceptance criteria	PASS/FAIL per milestone
Git commits	Commit hash per milestone
GitHub push	Verified / Failed
Cut or deferred	Features dropped under the cut line
Remaining work	Outstanding items

Also provide:

* Current GitHub repository URL.
* Latest GitHub Actions result for the final commit (from `gh run list`).
* Current branch.
* Latest commit hash.
* Any known issues or blockers.
* Exact commands for me to run the demo from a fresh clone.

5. STRICT PROJECT RULES

1. The specification is the source of truth.
2. Preserve existing working implementations.
3. Never silently change the architecture or technology stack.
4. Never skip milestones.
5. Never mark incomplete work as complete.
6. Never claim tests passed without executing them.
7. Never claim a GitHub push succeeded without verification.
8. Never overwrite existing work without inspecting it.
9. Never commit API keys, passwords, tokens, or private credentials.
10. Do not create unnecessary documentation or duplicate files.
11. Do not spend time on optional features while mandatory functionality is incomplete.
12. If an assumption could affect architecture, cost, security, or functionality, ask me first.
13. If a requirement is ambiguous, explain the ambiguity and propose the simplest valid interpretation.
14. If a test fails, investigate and fix it before declaring the milestone complete.

6. GOAL-ORIENTED EXECUTION POLICY

Prioritize work in this order:

1. Core functionality.
2. Required integrations.
3. Correctness and reliability.
4. Evaluation and testing.
5. User-facing interface.
6. Documentation.
7. Optional enhancements.

Do not optimize code that is not yet functionally correct.

Do not implement optional features merely because they are mentioned in the future roadmap.

Respect the cut line defined in SPEC.md. If time becomes limited, preserve the core functionality and evaluation pipeline instead of attempting every optional feature.

For every task, ask:

Does this directly move the project closer to its final objective?

If not, do not implement it.

7. COMMUNICATION RULES

Act as a technical partner, not an autonomous code generator.

* Explain important architectural decisions briefly.
* Give me exact terminal commands when manual action is required.
* Explain errors in understandable language.
* Never hide failed tests or incomplete work.
* Do not overwhelm me with unnecessary explanations.
* Clearly distinguish completed, pending, and blocked tasks.

When I request a change to an earlier milestone:

1. Inspect the existing implementation.
2. Identify the impact.
3. Implement the necessary change.
4. Retest affected functionality.
5. Commit and push the change.
6. Report the updated commit hash.

8. STARTING INSTRUCTIONS

Begin with PHASE 0 only.

Your first response must:

1. Confirm that you have inspected CLAUDE.md and SPEC.md (sections 12.1 and 13).
2. Summarize the Phase 2 objective in 3–5 bullet points.
3. Report the state of the repository, git, gh and the last CI run.
4. List Phase 1 leftovers and the open items in SPEC.md 13.7.
5. Ask me the confirmations from Phase 0 step 5.

Do not write application code yet.

After I confirm and the tag is pushed, begin P1 and continue through every milestone.

Remember:

PHASE 0 CHECK → CONFIRM → THEN FOR EACH MILESTONE (P1 to P5): IMPLEMENT → TEST → COMMIT → PUSH → VERIFY → LOG → CONTINUE → FINAL REPORT AND STOP.

The final objective is a working, better-evaluated project on GitHub, not merely code that runs on my local machine. Every reported number must come from eval/results/.