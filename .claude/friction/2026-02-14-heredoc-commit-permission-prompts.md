# HEREDOC Commit Messages Require Permission Prompts

**Date:** 2026-02-14
**Agent/User:** Claude Sonnet 4.5
**Severity:** Low

## Problem
Git commit commands using HEREDOC syntax for multi-line messages consistently prompt for permission approval, despite having `Bash(git commit *)` in the allow list.

The pattern that doesn't match automatically:
```bash
git commit -m "$(cat <<'EOF'
Multi-line commit message here

- Bullet points
- More details

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

## Steps to Reproduce
1. Have `Bash(git commit *)` in `.claude/settings.json` allow list
2. Run a git commit command with HEREDOC message syntax
3. Observe permission prompt despite the wildcard pattern

## Workaround
- Accept permission prompts manually each time
- Add specific HEREDOC pattern to settings.local.json (but this is session-specific)
- Alternative: Use simpler commit messages without HEREDOC (loses formatting)

## Suggested Fix
The permission matching system should:
1. Properly handle command substitution `$(...)` in pattern matching
2. Match `Bash(git commit *)` against all forms of git commit, including those with complex shell syntax
3. Document permission pattern matching behavior for complex shell constructs (HEREDOC, command substitution, etc.)

This may be a Claude Code permission system limitation worth reporting to the Claude Code team.

## Impact
- Minor workflow friction (extra clicks)
- Discourages well-formatted multi-line commit messages
- Creates inconsistency between what "should" be allowed and what actually is
