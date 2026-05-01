#!/usr/bin/env bash
set +e

cd "$(dirname "$0")/.."

git config user.name "${GIT_AUTOPUSH_USER_NAME:-Codex}"
git config user.email "${GIT_AUTOPUSH_USER_EMAIL:-codex@localhost}"

base_branch="$(git branch --show-current)"
safe_branch="${base_branch//\//-}"
push_branch="${GIT_AUTOPUSH_BRANCH:-autopush/${safe_branch:-detached}}"

end=$(( $(date +%s) + 5 * 60 * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) auto git checkpoint ====="
    echo "push_branch:${push_branch}"
    git add -A
    if git diff --cached --quiet; then
        echo "no staged changes"
    else
        git commit -m "auto checkpoint $(date -u +%Y%m%dT%H%M%SZ)"
    fi
    git push origin "HEAD:refs/heads/${push_branch}"
    sleep 300
done
