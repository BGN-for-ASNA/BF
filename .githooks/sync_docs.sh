#!/bin/bash

# Dynamically resolve paths relative to the script location
# Since this script lives in BF/.githooks/, its parent directory is the BF root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BF_MCP_DIR="$(cd "$BF_DIR/../BF_mcp" 2>/dev/null && pwd)"

SRC_DOCS="$BF_DIR/Documentation/"
DST_DOCS="$BF_MCP_DIR/mcp_server/Documentation/"
CACHE_FILE="$BF_MCP_DIR/_rag_cache.pkl"

# Rendered (quarto) documentation site.
#   RENDERED_SRC : quarto output dir (Documentation/_site). The trailing
#                  slash makes rsync copy its *contents* into RENDERED_DST.
#   RENDERED_DST : where the served/rendered copy lives (override with
#                  BF_DOCS_SITE_DIR); defaults to /home/sosa/work/website/BF
#   BF_SYNC_RENDER=1 : run `quarto render` first (incremental via
#                  `freeze: auto`). Off by default - a full render can
#                  execute notebooks and is too slow for a commit hook.
RENDERED_SRC="$BF_DIR/Documentation/_site/"
RENDERED_DST="${BF_DOCS_SITE_DIR:-/home/sosa/work/website/BF}"

# Colors for nice terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}[BF-Sync] Running documentation synchronization helper...${NC}"

# BF_mcp is optional: if it is missing we still refresh the rendered site.
SKIP_MCP=false
if [ -z "$BF_MCP_DIR" ] || [ ! -d "$BF_MCP_DIR" ]; then
    echo -e "${YELLOW}[BF-Sync] BF_mcp not found sibling to $BF_DIR - skipping MCP doc sync.${NC}"
    SKIP_MCP=true
fi

# Check the mode (commit vs merge)
MODE=$1
SHOULD_SYNC=false

if [ "$MODE" = "--commit" ]; then
    # Check if Documentation/ changed in the last commit
    if git -C "$BF_DIR" diff-tree --no-commit-id --name-only -r HEAD | grep -E '^Documentation/' > /dev/null; then
        echo -e "${YELLOW}[BF-Sync] Documentation changes detected in local commit.${NC}"
        SHOULD_SYNC=true
    fi
elif [ "$MODE" = "--merge" ]; then
    # Check if Documentation/ changed in the merge/pull
    if git -C "$BF_DIR" diff-tree --no-commit-id --name-only -r ORIG_HEAD HEAD | grep -E '^Documentation/' > /dev/null; then
        echo -e "${YELLOW}[BF-Sync] Documentation changes detected in merge/pull.${NC}"
        SHOULD_SYNC=true
    fi
else
    # Default: sync unconditionally
    echo -e "${YELLOW}[BF-Sync] Running in manual synchronization mode.${NC}"
    SHOULD_SYNC=true
fi

if [ "$SHOULD_SYNC" = true ]; then
    if [ ! -d "$SRC_DOCS" ]; then
        echo -e "${RED}[BF-Sync] Error: Source documentation directory $SRC_DOCS does not exist!${NC}"
        exit 1
    fi

    # -------------------------------------------------------------------
    # 1. Sync the raw docs into BF_mcp (for the RAG server)
    # -------------------------------------------------------------------
    if [ "$SKIP_MCP" != true ]; then
        if [ ! -d "$DST_DOCS" ]; then
            echo -e "${YELLOW}[BF-Sync] Target directory does not exist, creating it: $DST_DOCS${NC}"
            mkdir -p "$DST_DOCS"
        fi

        echo -e "${BLUE}[BF-Sync] Syncing files via rsync...${NC}"
        rsync -av --delete \
            --exclude="/.quarto/" \
            --exclude="**/*.quarto_ipynb" \
            --exclude="*.pkl" \
            --exclude=".git/" \
            --exclude="*(conflicted copy*" \
            "$SRC_DOCS" "$DST_DOCS"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[BF-Sync] Files synchronized successfully!${NC}"
        else
            echo -e "${RED}[BF-Sync] Error: rsync failed!${NC}"
            exit 1
        fi

        # Invalidate RAG cache so it rebuilds on next access
        if [ -f "$CACHE_FILE" ]; then
            echo -e "${BLUE}[BF-Sync] Invalidating RAG cache by deleting $CACHE_FILE...${NC}"
            rm -f "$CACHE_FILE"
            echo -e "${GREEN}[BF-Sync] RAG cache successfully invalidated.${NC}"
        fi

        # Commit changes in BF_mcp if it has a .git repo.
        # Git runs hooks with GIT_DIR (and GIT_INDEX_FILE, ...) exported for
        # the BF repo: without unsetting them, the `git add/commit` below
        # would stage mcp_server/Documentation/ into BF itself and commit it
        # there (this is how 674aea44 added 450 mcp_server/ files to BF).
        unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_COMMON_DIR
        if [ -d "$BF_MCP_DIR/.git" ]; then
            echo -e "${BLUE}[BF-Sync] Checking for staging changes in BF_mcp...${NC}"
            cd "$BF_MCP_DIR" || exit 1
            git add mcp_server/Documentation/

            if ! git diff --cached --quiet; then
                BF_COMMIT_INFO=$(git -C "$BF_DIR" log -1 --format="%h: %s")
                echo -e "${YELLOW}[BF-Sync] Committing changes in BF_mcp: Sync documentation from BF commit ($BF_COMMIT_INFO)${NC}"
                git commit -m "Sync documentation from BF commit ($BF_COMMIT_INFO)"
                echo -e "${GREEN}[BF-Sync] Successfully committed documentation changes in BF_mcp.${NC}"
            else
                echo -e "${GREEN}[BF-Sync] No new documentation changes to commit in BF_mcp.${NC}"
            fi
        fi
    fi

    # -------------------------------------------------------------------
    # 2. Refresh the rendered documentation site ($RENDERED_DST)
    # -------------------------------------------------------------------
    if [ "${BF_SYNC_RENDER:-0}" = "1" ]; then
        if command -v quarto >/dev/null 2>&1; then
            echo -e "${BLUE}[BF-Sync] Rendering docs with quarto (incremental)...${NC}"
            if ! quarto render "$BF_DIR/Documentation"; then
                echo -e "${RED}[BF-Sync] Warning: quarto render failed; syncing existing _site as-is.${NC}"
            fi
        else
            echo -e "${YELLOW}[BF-Sync] BF_SYNC_RENDER=1 but 'quarto' not on PATH; skipping render.${NC}"
        fi
    fi

    if [ -d "$RENDERED_SRC" ]; then
        mkdir -p "$RENDERED_DST"
        echo -e "${BLUE}[BF-Sync] Syncing rendered site -> $RENDERED_DST ...${NC}"
        # Additive: copy the _site contents in, but do NOT --delete — the
        # target lives inside the website repo and may hold files not
        # produced by this render (e.g. previously built PDFs).
        rsync -av \
            --exclude="/.git/" \
            "$RENDERED_SRC" "$RENDERED_DST/"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[BF-Sync] Rendered documentation updated at $RENDERED_DST.${NC}"
        else
            echo -e "${RED}[BF-Sync] Error: rsync of rendered site failed!${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}[BF-Sync] No rendered site at $RENDERED_SRC.${NC}"
        echo -e "${YELLOW}[BF-Sync] Run 'quarto render Documentation' or re-run with BF_SYNC_RENDER=1. Skipping rendered sync.${NC}"
    fi
else
    echo -e "${GREEN}[BF-Sync] No documentation changes detected. Skipping synchronization. All clear!${NC}"
fi
