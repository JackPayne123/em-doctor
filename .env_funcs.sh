longrun() { 
    LOGS_DIR="logs"
    mkdir -p "$LOGS_DIR"                         # Create logs folder if missing
    timestamp=$(date +"%Y%m%d_%H%M%S")
    cmdname=$(echo "$1" | tr " " "_")
    [ -z "$cmdname" ] && cmdname="run"
    "$@" 2>&1 | tee "$LOGS_DIR/${PWD##*/}_${cmdname}_${timestamp}.log" && echo -e "\a"
}

gitpush() {
    VENV_NAME=".venv"
    if [ -f "$VENV_NAME/.gh_token" ]; then
        GH_TOKEN=$(cat "$VENV_NAME/.gh_token")

        # Extract owner/repo from origin
        ORIGIN_URL=$(git remote get-url origin)
        OWNER_REPO=$(echo "$ORIGIN_URL" | sed -E 's|https://[^/]+/([^/]+/[^.]+)(\.git)?|\1|')

        # Reset origin with token
        git remote set-url origin "https://${GH_TOKEN}@github.com/${OWNER_REPO}.git"
    fi

    datetime=$(date +"%Y-%m-%d_%H-%M-%S")
    git add .
    git commit -m "${datetime} commit"
    git push
    echo "✅ Changes pushed with message: ${datetime} commit"
}
