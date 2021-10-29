proj_name=$1
git_path=".git"

if [ -d "$git_path" ]; then  
    rm -rf "$git_path"  
fi

git init .
git add .
git commit -m "Initial commit"