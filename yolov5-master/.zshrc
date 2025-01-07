# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/karthiknutulapati/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/karthiknutulapati/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/karthiknutulapati/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/karthiknutulapati/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# PYENV Initialization
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Prioritize pyenv's Python over system Python
export PATH="$PYENV_ROOT/shims:$PATH"

# Ensure Homebrew Python is accessible if needed
export PATH="/usr/local/opt/python@3.11/bin:$PATH"

