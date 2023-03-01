#!/bin/zsh

tmux start-server

[ -z "$(tmux list-sessions |grep skif-xrt)" ] \
&& echo "Creating session..." || { echo "Session found"; tmux attach -t skif-xrt; exit 0; }

tmux new-session -d -s skif-xrt -n editor -d "/bin/zsh -i"
tmux split-window -t skif-xrt:editor -h -d "/bin/zsh -i"
tmux select-pane -t skif-xrt:editor.1
tmux split-window -t skif-xrt:editor -d "/bin/zsh -i"

sleep 2

tmux send-keys -t skif-xrt:editor.0 "conda activate xrt && export PYTHONPATH=$PWD && nvim" C-m
tmux send-keys -t skif-xrt:editor.1 "htop" C-m
tmux send-keys -t skif-xrt:editor.2 "conda activate xrt && export PYTHONPATH=$PWD && clear" C-m

tmux select-pane -t skif-xrt:editor.0

tmux resize-pane -t skif-xrt:editor.0 -R 22
tmux resize-pane -t skif-xrt:editor.2 -U 9

tmux send-keys -t skif-xrt:editor.0 "\§"

tmux attach -t skif-xrt
