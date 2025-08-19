curl -LsSf https://astral.sh/uv/install.sh | sh  # Download uv
sudo cp /root/.local/bin/uv /usr/local/bin/uv
sudo cp /root/.local/bin/uvx /usr/local/bin/uvx
ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts  # Code copying
