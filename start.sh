#!/bin/sh
set -e
: "${PORT:=8000}"
echo "[start.sh] biotech-api starting on port ${PORT}"

# Create supervisord main config fresh (no dependency on system one)
cat > /tmp/supervisord.conf << 'CFG'
[supervisord]
nodaemon=true
logfile=/dev/stdout
logfile_maxbytes=0
loglevel=info
user=root
pidfile=/tmp/supervisord.pid

[unix_http_server]
file=/tmp/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[include]
files = /etc/supervisor/conf.d/*.conf
CFG

exec /usr/bin/supervisord -c /tmp/supervisord.conf
