systemctl stop miobot.service
git fetch origin
git reset --hard origin/main
systemctl start miobot.service
systemctl status miobot.service