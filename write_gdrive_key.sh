#!/bin/bash
cat <<EOF > $HOME/gdrive-key.json
${GDRIVE_SERVICE_ACCOUNT_JSON}
EOF