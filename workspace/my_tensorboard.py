#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from tensorboard.main import run_main
import os

if __name__ == '__main__':
    google_login = bool(os.environ.get('google_login', 'true'))
    if google_login:
        import os
        IS_COLAB_BACKEND = 'COLAB_GPU' in os.environ  # this is always set on Colab, the value is 0 or 1 depending on GPU presence
        if IS_COLAB_BACKEND:
          from google.colab import auth
          # Authenticates the Colab machine and also the TPU using your
          # credentials so that they can access your private GCS buckets.
          auth.authenticate_user()
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run_main())