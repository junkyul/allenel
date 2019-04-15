import json
import shutil
import sys
from allennlp.commands import main
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG, #DEBUG
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "/home/junkyul/conda/allenel/tests/fixtures/model.tar.gz",
    "/home/junkyul/conda/allenel/tests/fixtures/predict_input.json",
    "--include-package", "allenel",
    "--predictor", "el_linker",
]
main()