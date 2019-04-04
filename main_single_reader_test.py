from allenel.dataset_readers import EnityLinknigDatasetReader

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


reader = EnityLinknigDatasetReader(resource_path="/home/junkyul/conda/neural-el_resources")
file_path = "/home/junkyul/conda/allenel/tests/fixtures/train.mens.0"
train_data = list(reader.read(file_path))

print("read")

exit()

