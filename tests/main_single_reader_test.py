from allenel.dataset_readers import EnityLinknigDatasetReader

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


reader = EnityLinknigDatasetReader(resource_path="")
file_path = "/home/junkyul/conda/allenel/tests/fixtures/train.mens.short1"
train_data = reader.read(file_path)
for data in train_data:
    print(data)

