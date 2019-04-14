from allenel.dataset_readers import EnityLinknigDatasetReader, EnityLinknigDatasetMultiReader

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

multi_reader = EnityLinknigDatasetMultiReader(
    base_reader= EnityLinknigDatasetReader(resource_path=""),
    resource_path="",
    num_workers=1
)
file_pattern = "/home/junkyul/conda/allenel/tests/fixtures/train.mens.short*"
for i, p in enumerate(multi_reader.read(file_pattern)):
    print(p)

