from typing import Dict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.multiprocess_dataset_reader import MultiprocessDatasetReader
from array import ArrayType

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("el_multi_reader")
class EnityLinknigDatasetMultiReader(MultiprocessDatasetReader):
    crosswiki : Dict[str, tuple] = {}

    def __init__(self,
                 base_reader: DatasetReader,
                 resource_path: str,
                 num_workers: int,
                 epochs_per_read: int = 1,
                 output_queue_size: int = 1000,
                 ) -> None:
        super().__init__(base_reader, num_workers, epochs_per_read, output_queue_size)

        import pickle, os
        logger.info("reading crosswikis.pruned.pkl")
        EnityLinknigDatasetMultiReader.crosswiki = pickle.load(open(os.path.join(resource_path, "crosswikis.pruned.pkl"), "rb"))
        logger.info("end reading crosswikis.pruned.pkl")



