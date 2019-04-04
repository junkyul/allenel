from typing import Dict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.multiprocess_dataset_reader import MultiprocessDatasetReader
from array import ArrayType

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("el_multi_reader")
class EnityLinknigDatasetMultiReader(MultiprocessDatasetReader):
    wid_index: Dict[str, int]
    coherence_index: Dict[str, int]
    type_index : Dict[str, int]
    crosswiki_priors : Dict[str, tuple]
    glove_index: Dict[str, int]
    title_dict : Dict[str, str]
    glove_embedding : Dict[str, ArrayType]

    def __init__(self,
                 base_reader: DatasetReader,
                 resource_path: str,
                 num_workers: int,
                 epochs_per_read: int = 1,
                 output_queue_size: int = 1000,
                 ) -> None:
        super().__init__(base_reader, num_workers, epochs_per_read, output_queue_size)

        import pickle, os
        logger.info("reading knw_wid_vocab.pkl")
        kwn_wid_vocab= pickle.load( open(os.path.join(resource_path, "vocab/knwn_wid_vocab.pkl"), "rb") )
        logger.info("reading cohstringG9_vocab.pkl")
        conherence_vocab = pickle.load(open(os.path.join(resource_path, "vocab/cohstringG9_vocab.pkl"), "rb"))
        logger.info("reading label_vocab.pkl")
        type_vocab = pickle.load(open(os.path.join(resource_path, "vocab/label_vocab.pkl"), "rb"))
        logger.info("reading glove_word_vocab.pkl")
        glove_vocab = pickle.load(open(os.path.join(resource_path, "vocab/glove_word_vocab.pkl"), "rb"))
        EnityLinknigDatasetMultiReader.wid_index = kwn_wid_vocab[0]
        EnityLinknigDatasetMultiReader.coherence_index = conherence_vocab[0]
        EnityLinknigDatasetMultiReader.type_index = type_vocab[0]
        EnityLinknigDatasetMultiReader.glove_index = glove_vocab[0]
        logger.info("reading glove.pkl")
        EnityLinknigDatasetMultiReader.glove_embedding = pickle.load(open(os.path.join(resource_path, "glove.pkl"), "rb"))
        logger.info("reading crosswikis.pruned.pkl")
        EnityLinknigDatasetMultiReader.crosswiki_priors = pickle.load(open(os.path.join(resource_path, "crosswikis.pruned.pkl"), "rb"))
        logger.info("reading wid2Wikititle.pkl")
        EnityLinknigDatasetMultiReader.title_dict = pickle.load(open(os.path.join(resource_path,"vocab/wid2Wikititle.pkl"), "rb"))



