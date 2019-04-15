from ccg_nlpy import remote_pipeline
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

test_sentence = """
SOCCER-JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT.	
"""
spacy_tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
sentence_tokenized = spacy_tokenizer.split_words(test_sentence)
processed_sentence = " ".join([t.text for t in sentence_tokenized])


test_sentence2 = """
For example, given the sentence "Paris is the capital of France", the idea is to determine that "Paris" refers to the city of Paris and not to Paris Hilton or any other entity that could be referred as "Paris". NED is different from named entity recognition (NER) in that NER identifies the occurrence or mention of a named entity in text but it does not identify which specific entity it is.
"""
sentence_tokenized2 = spacy_tokenizer.split_words(test_sentence2)
processed_sentence2 = " ".join([t.text for t in sentence_tokenized2])

class tryPredictor():
    """
    >>> p = tryPredictor()
    >>> p._process_test_doc(processed_sentence)
    >>> mention_lines = p.convertSent2NerToMentionLines()
    """
    def __init__(self):
        self.pipeline = remote_pipeline.RemotePipeline(server_api='http://macniece.seas.upenn.edu:4001')

    def _process_test_doc(self, sentence_raw):
        """ taken from processTestDoc from neural-el project by Nitish Gupta """
        self.doctext = sentence_raw
        self.ccgdoc = self.pipeline.doc(self.doctext)
        self.doc_tokens = self.ccgdoc.get_tokens
        self.sent_end_token_indices = self.ccgdoc.get_sentence_end_token_indices
        self.sentences_tokenized = []
        for i in range(0, len(self.sent_end_token_indices)):
            start = self.sent_end_token_indices[i - 1] if i != 0 else 0
            end = self.sent_end_token_indices[i]
            sent_tokens = self.doc_tokens[start:end]
            self.sentences_tokenized.append(sent_tokens)

        # List of ner dicts from ccg pipeline
        self.ner_cons_list = []
        try:
            self.ner_cons_list = self.ccgdoc.get_ner_conll.cons_list
        except:
            print("NO NAMED ENTITIES IN THE DOC. EXITING")

        self.sentidx2ners = {}
        for ner in self.ner_cons_list:
            found = False
            # idx = sentIdx, j = sentEndTokenIdx
            for idx, j in enumerate(self.sent_end_token_indices):
                sent_start_token = self.sent_end_token_indices[idx - 1] \
                    if idx != 0 else 0
                # ner['end'] is the idx of the token after ner
                if ner['end'] < j:
                    if idx not in self.sentidx2ners:
                        self.sentidx2ners[idx] = []
                    ner['start'] = ner['start'] - sent_start_token
                    ner['end'] = ner['end'] - sent_start_token - 1
                    self.sentidx2ners[idx].append(
                        (self.sentences_tokenized[idx], ner))
                    found = True
                if found:
                    break

    def convertSent2NerToMentionLines(self):
        '''Convert NERs from document to list of mention strings'''
        mentions = []
        # Make Document Context String for whole document
        cohStr = ""
        for sent_idx, s_nerDicts in self.sentidx2ners.items():
            for s, ner in s_nerDicts:
                cohStr += ner['tokens'].replace(' ', '_') + ' '
        cohStr = cohStr.strip()

        for idx in range(0, len(self.sentences_tokenized)):
            if idx in self.sentidx2ners:
                sentence = ' '.join(self.sentences_tokenized[idx])
                s_nerDicts = self.sentidx2ners[idx]
                for s, ner in s_nerDicts:
                    mention = "%s\t%s\t%s" % ("unk_mid", "@@UNKNOWN@@", "unkWT")
                    mention = mention + str('\t') + str(ner['start'])
                    mention = mention + '\t' + str(ner['end'])
                    mention = mention + '\t' + str(ner['tokens'])
                    mention = mention + '\t' + sentence
                    mention = mention + '\t' + "@@UNKNOWN@@"
                    cur_coh = set(cohStr.split())
                    cur_coh.remove( ner['tokens'].replace(' ', '_'))
                    mention = mention + '\t' + " ".join(cur_coh)
                    mentions.append(mention)
        return mentions


p = tryPredictor()
p._process_test_doc(processed_sentence)
mention_lines = p.convertSent2NerToMentionLines()
# assert(mention_lines[0] == "unk_mid	unk_wid	unkWT	2	2	JAPAN	SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .	UNK_TYPES	JAPAN CHINA")
# assert(mention_lines[1] == "unk_mid	unk_wid	unkWT	7	7	CHINA	SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .	UNK_TYPES	JAPAN CHINA")
for line in mention_lines:
    print(line)

p._process_test_doc(processed_sentence2)
mention_lines = p.convertSent2NerToMentionLines()
for line in mention_lines:
    print(line)
